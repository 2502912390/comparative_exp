"""InfoMGF official model/graph formulas with bounded-memory execution.

Source: zxlearningdeep/InfoMGF commit 27962685731a7cde96fb839c3fcf1b45e6cfca2f.
Dense and MAG sparse graph formulas are intentionally distinct.
"""
import itertools
import hashlib
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import dgl
import dgl.function as fn

EOS = 1e-10


def sparse_tensor(rows, cols, values, n):
    return torch.sparse_coo_tensor(torch.stack((rows, cols)), values, (n, n)).coalesce()


def topk_edges(embedding, k, block=1000):
    """Exact all-node cosine top-k; discrete indices have no gradient upstream.

    Recompute selected dot products after topology selection, which retains the
    original top-k gradient including both endpoints without retaining B×N.
    """
    x = F.normalize(embedding, dim=1, p=2)
    n = x.shape[0]
    indices = []
    with torch.no_grad():
        for start in range(0, n, block):
            indices.append((x[start:start+block] @ x.T).topk(k+1, dim=-1).indices)
    cols = torch.cat(indices).flatten()
    rows = torch.arange(n, device=x.device).repeat_interleave(k+1)
    # Bound endpoint-gather memory on MAG as well.
    values = []
    for start in range(0, rows.numel(), 65536):
        r, c = rows[start:start+65536], cols[start:start+65536]
        if torch.is_grad_enabled() and x.requires_grad:
            values.append(checkpoint(lambda a, r, c: (a[r]*a[c]).sum(1), x, r, c,
                                     use_reentrant=False))
        else:
            values.append((x[r]*x[c]).sum(1))
    return rows, cols, torch.cat(values)


def normalize_sparse(a):
    a = a.coalesce()
    r, c = a.indices()
    degree = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device).index_add(0, r, a.values())
    inv = 1/(degree.sqrt()+EOS)
    return sparse_tensor(r, c, a.values()*inv[r]*inv[c], a.shape[0])


def learned_graph(embedding, k, official_sparse, dropout, training, block=1000):
    r, c, v = topk_edges(embedding, k, block)
    n = embedding.shape[0]
    if official_sparse:
        # knn_fast norm = outgoing signed sum + incoming signed sum BEFORE ReLU.
        norm = torch.zeros(n, device=v.device, dtype=v.dtype).index_add(0, r, v).index_add(0, c, v)
        v = v*(norm[r].pow(-0.5)*norm[c].pow(-0.5))
        vals = F.dropout(F.relu(torch.cat((v, v))), p=dropout, training=training)
        # Preserve duplicate edges and their independent dropout from DGL.
        g = dgl.graph((torch.cat((r,c)), torch.cat((c,r))), num_nodes=n, device=v.device)
        g.edata['w'] = vals
        return g
    a = sparse_tensor(torch.cat((r,c)), torch.cat((c,r)), torch.cat((v,v))*0.5, n)
    a = sparse_tensor(*a.indices(), F.relu(a.values()), n)
    a = normalize_sparse(a)
    # Original dense dropout independently masks all nonzero directed entries.
    return sparse_tensor(*a.indices(), F.dropout(a.values(), p=dropout, training=training), n)


def graph_mm(a, x):
    if torch.is_tensor(a):
        return torch.sparse.mm(a, x)
    with a.local_scope():
        a.ndata['h'] = x
        a.update_all(fn.u_mul_e('h','w','m'), fn.sum('m','h'))
        return a.ndata['h']


def graph_dropout(a, p, training, preserve_diagonal=False, official_sparse=False):
    a = a.coalesce()
    r,c = a.indices()
    v = a.values()
    if preserve_diagonal:
        off = r != c
        idx = torch.cat((a.indices()[:,off], a.indices()[:,~off]), 1)
        val = torch.cat((F.dropout(v[off], p=p, training=training), v[~off]))
    else:
        idx = a.indices()
        val = F.dropout(v,p=p,training=training)
    if official_sparse:
        # Official torch_sparse_to_dgl_graph uses rows as source (transpose).
        g=dgl.graph((idx[0],idx[1]), num_nodes=a.shape[0], device=a.device)
        g.edata['w']=val.detach()
        return g
    return torch.sparse_coo_tensor(idx,val,a.shape).coalesce()


class Attentive(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w=nn.Parameter(torch.ones(width))
    def forward(self,x):
        return x*self.w  # exact x @ diag(w)


class Learner(nn.Module):
    def __init__(self,width,cfg):
        super().__init__()
        self.layers=nn.ModuleList([Attentive(width),Attentive(width)])
        self.cfg=cfg
    def forward(self,x):
        x=self.layers[0](x)
        x=F.relu(x) if self.cfg['activation_learner']=='relu' else torch.tanh(x)
        return self.layers[1](x)
    def graph(self,x):
        return learned_graph(self(x),self.cfg['k'],self.cfg['sparse'],
                             self.cfg['dropedge_rate'],self.training,self.cfg['graph_block'])


class Encoder(nn.Module):
    def __init__(self,indim,hidden,outdim,nlayers,dropout,pre_dropout):
        super().__init__()
        widths=[indim]+[hidden]*(nlayers-1)+[outdim]
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(widths[:-1],widths[1:])])
        self.dropout,self.pre_dropout=dropout,pre_dropout
    def forward(self,x,a):
        if self.pre_dropout:
            x=F.dropout(x,p=self.dropout,training=self.training)
        for i,layer in enumerate(self.layers):
            x=graph_mm(a,layer(x))
            if i<len(self.layers)-1:
                x=F.dropout(F.relu(x),p=self.dropout,training=self.training)
        return x


def lower_bound(a,b,block=256):
    """Original symmetric exp-cosine InfoNCE, exact full denominator.

    No NxN identity/similarity stored; checkpoint recomputes row blocks for
    backward. EOS in cosine and denominator matches official model.py.
    """
    def directional(x,y):
        total=x.new_zeros(())
        for start in range(0,x.shape[0],block):
            def part(x,y,start=start):
                q=x[start:start+block]
                sim=(q@y.T)/(q.norm(dim=1)[:,None]*y.norm(dim=1)[None,:]+EOS)/0.2
                mat=sim.exp()
                p=mat[torch.arange(q.shape[0],device=x.device),torch.arange(start,start+q.shape[0],device=x.device)]
                return -(p/(mat.sum(1)+EOS)).log().sum()
            value=checkpoint(part,x,y,use_reentrant=False) if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad) else part(x,y)
            total=total+value
        return total/x.shape[0]
    return (directional(a,b)+directional(b,a))/2


class InfoMGF(nn.Module):
    def __init__(self,inputdim,viewdims,cfg):
        super().__init__()
        self.cfg,self.num_g=cfg,len(viewdims)
        self.specific=nn.ModuleList([Learner(d,cfg) for d in viewdims])
        self.fused=Learner(sum(viewdims)+inputdim,cfg)
        self.encoder=Encoder(inputdim,cfg['hidden_dim'],cfg['rep_dim'],cfg['nlayers'],cfg['dropout'],True)
        def proj():
            return nn.Sequential(nn.Linear(cfg['rep_dim'],cfg['proj_dim']),nn.ReLU(inplace=True),nn.Linear(cfg['proj_dim'],cfg['proj_dim']))
        self.proj_s=nn.ModuleList([proj() for _ in viewdims])
        self.proj_u=nn.ModuleList([proj() for _ in viewdims])
        self.proj_f=nn.ModuleList([proj() for _ in viewdims])
    def encode(self,x,g):
        return F.normalize(self.encoder(x,g),dim=1,p=2)
    def loss(self,x,contexts,adjs):
        graphs=[learner.graph(context) for learner,context in zip(self.specific,contexts)]
        fg=self.fused.graph(torch.cat(contexts+[x],dim=1))
        zs=[self.encode(x,g) for g in graphs]
        zf=self.encode(x,fg)
        augmented=[graph_dropout(a,self.cfg['dropedge_rate'],self.training,True,self.cfg['sparse']) for a in adjs]
        zu=[self.encode(x,g) for g in augmented]
        n=x.shape[0]
        if self.cfg['contrast_batch_size']:
            import random
            ids=list(range(n));random.shuffle(ids)
            batches=[ids[i:i+self.cfg['contrast_batch_size']] for i in range(0,n,self.cfg['contrast_batch_size'])]
        else:
            batches=[list(range(n))]
        total=x.new_zeros(())
        for batch in batches:
            ss=[p(z[batch]) for p,z in zip(self.proj_s,zs)]
            uu=[p(z[batch]) for p,z in zip(self.proj_u,zu)]
            lb=lambda a,b: lower_bound(a,b,self.cfg['contrast_block'])
            pairs=list(itertools.combinations(range(self.num_g),2))
            smi=sum(lb(ss[i],ss[j]) for i,j in pairs)/len(pairs)
            fused=sum(lb(self.proj_f[i](zf[batch]),ss[i]) for i in range(self.num_g))/self.num_g
            umi=sum(lb(ss[i],uu[i]) for i in range(self.num_g))/self.num_g
            total=total+(smi+fused+umi)*(len(batch)/n)
        return total


class Classifier(nn.Module):
    def __init__(self,inputdim,classes,cfg):
        super().__init__()
        self.cfg=cfg
        self.encoder=Encoder(inputdim,cfg['hidden_dim_cls'],classes,cfg['nlayers_cls'],cfg['dropout_cls'],False)
    def forward(self,x,g):
        if torch.is_tensor(g):
            g=graph_dropout(g,self.cfg['dropedge_cls'],self.training)
        else:
            local=g.local_var()
            local.edata['w']=F.dropout(g.edata['w'],p=self.cfg['dropedge_cls'],training=self.training)
            g=local
        return self.encoder(x,g)


def pack_graph(g):
    if torch.is_tensor(g):
        g=g.coalesce()
        state={'type':'coo','rows':g.indices()[0].cpu(),'cols':g.indices()[1].cpu(),'values':g.values().detach().cpu(),'n':g.shape[0]}
        state['sha256']=graph_state_sha256(state)
        return state
    r,c=g.edges()
    state={'type':'dgl','rows':r.cpu(),'cols':c.cpu(),'values':g.edata['w'].detach().cpu(),'n':g.num_nodes()}
    state['sha256']=graph_state_sha256(state)
    return state


def graph_state_sha256(state):
    h=hashlib.sha256(f"{state['type']}:{state['n']}".encode())
    for key in ['rows','cols','values']:
        value=state[key].detach().cpu().contiguous()
        h.update(str(value.dtype).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def unpack_graph(state,device):
    if 'sha256' in state and state['sha256']!=graph_state_sha256(state):
        raise ValueError('Frozen model graph SHA256 mismatch')
    r,c,v=(state[k].to(device) for k in ('rows','cols','values'))
    if state['type']=='coo':
        return sparse_tensor(r,c,v,state['n'])
    g=dgl.graph((r,c),num_nodes=state['n'],device=device)
    g.edata['w']=v
    return g
