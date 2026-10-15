"""Small exact official formula oracles for topology, gradients and kernels."""
import argparse
import json
import torch
import torch.nn.functional as F
from .mechanism import learned_graph,topk_edges,lower_bound,graph_mm,pack_graph,InfoMGF,Classifier,graph_dropout


def dense_oracle(x,k,sparse):
    x=F.normalize(x,dim=1,p=2);sim=x@x.T
    vals,idx=sim.topk(k+1,dim=-1)
    if not sparse:
        mask=torch.zeros_like(sim).scatter(1,idx,1)
        a=sim*mask;a=(a+a.T)/2;a=F.relu(a)
        inv=1/(a.sum(1).sqrt()+1e-10)
        return inv[:,None]*a*inv[None,:]
    n=x.shape[0];r=torch.arange(n,device=x.device).repeat_interleave(k+1);c=idx.flatten();v=vals.flatten()
    norm=torch.zeros(n,device=x.device,dtype=x.dtype).index_add(0,r,v).index_add(0,c,v)
    v=F.relu(v*norm[r].pow(-0.5)*norm[c].pow(-0.5))
    # Sparse DGL convention uses column receive, row send.
    a=torch.sparse_coo_tensor(torch.stack((torch.cat((c,r)),torch.cat((r,c)))),torch.cat((v,v)),(n,n)).coalesce()
    return a.to_dense()


def contrast_oracle(a,b):
    sim=(a@b.T)/(a.norm(dim=1)[:,None]*b.norm(dim=1)[None,:]+1e-10)/.2
    mat=sim.exp()
    return (-(mat.diagonal()/(mat.sum(1)+1e-10)).log().mean()-(mat.diagonal()/(mat.sum(0)+1e-10)).log().mean())/2


def check(device):
    torch.manual_seed(12)
    results={}
    indices=torch.tensor([[0,0,1,2,3],[0,1,2,0,3]],device=device)
    values=torch.tensor([.5,.2,.7,.3,1.],device=device,dtype=torch.float64)
    asymmetric=torch.sparse_coo_tensor(indices,values,(4,4)).coalesce()
    inp=torch.randn(4,3,device=device,dtype=torch.float64,requires_grad=True)
    for sparse in [False,True]:
        operator=graph_dropout(asymmetric,0,False,True,sparse)
        actual=graph_mm(operator,inp)
        expected=(asymmetric.to_dense().T if sparse else asymmetric.to_dense())@inp
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        g=torch.autograd.grad(actual.square().sum(),inp,retain_graph=True)[0]
        r=torch.autograd.grad(expected.square().sum(),inp,retain_graph=True)[0]
        torch.testing.assert_close(g,r,rtol=1e-12,atol=1e-12)
    results['asymmetric_provided_direction']='dense A, official DGL A.T; forward and gradient passed'
    for sparse in [False,True]:
        # Positive inputs match first-ReLU learner; exact cosine ties avoided.
        x=(torch.randn(13,5,device=device,dtype=torch.float64).abs()+.1).requires_grad_()
        g=learned_graph(x,3,sparse,0,False,4)
        y=graph_mm(g,torch.eye(13,device=device,dtype=x.dtype))
        oracle=dense_oracle(x,3,sparse)
        torch.testing.assert_close(y,oracle,rtol=1e-10,atol=1e-10)
        w=torch.randn_like(y)
        grad=torch.autograd.grad((y*w).sum(),x,retain_graph=True)[0]
        ref=torch.autograd.grad((oracle*w).sum(),x)[0]
        torch.testing.assert_close(grad,ref,rtol=1e-8,atol=1e-9)
        results['official_sparse' if sparse else 'official_dense']={'forward_max_error':float((y-oracle).abs().max()),'gradient_max_error':float((grad-ref).abs().max())}
    a=torch.randn(19,8,device=device,dtype=torch.float64,requires_grad=True)
    b=torch.randn_like(a,requires_grad=True)
    value=lower_bound(a,b,4);ref=contrast_oracle(a,b)
    torch.testing.assert_close(value,ref,rtol=1e-10,atol=1e-10)
    grad=torch.autograd.grad(value,(a,b),retain_graph=True);refgrad=torch.autograd.grad(ref,(a,b))
    for g,r in zip(grad,refgrad):torch.testing.assert_close(g,r,rtol=1e-9,atol=1e-10)
    results['contrast']={'loss':float(value),'gradient_max_error':max(float((g-r).abs().max()) for g,r in zip(grad,refgrad))}
    print(json.dumps({'device':device,'status':'passed','checks':results},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');args=p.parse_args();check(args.device)
