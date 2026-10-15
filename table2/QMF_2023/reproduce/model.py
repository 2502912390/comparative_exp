"""Native RGBD QMF energy/CE/history ranking with static-vector classifiers."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class QMF(nn.Module):
    def __init__(self,dims,classes,cfg):
        super().__init__()
        # Native hidden=[]: one classifier per precomputed representation.
        self.heads=nn.ModuleList([nn.Linear(d,classes) for d in dims])
        self.confidence_scale=cfg['confidence_scale']
    def outputs(self,views):
        if len(views)!=len(self.heads):raise ValueError('Every view required')
        logits=[head(x) for head,x in zip(self.heads,views)]
        confidence=[torch.logsumexp(z,dim=1,keepdim=True)*self.confidence_scale for z in logits]
        # RGBD native path propagates through quality weights (text path detaches).
        fused=sum(z*c for z,c in zip(logits,confidence))
        return fused,logits,confidence
    def forward(self,views):return self.outputs(views)[0]

class History:
    """Only train-local indices exist; zero spread gives equal rank/margin 0."""
    def __init__(self,train_ids):
        self.train_ids=train_ids.detach().cpu().numpy().astype(np.int64)
        self.lookup={int(n):i for i,n in enumerate(self.train_ids)}
        self.correctness=np.zeros(len(self.train_ids),dtype=np.float64)
        self.confidence=np.zeros(len(self.train_ids),dtype=np.float64)
    def positions(self,ids):
        try:return np.array([self.lookup[int(i)] for i in ids.detach().cpu().tolist()],dtype=np.int64)
        except KeyError as exc:raise ValueError('History cannot access nontrain node') from exc
    def target_margin(self,ids,other):
        first,second=self.positions(ids),self.positions(other)
        lo,hi=self.correctness.min(),self.correctness.max()
        if hi==lo:
            one=np.zeros(len(first));two=np.zeros(len(second))
        else:
            one=(self.correctness[first]-lo)/(hi-lo)
            two=(self.correctness[second]-lo)/(hi-lo)
        return (torch.as_tensor(np.sign(one-two),dtype=torch.float32,device=ids.device),
                torch.as_tensor(np.abs(one-two),dtype=torch.float32,device=ids.device))
    def update(self,ids,loss,confidence):
        pos=self.positions(ids)
        if len(np.unique(pos))!=len(pos):raise ValueError('Duplicate train IDs in history update')
        self.correctness[pos]+=loss.detach().cpu().numpy()
        self.confidence[pos]=confidence.detach().reshape(-1).cpu().numpy()
    def state_dict(self):
        return {'train_ids':self.train_ids.copy(),'correctness':self.correctness.copy(),'confidence':self.confidence.copy()}
    def load_state_dict(self,state):
        if not np.array_equal(self.train_ids,state['train_ids']):raise ValueError('History train identity changed')
        for key in ('correctness','confidence'):
            value=np.asarray(state[key],dtype=np.float64)
            if value.shape!=self.correctness.shape or not np.isfinite(value).all():raise ValueError('Invalid history state')
            setattr(self,key,value.copy())

def rank_loss(confidence,ids,history):
    c=confidence.reshape(-1)
    target,margin=history.target_margin(ids,torch.roll(ids,-1))
    nonzero=torch.where(target==0,torch.ones_like(target),target)
    shifted=torch.roll(c,-1)+margin/nonzero
    return F.margin_ranking_loss(c,shifted,-target,margin=0.0)

def make_model(dims,classes,cfg):return QMF(dims,classes,cfg)
def make_aux(train_ids):return {'train_ids':train_ids.cpu().clone(),'history':None}
def auxiliary_state(aux):return {'train_ids':aux['train_ids'],'history':[h.state_dict() for h in aux['history']]}
def restore_aux(state,train_ids):
    aux=make_aux(train_ids)
    if not torch.equal(aux['train_ids'],state['train_ids'].cpu()):raise ValueError('Saved train IDs mismatch')
    aux['history']=[History(train_ids) for _ in state['history']]
    for h,s in zip(aux['history'],state['history']):h.load_state_dict(s)
    return aux
def training_loss(model,views,labels,ids,aux,cfg):
    fused,logits,confidence=model.outputs(views)
    if aux['history'] is None:aux['history']=[History(aux['train_ids']) for _ in views]
    ce=F.cross_entropy(fused,labels)+sum(F.cross_entropy(z,labels) for z in logits)
    ranks=sum(rank_loss(c,ids,h) for c,h in zip(confidence,aux['history']))
    # Native RGBD ordering: rank against previous history, then accumulate CE.
    for z,c,h in zip(logits,confidence,aux['history']):h.update(ids,F.cross_entropy(z,labels,reduction='none'),c)
    return ce+cfg['ranking_weight']*ranks,{'ce':float(ce.detach()),'rank':float(ranks.detach())}
def optimizer_groups(model,cfg):return model.parameters()
