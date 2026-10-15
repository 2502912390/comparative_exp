"""Official SubNet and LMF factor algebra, generalized to static V views."""
import torch
from torch import nn
from torch.nn import functional as F

class SubNet(nn.Module):
    def __init__(self,in_size,hidden_size,dropout):
        super().__init__()
        self.norm=nn.BatchNorm1d(in_size)
        self.drop=nn.Dropout(dropout)
        self.linear_1=nn.Linear(in_size,hidden_size)
        self.linear_2=nn.Linear(hidden_size,hidden_size)
        self.linear_3=nn.Linear(hidden_size,hidden_size)
    def forward(self,x):
        x=self.drop(self.norm(x))
        return F.relu(self.linear_3(F.relu(self.linear_2(F.relu(self.linear_1(x))))))

class LMF(nn.Module):
    def __init__(self,dims,classes,cfg):
        super().__init__()
        self.encoders=nn.ModuleList([SubNet(d,cfg['hidden'],cfg['dropout']) for d in dims])
        self.factors=nn.ParameterList([nn.Parameter(torch.empty(cfg['rank'],cfg['hidden']+1,classes)) for _ in dims])
        self.fusion_weights=nn.Parameter(torch.empty(1,cfg['rank']))
        self.fusion_bias=nn.Parameter(torch.zeros(1,classes))
        for p in [*self.factors,self.fusion_weights]:nn.init.xavier_normal_(p)
        # Official declares post_fusion_dropout but never executes it.
        self.post_fusion_dropout=nn.Dropout(cfg['post_fusion_dropout'])
    def fuse(self,hidden):
        product=None
        for h,factor in zip(hidden,self.factors):
            padded=torch.cat((h.new_ones((h.shape[0],1)),h),dim=1)
            projected=torch.matmul(padded,factor)
            product=projected if product is None else product*projected
        return (torch.matmul(self.fusion_weights,product.permute(1,0,2)).squeeze(1)+self.fusion_bias)
    def forward(self,views):
        if len(views)!=len(self.encoders):raise ValueError('Every view required')
        return self.fuse([enc(x) for enc,x in zip(self.encoders,views)])

def make_model(dims,classes,cfg):return LMF(dims,classes,cfg)
def make_aux(train_ids):return {}
def auxiliary_state(aux):return {}
def restore_aux(state,train_ids):return {}
def training_loss(model,views,labels,ids,aux,cfg):
    scores=model(views)
    loss=F.cross_entropy(scores,labels,reduction='sum')
    return loss,{'ce_sum':float(loss.detach())}
def optimizer_groups(model,cfg):
    factors=list(model.factors)
    factor_ids={id(p) for p in factors}
    return [{'params':factors,'lr':cfg['factor_lr']},
            {'params':[p for p in model.parameters() if id(p) not in factor_ids],'lr':cfg['lr']}]
