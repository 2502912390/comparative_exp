"""Literal active native node-classification loss; no test labels enter it."""
import torch
import torch.nn.functional as F
def mutual_kl(z_euc,z_hyp,tau=2.0,index=None):
    if index is not None:z_euc,z_hyp=z_euc[index],z_hyp[index]
    log_euc=F.log_softmax(z_euc/tau,dim=-1);prob_euc=F.softmax(z_euc/tau,dim=-1)
    log_hyp=F.log_softmax(z_hyp/tau,dim=-1);prob_hyp=F.softmax(z_hyp/tau,dim=-1)
    return F.kl_div(log_euc,prob_hyp,reduction='batchmean')+F.kl_div(log_hyp,prob_euc,reduction='batchmean')
def class_weights(labels,train,classes,strength=1.):
    counts=torch.bincount(labels[train],minlength=classes).float()
    weight=counts.sum()/counts.clamp(min=1.);weight=weight/weight.mean()
    return 1.+strength*(weight-1.)
def objective(out,labels,train,config,weight):
    task=F.cross_entropy(out['logits'][train],labels[train],weight=weight,label_smoothing=config['label_smoothing'])
    if config['mutual_target']!='embeddings' or config['focal_gamma']!=0 or config['lambda_behavior']!=0:raise ValueError('Fixed native defaults expected')
    mutual=mutual_kl(out['emb_euc'],out['emb_hyp'],config['tau'],train)
    return task+config['lambda_mutual']*mutual,task,mutual
