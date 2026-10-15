"""Literal released-code oracles, with no author utils/model imports."""
import argparse,ast,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .common import METHOD_DIR,config_for
from .native.model import MRiemGNN
from .native.manifolds.euclidean import EuclideanManifold
from .native.manifolds.hyperboloid import HyperboloidManifold
from .data import composite_graphs
from .objective import mutual_kl,objective,class_weights

def official():
    ns={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'defaultdict':defaultdict,
        'EuclideanManifold':EuclideanManifold,'HyperboloidManifold':HyperboloidManifold}
    for filename,names in [('kernels.py',{'SpectroRiemannianKernel'}),
        ('layer_mriemgnn.py',{'MRiemGNNLayer'}),('model_mriemgnn.py',{'MRiemGNN'}),
        ('data.py',{'build_adj_dict'}),('train.py',{'mutual_kl','classification_loss'})]:
        path=METHOD_DIR/'code'/filename;tree=ast.parse(path.read_text())
        tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
        exec(compile(tree,str(path),'exec'),ns)
    return ns

def check(device):
    torch.set_num_threads(2);torch.manual_seed(1024);ns=official();report={}
    raw=[torch.tensor([[0,0,0,1,2,5],[1,1,2,2,0,4]]),torch.tensor([[0,2,3],[1,0,5]]),torch.empty(2,0,dtype=torch.long)]
    edges=torch.cat(raw,1);types=torch.cat([torch.full((a.shape[1],),i,dtype=torch.long) for i,a in enumerate(raw)])
    for reverse in [False,True]:
        got,patterns=composite_graphs(raw,7,reverse,True)
        ref,names=ns['build_adj_dict'](edges,types,3,True,['r0','r1','r2'],reverse)
        assert len(got)==len(ref)
        for i,(pattern,a) in enumerate(zip(patterns,got)):
            expect=sorted(map(tuple,ref[i].T.tolist()));actual=sorted(map(tuple,a.indices().T.tolist()))
            assert actual==expect
            assert names[i]=='+'.join(f'r{v}' for v in range(3) if pattern&(1<<v))
    report['composite_relation_sets_and_reverse_edges']='exact released build_adj_dict parity; all raw relations recoverable'
    n=7;h=5
    edges={0:torch.tensor([[0,0,0,1,2,5],[1,1,2,2,0,4]],device=device),
           1:torch.tensor([[0,2,3],[1,0,5]],device=device),2:torch.empty(2,0,dtype=torch.long,device=device)}
    explicit={};sparse={}
    for r,ix in edges.items():
        deg=torch.bincount(ix[0],minlength=n).clamp(min=1).double().unsqueeze(-1)
        explicit[r]=(ix,deg)
        sparse[r]=(torch.sparse_coo_tensor(ix,torch.ones(ix.shape[1],device=device,dtype=torch.float64),(n,n)).coalesce(),deg)
    kwargs=dict(in_dim=h,hid_dim=8,out_dim=3,relation_types=[0,1,2],spaces=['Euclidean','Hyperbolic'],
        num_layers=2,num_kernels=4,dropout=0.,relation_dropout=0.,kernel_learnable=True,input_skip=False)
    ref=ns['MRiemGNN'](**kwargs).double().to(device);got=MRiemGNN(**kwargs).double().to(device)
    got.load_state_dict(ref.state_dict());ref.eval();got.eval()
    x=torch.randn(n,h,device=device,dtype=torch.float64,requires_grad=True)
    xx=x.detach().clone().requires_grad_();a=ref(x,explicit);b=got(xx,sparse)
    for name in ['emb_euc','emb_hyp','fused','Euclidean','Hyperbolic','logits']:
        torch.testing.assert_close(a[name],b[name],atol=1e-10,rtol=1e-10)
    ix=torch.tensor([0,1,2,4],device=device)
    la=a['logits'].square().mean()+.01*ns['mutual_kl'](a['emb_euc'],a['emb_hyp'],2.,ix)
    lb=b['logits'].square().mean()+.01*mutual_kl(b['emb_euc'],b['emb_hyp'],2.,ix)
    pa=[(name,p) for name,p in ref.named_parameters() if p.requires_grad]
    pb=[(name,p) for name,p in got.named_parameters() if p.requires_grad]
    assert [name for name,_ in pa]==[name for name,_ in pb]
    ga=torch.autograd.grad(la,[x]+[p for _,p in pa],allow_unused=True)
    gb=torch.autograd.grad(lb,[xx]+[p for _,p in pb],allow_unused=True)
    errors=[];unused=[]
    for name,u,v in zip(['input']+[name for name,_ in pa],ga,gb):
        if u is None or v is None:
            assert u is None and v is None;unused.append(name);continue
        torch.testing.assert_close(u,v,atol=1e-9,rtol=1e-8)
        assert torch.isfinite(v).all();errors.append(float((u-v).abs().max()))
    curvature=[name for name,_ in pa if '.curvature.' in name]
    assert curvature and all(name in unused for name in curvature)
    report['native_full_model_sparse_parity']={'logits_max_abs':float((a['logits']-b['logits']).abs().max()),
        'input_parameter_gradient_max_abs':max(errors),'duplicate_edges_empty_relation_asymmetry':True}
    report['author_curvature_parameters_unused_in_default_forward']=curvature
    cfg=config_for('acm');y=torch.tensor([0,0,1,1,1,2,2],device=device)
    w=class_weights(y,ix,3).double()
    loss,ce,kl=objective(b,y,ix,cfg,w)
    want=ns['classification_loss'](b['logits'][ix],y[ix],weight=w)+cfg['lambda_mutual']*ns['mutual_kl'](b['emb_euc'],b['emb_hyp'],cfg['tau'],ix)
    torch.testing.assert_close(loss,want,atol=1e-12,rtol=1e-12)
    assert cfg['lambda_mutual']==.01 and cfg['hid_dim']==128 and not cfg['hyperbolic_proj']
    report['weighted_CE_plus_train_only_symmetric_KL']='literal released objective matches; lambda_mutual0.01 retained'
    manifold=HyperboloidManifold();p=manifold.proj(torch.randn(7,5,dtype=torch.float64))
    torch.testing.assert_close(-manifold.minkowski_dot(p,p),torch.ones(7,dtype=torch.float64),atol=1e-12,rtol=1e-12)
    report['unused_Lorentz_projection_constraint']='passed'
    try:manifold.expmap(torch.zeros(2,5),torch.zeros(2,5))
    except TypeError as exc:report['unused_author_expmap_helper_default_float_curvature_error']=str(exc)
    import importlib.util,scipy,sklearn
    report['environment']={'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
        'cuda_available':torch.cuda.is_available(),'pyg_present_but_unused':importlib.util.find_spec('torch_geometric') is not None}
    return {'status':'passed','device':device,'checks':report,'executor_model':'current inherited session model; exact identifier not exposed','independent_reviewer':'not_run'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path);o=p.parse_args()
    result=check(o.device)
    if o.output:o.output.parent.mkdir(parents=True,exist_ok=True);o.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
