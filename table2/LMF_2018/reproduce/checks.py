"""Necessary literal-author/math/gradient/data witnesses, no parameter search."""
import argparse,ast,copy,importlib.util,json,math,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from shared.coe_benchmark import seed_everything,write_json
from shared.coe_benchmark.artifacts import environment_metadata
from .common import METHOD,METHOD_DIR,DATASETS,config_for,prepare,batches
from .model import make_model

def assert_close(a,b,**kwargs):torch.testing.assert_close(a,b,**kwargs)
def compare_gradient(y1,y2,params1,params2):
    assert_close(y1,y2,rtol=1e-5,atol=1e-7)
    a=torch.autograd.grad(y1.square().sum(),params1)
    b=torch.autograd.grad(y2.square().sum(),params2)
    for x,y in zip(a,b):assert_close(x,y,rtol=2e-5,atol=1e-7)
    return max(float((x-y).abs().max()) for x,y in zip(a,b))

def lmf(device,cfg):
    from .model import SubNet
    spec=importlib.util.spec_from_file_location('_isolated_lmf_author_oracle',METHOD_DIR/'code/model.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    ours=SubNet(7,16,0.3).to(device).eval();ref=native.SubNet(7,16,0.3).to(device).eval();ref.load_state_dict(ours.state_dict())
    x=torch.randn(4,7,device=device,requires_grad=True);xx=x.detach().clone().requires_grad_()
    subnet_delta=compare_gradient(ours(x),ref(xx),[x,*ours.parameters()],[xx,*ref.parameters()])
    ours=make_model([16]*3,3,cfg).to(device)
    ref=native.LMF((16,16,16),(16,16,16),16,(0.,0.,0.,0.5),3,cfg['rank']).to(device)
    ref.audio_subnet=nn.Identity();ref.video_subnet=nn.Identity();ref.text_subnet=nn.Identity()
    for dst,src in zip([ref.audio_factor,ref.video_factor,ref.text_factor,ref.fusion_weights,ref.fusion_bias],
                       [*ours.factors,ours.fusion_weights,ours.fusion_bias]):dst.data.copy_(src.data)
    h=[torch.randn(4,16,device=device,requires_grad=True) for _ in range(3)]
    hh=[x.detach().clone().requires_grad_() for x in h]
    ours_params=[*ours.factors,ours.fusion_weights,ours.fusion_bias]
    ref_params=[ref.audio_factor,ref.video_factor,ref.text_factor,ref.fusion_weights,ref.fusion_bias]
    fusion_delta=compare_gradient(ours.fuse(h),ref(*hh),h+ours_params,hh+ref_params)
    algebra=[]
    for views in (2,5):
        m=make_model([2]*views,3,{**cfg,'hidden':2,'rank':2}).to(device).double()
        hidden=[torch.randn(3,2,device=device,dtype=torch.float64,requires_grad=True) for _ in range(views)]
        z=[torch.cat((x.new_ones((3,1)),x),1) for x in hidden]
        # Explicit outer-product weight/input tensors, independent of fused code.
        outs=[]
        for batch in range(3):
            tensor=z[0][batch]
            for x in z[1:]:tensor=torch.outer(tensor.flatten(),x[batch]).flatten()
            classes=[]
            for c in range(3):
                weight=None
                for r in range(2):
                    part=m.factors[0][r,:,c]
                    for f in m.factors[1:]:part=torch.outer(part.flatten(),f[r,:,c]).flatten()
                    part=part*m.fusion_weights[0,r]
                    weight=part if weight is None else weight+part
                classes.append((tensor*weight).sum()+m.fusion_bias[0,c])
            outs.append(torch.stack(classes))
        literal=torch.stack(outs);fast=m.fuse(hidden)
        assert_close(fast,literal,rtol=1e-11,atol=1e-12)
        ps=hidden+[*m.factors,m.fusion_weights,m.fusion_bias]
        g1=torch.autograd.grad(fast.square().sum(),ps,retain_graph=True)
        g2=torch.autograd.grad(literal.square().sum(),ps)
        for a,b in zip(g1,g2):assert_close(a,b,rtol=1e-10,atol=1e-11)
        algebra.append(dict(views=views,max_abs=float((fast-literal).abs().max())))
    bn=SubNet(3,16,0.3).to(device).train()
    try:bn(torch.randn(1,3,device=device))
    except ValueError:singleton='native BatchNorm rejects B=1; merge tail, not an environment failure'
    else:raise AssertionError('Expected native BN singleton error')
    chunks=batches(torch.arange(17,device=device),16,merge_singleton=True)
    assert [len(c) for c in chunks]==[17] and torch.equal(torch.cat(chunks),torch.arange(17,device=device))
    bn(torch.randn(17,3,device=device)).sum().backward();bn.eval();bn(torch.randn(1,3,device=device))
    return dict(subnet_gradient_max_abs=subnet_delta,fusion_gradient_max_abs=fusion_delta,outer_product_oracles=algebra,batchnorm=singleton)

def qmf(device,cfg):
    from .model import QMF,History,rank_loss,make_aux,training_loss
    # Load only literal native class AST; never import raw-image/text dependencies.
    path=METHOD_DIR/'code/RGBD-scene-recognition/models/QMF.py'
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QMF')
    ns={'torch':torch,'nn':nn,'ImageEncoder':lambda args:nn.Identity()}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    args=types.SimpleNamespace(img_hidden_sz=4,num_image_embeds=1,hidden=[],n_classes=3,dropout=0.1)
    ref=ns['QMF'](args).to(device);ours=QMF([4,4],3,cfg).to(device)
    ref.clf_rgb[0].load_state_dict(ours.heads[0].state_dict());ref.clf_depth[0].load_state_dict(ours.heads[1].state_dict())
    x=[torch.randn(7,4,device=device,requires_grad=True) for _ in range(2)]
    xx=[i.detach().clone().requires_grad_() for i in x]
    refout=ref(*xx);out=ours.outputs(x)
    for a,b in zip([out[0],*out[1],*out[2]],refout):assert_close(a,b)
    grad=compare_gradient(out[0],refout[0],x+list(ours.heads.parameters()),xx+list(ref.clf_rgb.parameters())+list(ref.clf_depth.parameters()))
    # Literal native History and ranking function on the actual selected device.
    hp=METHOD_DIR/'code/RGBD-scene-recognition/utils/crl_utils.py'
    rt=METHOD_DIR/'code/RGBD-scene-recognition/train_QMF_nyud2.py'
    nodes=[n for n in ast.parse(hp.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='History']
    nodes += [n for n in ast.parse(rt.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='rank_loss']
    class DeviceFix(ast.NodeTransformer):
        def visit_Call(self,n):
            n=self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='cuda':
                return ast.Call(func=ast.Attribute(value=n.func.value,attr='to',ctx=ast.Load()),args=[ast.Name(id='_device',ctx=ast.Load())],keywords=[])
            return n
    tree=DeviceFix().visit(ast.Module(body=nodes,type_ignores=[]));ast.fix_missing_locations(tree)
    ns={'torch':torch,'nn':nn,'np':np,'_device':device};exec(compile(tree,'_native_history_oracle','exec'),ns)
    ids=torch.arange(7,device=device);ours_h=History(ids);ref_h=ns['History'](7)
    initial=rank_loss(torch.randn(7,1,device=device),ids,ours_h)
    assert float(initial)==0
    for _ in range(3):
        losses=torch.rand(7,device=device);conf=torch.randn(7,1,device=device)
        ours_h.update(ids,losses,conf);ref_h.correctness_update(ids,losses,conf.reshape(-1))
    np.testing.assert_array_equal(ours_h.correctness,ref_h.correctness)
    c=torch.randn(7,1,device=device,requires_grad=True);cc=c.detach().clone().requires_grad_()
    r1=rank_loss(c,ids,ours_h);r2=ns['rank_loss'](cc,ids,ref_h)
    assert_close(r1,r2);assert_close(torch.autograd.grad(r1,c)[0],torch.autograd.grad(r2,cc)[0])
    try:ours_h.update(torch.tensor([999],device=device),torch.tensor([1.],device=device),torch.tensor([1.],device=device))
    except ValueError:pass
    else:raise AssertionError('Nontrain History update accepted')
    # All 2/3/5 view objectives and history state round trips remain finite.
    for v in (2,3,5):
        m=QMF(list(range(2,2+v)),4,cfg).to(device)
        aux=make_aux(ids);views=[torch.randn(7,d,device=device) for d in range(2,2+v)]
        objective,_=training_loss(m,views,torch.arange(7,device=device)%4,ids,aux,cfg)
        assert torch.isfinite(objective);objective.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters())
        for h in aux['history']:
            loaded=History(ids);loaded.load_state_dict(h.state_dict());np.testing.assert_array_equal(h.correctness,loaded.correctness)
    return dict(literal_rgbd_forward_gradient_max_abs=grad,native_history_and_rank_gradient='matched',zero_spread_ties='finite rank0 margin0',train_only_history='enforced',views=[2,3,5])

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--run-id',default='dev_preflight_20261015_a');p.add_argument('--audit-data',action='store_true');a=p.parse_args()
    seed_everything(1024);cfg=config_for('esp');device=torch.device(a.device)
    evidence={'method':METHOD,'device':str(device),'environment':environment_metadata(),'executor_model':'current inherited session model; exact identifier not exposed','independent_reviewer':'not_run'}
    evidence['mechanism']=lmf(device,cfg) if METHOD=='LMF' else qmf(device,cfg)
    if a.audit_data:
        evidence['datasets']=[]
        for d in DATASETS:
            data,views,meta=prepare(d,'cpu')
            # Full features/identities/splits verified by public loader. Model
            # witness uses a few actual train samples and every input column/view.
            ids=data.indices('train')[:4]
            model=make_model(meta['input_dims'],data.classes,config_for(d)).to(device).train()
            scores=model([x[ids].to(device) for x in views]);loss=nn.functional.cross_entropy(scores,data.labels[ids].to(device));loss.backward()
            assert torch.isfinite(loss)
            evidence['datasets'].append(dict(dataset=d,nodes=data.nodes,classes=data.classes,input_dims=meta['input_dims'],
                splits={k:int(v.sum()) for k,v in data.masks.items()},adaptation=meta,identity=data.identity()))
    target=METHOD_DIR/'exp'/a.run_id/(('cuda' if device.type=='cuda' else 'cpu')+'_checks.json')
    if target.exists():raise FileExistsError(target)
    write_json(target,evidence);print('WITNESS PASS '+METHOD+' '+str(device),flush=True);print(json.dumps(evidence,default=str),flush=True)
if __name__=='__main__':main()
