"""Single fixed-default DMG run: unsupervised native objective then one linear head."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "reproduce"

import argparse,json,time,sys
from pathlib import Path
import torch
from torch import nn
from shared.coe_benchmark import (load_dataset,seed_everything,RunPaths,build_manifest,BestCheckpoint,load_best,export_result)
from shared.coe_benchmark.io import write_json
from .common import METHOD_DIR,DATASETS,resolve_config,args_for,prepare_graphs,source_files
from .model import GNNDAE,Measure_F,LogReg,drop_feature_record,update_S,training_step


def reconstruct(data,payload,device):
    args=args_for(payload['config'],data)
    adjs,_,_=prepare_graphs(data,args.sc,device)
    model=GNNDAE(args).to(device);model.load_state_dict(payload['model_state']);model.eval()
    head=LogReg(args.c_dim+args.p_dim,data.classes).to(device);head.load_state_dict(payload['head_state']);head.eval()
    features=[]
    for canonical,mask in zip(data.features,payload['feature_drop_masks']):
        if mask.dtype!=torch.bool or mask.shape!=(canonical.shape[1],): raise ValueError('Feature mask differs from canonical dimensions')
        x=canonical.to(device).clone();x[:,mask.to(device)]=0;features.append(x)
    if len(features)!=len(data.features):raise ValueError('Missing relation feature drop masks')
    with torch.no_grad():
        _,private=model.embed(features,adjs)
        embedding=torch.cat([payload['S'].to(device),sum(private)/args.num_view],1)
        scores=head(embedding)
    return scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--run-id',required=True);p.add_argument('--phase',choices=['formal','smoke','dev'],default='formal');p.add_argument('--output-root');p.add_argument('--config');p.add_argument('--smoke-iters',type=int,default=1);p.add_argument('--smoke-inner',type=int,default=1);p.add_argument('--smoke-probe',type=int,default=3)
    opts=p.parse_args()
    if opts.phase=='formal' and opts.seed not in range(1024,1029): raise ValueError('Formal seeds are 1024..1028')
    if opts.output_root and Path(opts.output_root).resolve()!=METHOD_DIR/'exp': raise ValueError('output-root must be this method exp')
    if opts.config and Path(opts.config).resolve()!=METHOD_DIR/'configs/author_defaults.json': raise ValueError('Only fixed author configuration is supported')
    seed_everything(opts.seed);torch.set_num_threads(2)
    config=resolve_config(opts.dataset)
    if opts.phase!='formal':config.update(num_iters=opts.smoke_iters,inner_epochs=opts.smoke_inner,test_epo=opts.smoke_probe,check_only=True)
    paths=RunPaths.create(METHOD_DIR,opts.run_id,opts.dataset,opts.seed,opts.phase)
    # The matrix redirects stdout; direct invocations also retain an independent method log.
    class Tee:
        def __init__(self,base,stream):self.base,self.stream=base,stream
        def write(self,s):self.base.write(s);self.stream.write(s);self.stream.flush()
        def flush(self):self.base.flush();self.stream.flush()
    console=(paths.log/'console.log').open('a',buffering=1);sys.stdout=Tee(sys.stdout,console);sys.stderr=Tee(sys.stderr,console)
    start=time.monotonic();data=load_dataset(opts.dataset);args=args_for(config,data)
    adjs,neighbors,adaptation=prepare_graphs(data,args.sc,opts.device)
    feature_records=[drop_feature_record(x.to(opts.device),args.feature_drop) for x in data.features]
    features=[x[0] for x in feature_records];feature_drop_masks=[x[1] for x in feature_records]
    model=GNNDAE(args).to(opts.device)
    measurements=nn.ModuleList([Measure_F(args.c_dim,args.p_dim,[args.phi_hidden_size]*args.phi_num_layers,[args.phi_hidden_size]*args.phi_num_layers).to(opts.device) for _ in features])
    groups=[dict(params=m.parameters(),lr=args.lr_max,weight_decay=args.weight_decay) for m in measurements]
    groups.append(dict(params=model.parameters(),lr=args.lr_min))
    optimizer=torch.optim.Adam(groups,lr=args.lr_min)
    head=LogReg(args.c_dim+args.p_dim,data.classes).to(opts.device)
    params=sum(x.numel() for m in (model,measurements,head) for x in m.parameters())
    author=json.loads((METHOD_DIR/'sources.json').read_text())['code_commit']
    manifest=build_manifest(data,paths,method='DMG',config=config,source_files=source_files(),adaptation=adaptation,parameter_count=params,device=opts.device,author_commit=author)
    print(json.dumps({'event':'start','dataset':opts.dataset,'seed':opts.seed,'config':config,'nodes':data.nodes,'views':len(features),'parameters':params}),flush=True)
    trajectory=(paths.exp/'training.jsonl').open('a',buffering=1)
    best_loss=1e9;cnt_wait=0
    pretrain_start=time.monotonic()
    for itr in range(1,args.num_iters+1):
        S=update_S(model,features,adjs,args.c_dim)
        for inner in range(args.inner_epochs):
            metrics=training_step(model,measurements,optimizer,S,features,adjs,neighbors,args,itr*inner)
        loss=metrics['loss']
        if loss<best_loss:best_loss=loss;cnt_wait=0
        elif loss>best_loss and itr>100:cnt_wait+=1
        event={'stage':'unsupervised','iteration':itr,'native_stop_count':cnt_wait,**metrics}
        trajectory.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
        if cnt_wait==args.patience:break
    pretrain_seconds=time.monotonic()-pretrain_start
    model.eval();measurements.eval()
    S=update_S(model,features,adjs,args.c_dim)
    with torch.no_grad():
        _,private=model.embed(features,adjs);embedding=torch.cat([S,sum(private)/args.num_view],1).detach()
    head_optimizer=torch.optim.Adam(head.parameters(),lr=args.test_lr)
    train,val=data.indices('train').to(opts.device),data.indices('val').to(opts.device)
    labels=data.labels.to(opts.device)
    checkpoint=BestCheckpoint(paths.exp/'best.pt',patience_checks=0)
    fixed_payload={'model_state':model.state_dict(),'measurements_state':measurements.state_dict(),'pretrain_optimizer_state':optimizer.state_dict(),'S':S,'feature_drop_masks':feature_drop_masks,'config':config,'config_sha256':manifest['config_sha256'],'pretrain_iteration':itr}
    for epoch in range(1,args.test_epo+1):
        head.train();head_optimizer.zero_grad();loss=nn.functional.cross_entropy(head(embedding[train]),labels[train]);loss.backward();head_optimizer.step();head.eval()
        with torch.no_grad():val_acc=float((head(embedding[val]).argmax(1)==labels[val]).float().mean())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite linear probe CE')
        if val_acc>checkpoint.best_acc:
            checkpoint.consider(epoch,val_acc,{**fixed_payload,'head_state':head.state_dict(),'head_optimizer_state':head_optimizer.state_dict()})
        else:checkpoint.consider(epoch,val_acc,{})
        event={'stage':'linear_probe','epoch':epoch,'train_ce':float(loss.detach()),'val_acc':val_acc,'best_epoch':checkpoint.best_epoch}
        trajectory.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
    trajectory.close()
    best=load_best(paths.exp/'best.pt');scores=reconstruct(data,best['payload'],opts.device)
    result=export_result(paths,data,checkpoint=paths.exp/'best.pt',best_epoch=best['epoch'],best_validation_acc=best['validation_acc'],node_ids=data.node_ids,predictions=scores.argmax(1),scores=scores,score_kind='logits',timing={'total_seconds':time.monotonic()-start,'pretrain_seconds':pretrain_seconds})
    write_json(paths.exp/'stages.json',{'pretrain_iterations':itr,'native_pretrain_patience':args.patience,'probe_epochs':args.test_epo,'probe_runs':1,'test_evaluations':1,'executor_model':'inherited_current_session_model; exact_id_not_exposed'})
    print(json.dumps({'event':'completed','result':result}),flush=True)

if __name__=='__main__':main()
