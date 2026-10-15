"""Fixed native-default supervised MRiemGNN; no intermediate test evaluation."""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]));__package__='reproduce'
import argparse,json,sys,time
from pathlib import Path
import torch
from shared.coe_benchmark import load_dataset,seed_everything,RunPaths,build_manifest,BestCheckpoint,load_best,export_result
from shared.coe_benchmark.io import write_json
from .common import METHOD_DIR,DATASETS,config_for,create_model,source_files
from .data import prepare
from .objective import objective,class_weights

def inference(data,payload,device):
    cfg=payload['config'];x,adjs,_,meta=prepare(data,cfg,device,transform=payload['feature_transform'])
    if meta!=payload['adaptation']:raise ValueError('Replay input adapter or frozen statistics changed')
    model=create_model(data,cfg,len(adjs)).to(device)
    model.load_state_dict(payload['model_state'],strict=True);model.eval()
    with torch.no_grad():return model(x,adjs,beta=cfg['beta'])['logits']

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--run-id',required=True);p.add_argument('--phase',choices=['formal','smoke','dev'],default='formal')
    p.add_argument('--output-root');p.add_argument('--config');p.add_argument('--smoke-epochs',type=int)
    o=p.parse_args()
    if o.phase=='formal' and (o.seed not in range(1024,1029) or o.smoke_epochs is not None):raise ValueError('Formal fixed seeds and native budget required')
    if o.output_root and Path(o.output_root).resolve()!=METHOD_DIR/'exp':raise ValueError('Use this method exp only')
    if o.config and Path(o.config).resolve()!=METHOD_DIR/'configs/author_defaults.json':raise ValueError('Only fixed registered author defaults')
    torch.set_num_threads(2);seed_everything(o.seed)
    cfg=config_for(o.dataset)
    if o.phase!='formal':cfg.update(epochs=o.smoke_epochs or 3,phase_budget='necessary_smoke_only')
    job=RunPaths.create(METHOD_DIR,o.run_id,o.dataset,o.seed,o.phase)
    class Tee:
        def __init__(self,a,b):self.a,self.b=a,b
        def write(self,s):self.a.write(s);self.b.write(s);self.b.flush()
        def flush(self):self.a.flush();self.b.flush()
    log=(job.log/'console.log').open('a',buffering=1);sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    started=time.monotonic();data=load_dataset(o.dataset)
    expected={'acm':(3025,3),'dblp':(2957,4),'yelp':(2614,3),'amazon':(11944,2),'mag':(113919,4)}
    if (data.nodes,data.classes)!=expected[o.dataset]:raise ValueError('Wrong canonical dataset version')
    print(json.dumps({'event':'canonical_data_verified','dataset':o.dataset,'seed':o.seed,'nodes':data.nodes,'classes':data.classes,'raw_views':len(data.features)}),flush=True)
    x,adjs,transform,meta=prepare(data,cfg,o.device)
    model=create_model(data,cfg,len(adjs)).to(o.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    inventory={'total':sum(p.numel() for p in model.parameters()),'trainable_declared':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'fixed_kernel_parameters':sum(p.numel() for n,p in model.named_parameters() if '.kernels.' in n and not p.requires_grad),
        'note':'Native unused curvature, auxiliary and link-decoder parameters remain serialized; only node-classification branch is executed.'}
    author=json.loads((METHOD_DIR/'sources.json').read_text())['code_commit']
    manifest=build_manifest(data,job,method='MRiemGNN',config=cfg,source_files=source_files(),adaptation=meta,
        parameter_count=inventory['trainable_declared'],device=o.device,author_commit=author)
    write_json(job.exp/'parameter_inventory.json',inventory)
    labels=data.labels.to(o.device);train=data.indices('train').to(o.device);val=data.indices('val').to(o.device)
    weight=class_weights(labels,train,data.classes,cfg['class_weight_strength']) if cfg['class_weight']=='balanced' else None
    selector=BestCheckpoint(job.exp/'best.pt',patience_checks=cfg['patience'])
    history=(job.exp/'training.jsonl').open('a',buffering=1)
    print(json.dumps({'event':'start','config':cfg,'derived_relations':len(adjs),'edge_counts':meta['edge_counts'],'parameters':inventory}),flush=True)
    for epoch in range(1,cfg['epochs']+1):
        t=time.monotonic();model.train();optimizer.zero_grad()
        out=model(x,adjs,beta=cfg['beta']);loss,ce,kl=objective(out,labels,train,cfg,weight)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native MRiemGNN objective; do not change defaults or omit a term')
        loss.backward();optimizer.step();model.eval()
        with torch.no_grad():
            logits=model(x,adjs,beta=cfg['beta'])['logits']
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite MRiemGNN validation outputs')
            acc=float((logits[val].argmax(1)==labels[val]).float().mean())
        payload={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'config':cfg,
                 'config_sha256':manifest['config_sha256'],'feature_transform':transform,'adaptation':meta}
        selector.consider(epoch,acc,payload)
        event={'epoch':epoch,'train_loss':float(loss.detach()),'train_ce':float(ce.detach()),'train_mutual_kl':float(kl.detach()),
               'val_acc':acc,'best_epoch':selector.best_epoch,'bad_checks':selector.bad_checks,'epoch_seconds':time.monotonic()-t}
        history.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
        del out,logits,loss,ce,kl,payload
        if selector.should_stop:break
    history.close();best=load_best(job.exp/'best.pt')
    scores=inference(data,best['payload'],o.device)
    if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite final scores')
    result=export_result(job,data,checkpoint=job.exp/'best.pt',best_epoch=best['epoch'],best_validation_acc=best['validation_acc'],
        node_ids=data.node_ids,predictions=scores.argmax(1),scores=scores,score_kind='logits',timing={'total_seconds':time.monotonic()-started})
    write_json(job.exp/'stages.json',{'epochs_run':epoch,'max_epochs':cfg['epochs'],'native_patience':cfg['patience'],
        'test_evaluations':1,'dtype':'float32','no_hpo':True,'canonical_views':len(data.features),'derived_relations':len(adjs),
        'curvature_status':'declared by author but unused in the default forward; no invented geometric update',
        'executor_model':'current inherited session model; exact identifier not exposed','independent_reviewer':'not_run'})
    print(json.dumps({'event':'completed','result':result}),flush=True)
if __name__=='__main__':main()
