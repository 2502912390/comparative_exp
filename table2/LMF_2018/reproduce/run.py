"""Fixed author configuration, train-only objective, validation best, final test."""
import argparse,json,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from shared.coe_benchmark import SEEDS,RunPaths,seed_everything,build_manifest,BestCheckpoint,load_best,export_result,write_json
from .common import METHOD_DIR,METHOD,DATASETS,config_for,source_files,prepare,batches,infer
from .model import make_model,make_aux,auxiliary_state,optimizer_groups

def train(args):
    if args.phase=='formal' and (args.seed not in SEEDS or args.smoke_epochs is not None):raise ValueError('Formal seeds and budget frozen')
    if args.output_root and Path(args.output_root).resolve()!=(METHOD_DIR/'exp').resolve():raise ValueError('Outputs must remain method-local')
    cfg=config_for(args.dataset)
    if args.config and json.loads(Path(args.config).read_text())!=cfg:raise ValueError('No configuration overrides or HPO')
    seed_everything(args.seed)
    job=RunPaths.create(METHOD_DIR,args.run_id,args.dataset,args.seed,args.phase)
    stream=(job.log/'console.log').open('x',buffering=1);sys.stdout=sys.stderr=stream
    begin=time.time();print(json.dumps(dict(stage='data_prepare_begin',dataset=args.dataset,seed=args.seed)),flush=True)
    data,features,adaptation=prepare(args.dataset,args.device)
    labels=data.labels.to(args.device);train_ids=data.indices('train').to(args.device);val_ids=data.indices('val').to(args.device)
    model=make_model(adaptation['input_dims'],data.classes,cfg).to(args.device)
    params=sum(p.numel() for p in model.parameters())
    adaptation.update(variant=cfg['variant'],encoder=cfg['encoder'],fusion=cfg['fusion'])
    manifest=build_manifest(data,job,method=METHOD,config=cfg,source_files=source_files(),adaptation=adaptation,
        parameter_count=params,device=args.device,author_commit=json.loads((METHOD_DIR/'sources.json').read_text())['code_commit'])
    optimizer=torch.optim.Adam(optimizer_groups(model,cfg),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=None
    if METHOD=='QMF':scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=cfg['lr_factor'],patience=cfg['lr_patience'])
    aux=make_aux(train_ids)
    selector=BestCheckpoint(job.exp/'best.pt',patience_checks=0)
    tracefile=(job.exp/'training.jsonl').open('x',buffering=1)
    def trace(obj):
        obj['seconds']=time.time()-begin;tracefile.write(json.dumps(obj)+'\n');print(json.dumps(obj),flush=True)
    trace(dict(stage='start',dataset=args.dataset,seed=args.seed,nodes=data.nodes,classes=data.classes,input_dims=adaptation['input_dims'],parameters=params,config=cfg))
    stop_best=float('inf');stop_bad=0;updates=0
    limit=args.smoke_epochs if args.smoke_epochs is not None else cfg['epochs']
    for epoch in range(1,limit+1):
        model.train();optimizer.zero_grad();total_loss=0.;seen=0
        chunks=batches(train_ids,cfg['batch_size'],shuffle=True,merge_singleton=METHOD=='LMF')
        # Complete every accumulation group, including the final short group.
        # Do not discard train samples/gradients on small datasets.
        for first in range(0,len(chunks),cfg['gradient_accumulation_steps']):
            group=chunks[first:first+cfg['gradient_accumulation_steps']]
            optimizer.zero_grad()
            for ix in group:
                from .model import training_loss
                loss,diagnostic=training_loss(model,[x[ix] for x in features],labels[ix],ix,aux,cfg)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
                (loss/len(group)).backward();total_loss+=float(loss.detach());seen+=len(ix)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
            optimizer.step();updates+=1
        val_scores=infer(model,features,val_ids,cfg['inference_batch_size'])
        val_acc=float((val_scores.argmax(1)==labels[val_ids]).double().mean())
        val_ce=float(F.cross_entropy(val_scores,labels[val_ids]))
        if scheduler:scheduler.step(val_acc)
        improved=selector.consider(epoch,val_acc,dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler else None,auxiliary_state=auxiliary_state(aux),
            config=cfg,config_sha256=manifest['config_sha256'],input_dims=adaptation['input_dims'],classes=data.classes,
            adaptation=adaptation,train_ids=data.indices('train'),inference_batch_size=cfg['inference_batch_size']))
        if METHOD=='LMF':
            if val_ce<stop_best:stop_best=val_ce;stop_bad=0
            else:stop_bad+=1
        trace(dict(stage='epoch',epoch=epoch,training_objective_sum=total_loss,train_seen=seen,optimizer_updates=updates,
                   val_acc=val_acc,val_ce=val_ce,best_epoch=selector.best_epoch,best_val_acc=selector.best_acc,learning_rates=[g['lr'] for g in optimizer.param_groups],**diagnostic))
        if (METHOD=='LMF' and stop_bad>=cfg['patience']) or (METHOD=='QMF' and selector.bad_checks>=cfg['patience']):break
    selected=load_best(job.exp/'best.pt');payload=selected['payload']
    model.load_state_dict(payload['model_state'],strict=True)
    scores=infer(model,features,data.node_ids.to(args.device),cfg['inference_batch_size']).cpu()
    export_result(job,data,checkpoint=job.exp/'best.pt',best_epoch=selected['epoch'],best_validation_acc=selected['validation_acc'],
                  node_ids=data.node_ids,predictions=scores.argmax(1),scores=scores,score_kind='logits',timing={'total_seconds':time.time()-begin})
    write_json(job.exp/'stages.json',dict(train_epochs=epoch,optimizer_updates=updates,train_labels_only=True,
        test_evaluations=1,precision='FP32',train_stop_rule=cfg['stopping_rule'],parameter_policy='author_default_no_hpo'))
    trace(dict(stage='final_test',completed=True,job_dir=str(job.exp)));tracefile.close()

def parser():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--phase',choices=['formal','smoke','dev'],default='formal');p.add_argument('--run-id',required=True)
    p.add_argument('--output-root');p.add_argument('--config');p.add_argument('--smoke-epochs',type=int)
    return p
if __name__=='__main__':train(parser().parse_args())
