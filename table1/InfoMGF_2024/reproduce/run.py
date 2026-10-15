"""Fixed-default InfoMGF training, val-only selection, once-only test."""
import argparse
import copy
import json
import time
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    __package__='reproduce'
import torch
import torch.nn.functional as F
from shared.coe_benchmark import (SEEDS,RunPaths,seed_everything,build_manifest,
    BestCheckpoint,ValidationSelector,load_best,load_dataset,export_result,write_json)
from .data import prepare
from .mechanism import InfoMGF,Classifier,pack_graph,unpack_graph

METHOD_DIR=Path(__file__).resolve().parents[1]
DATASETS=('acm','dblp','yelp','amazon','mag','esp','flickr','iapr','nus')


def core_files():
    return sorted([*METHOD_DIR.joinpath('reproduce').glob('*.py'),
                   *METHOD_DIR.joinpath('configs').glob('*.json'),
                   *METHOD_DIR.joinpath('environment').glob('*.json'),METHOD_DIR/'ADAPTATION.md',METHOD_DIR/'sources.json',
                   *METHOD_DIR.joinpath('code/InfoMGF').glob('*.py')])


def config_for(dataset):
    return json.loads((METHOD_DIR/'configs/author_defaults.json').read_text())[dataset]


def accuracy(logits,labels,mask):
    return float((logits[mask].argmax(1)==labels[mask]).float().mean())


def fit_classifier(x,g,labels,masks,cfg,trace,outer_epoch,smoke_epochs=None):
    head=Classifier(x.shape[1],int(labels.max())+1,cfg).to(x.device)
    opt=torch.optim.Adam(head.parameters(),lr=cfg['lr_cls'],weight_decay=cfg['w_decay_cls'])
    selector=ValidationSelector(cfg['patience_cls'])
    best=None
    limit=smoke_epochs or cfg['epochs_cls']
    for epoch in range(1,limit+1):
        head.train();opt.zero_grad()
        loss=F.nll_loss(F.log_softmax(head(x,g),dim=1)[masks['train']],labels[masks['train']])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite downstream loss')
        loss.backward();opt.step()
        if epoch%10==0 or (smoke_epochs and epoch==limit):
            head.eval()
            with torch.no_grad():val=accuracy(head(x,g),labels,masks['val'])
            trace({'stage':'classifier','encoder_epoch':outer_epoch,'epoch':epoch,'train_loss':float(loss),'val_acc':val})
            if selector.consider(epoch,val):
                best={'head_state':copy.deepcopy(head.state_dict()),'head_optimizer_state':copy.deepcopy(opt.state_dict()),'head_epoch':epoch}
            if selector.should_stop:break
    if best is None:raise RuntimeError('No downstream checkpoint evaluated under author schedule')
    return selector.best_acc,best


def train(args):
    if args.phase=='formal' and args.seed not in SEEDS:raise ValueError('Formal seed not registered')
    if args.output_root and Path(args.output_root).resolve()!=(METHOD_DIR/'exp').resolve():raise ValueError('output-root must be this method exp')
    if args.phase=='formal' and args.smoke_epochs:raise ValueError('Formal cannot shorten author budgets')
    seed_everything(args.seed)
    cfg=config_for(args.dataset)
    if args.config:
        supplied=json.loads(Path(args.config).read_text())
        if supplied!=cfg:raise ValueError('No configuration overrides or hyperparameter search allowed')
    data=load_dataset(args.dataset)
    job=RunPaths.create(METHOD_DIR,args.run_id,args.dataset,args.seed,args.phase)
    console=(job.log/'console.log').open('w',buffering=1)
    sys.stdout=sys.stderr=console
    start=time.time()
    x,adjs,contexts,meta=prepare(data,cfg,args.device)
    model=InfoMGF(x.shape[1],[f.shape[1] for f in contexts],cfg).to(args.device)
    with torch.random.fork_rng(devices=[]):
        head_template=Classifier(x.shape[1],len(data.class_mapping),cfg)
        parameter_count=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in head_template.parameters())
    del head_template
    manifest=build_manifest(data,job,method='InfoMGF',config=cfg,source_files=core_files(),
        adaptation=meta,parameter_count=parameter_count,device=args.device,
        author_commit=json.loads((METHOD_DIR/'sources.json').read_text())['code_commit'])
    labels=data.labels.to(args.device);masks={k:v.to(args.device) for k,v in data.masks.items()}
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['w_decay'])
    selector=BestCheckpoint(job.exp/'best.pt',patience_checks=0)
    tracefile=(job.exp/'training.jsonl').open('a',buffering=1)
    def trace(record):
        record['seconds']=time.time()-start
        tracefile.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
    trace({'stage':'start','dataset':args.dataset,'seed':args.seed,'config':cfg,'nodes':x.shape[0],'input_dim':x.shape[1],'relations':len(adjs),'parameters':parameter_count})
    limit=args.smoke_epochs or cfg['epochs']
    for epoch in range(1,limit+1):
        model.train();opt.zero_grad()
        loss=model.loss(x,contexts,adjs)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite InfoMGF loss')
        loss.backward();opt.step()
        trace({'stage':'unsupervised','epoch':epoch,'loss':float(loss)})
        if epoch%cfg['eval_freq']==0 or (args.smoke_epochs and epoch==limit):
            model.eval()
            with torch.no_grad():g=model.fused.graph(torch.cat(contexts+[x],dim=1))
            # A fresh native evaluation head per representation check, as official.
            # Never read test labels here; save head together with encoder and graph.
            val,head=fit_classifier(x,g,labels,masks,cfg,trace,epoch,args.smoke_epochs)
            selector.consider(epoch,val,{'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),
                **head,'graph_state':pack_graph(g),'config':cfg,'config_sha256':manifest['config_sha256'],
                'input_dim':x.shape[1],'view_dims':[f.shape[1] for f in contexts],
                'classes':len(data.class_mapping),'inference_chain':'canonical_features->saved_learned_graph->saved_native_gcn'})
            trace({'stage':'representation_validation','epoch':epoch,'val_acc':val,'best_epoch':selector.best_epoch,'best_val_acc':selector.best_acc})
    selected=load_best(job.exp/'best.pt')
    payload=selected['payload']
    write_json(job.exp/'learned_graph.json',{'sha256':payload['graph_state'].get('sha256'),
        'type':payload['graph_state']['type'],'nodes':payload['graph_state']['n'],
        'edges':payload['graph_state']['values'].numel(),
        'source':'InfoMGF selected fused learner, not canonical/CCoE model graph'})
    head=Classifier(x.shape[1],len(data.class_mapping),cfg).to(args.device)
    head.load_state_dict(payload['head_state'],strict=True);head.eval()
    g=unpack_graph(payload['graph_state'],args.device)
    with torch.no_grad():scores=head(x,g)
    export_result(job,data,checkpoint=job.exp/'best.pt',best_epoch=selected['epoch'],
        best_validation_acc=selected['validation_acc'],node_ids=data.node_ids,
        predictions=scores.argmax(1).cpu(),scores=scores.cpu(),score_kind='logits',timing={'total_seconds':time.time()-start})
    trace({'stage':'final_test','completed':True,'job_dir':str(job.exp)})
    tracefile.close()


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--phase',choices=['formal','smoke','dev'],default='formal')
    p.add_argument('--run-id',required=True);p.add_argument('--output-root');p.add_argument('--config')
    p.add_argument('--smoke-epochs',type=int)
    return p


if __name__=='__main__':train(parser().parse_args())
