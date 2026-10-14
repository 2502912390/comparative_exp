import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from shared.coe_benchmark import (RunPaths, BestCheckpoint, build_manifest, export_result,
    seed_everything, capture_rng, restore_rng, load_best, adaptation_metadata, verify_model_replay)
from shared.coe_benchmark.io import write_json
from .common import METHOD, DATASETS, setup, inputs, sources, read_config
from .model import build_model, freeze_projections


def run(args, paths):
    setup()
    seed_everything(args.seed)
    cfg = read_config(args.config)
    if args.phase == 'formal' and args.max_epochs is not None:
        raise ValueError('Formal budget cannot be overridden')
    if args.phase != 'formal':
        cfg = dict(cfg, epochs=args.max_epochs or 2)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    data, x, adjs = inputs(args.dataset, device)
    if cfg['rb_order'] > 1:
        raise ValueError('Higher relational orders need explicit graph preparation')
    model = build_model(x.shape[1], data.classes, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    manifest = build_manifest(data, paths, method='NodeFormer', config=cfg, source_files=sources(),
        adaptation=adaptation_metadata(data,'relation_union'),
        parameter_count=sum(p.numel() for p in model.parameters()), device=str(device),
        author_commit=json.loads((METHOD/'sources.json').read_text())['code_commit'])
    print(json.dumps({'event':'start', 'dataset':args.dataset, 'seed':args.seed,
        'nodes':data.nodes, 'edges':adjs[0].shape[1], 'config':cfg,
        'parameters':manifest['parameters']}, ensure_ascii=False), flush=True)
    y_train = data.labels[data.masks['train']].to(device)
    train_idx, val_idx = (data.indices(s).to(device) for s in ('train','val'))
    y_val = data.labels[data.masks['val']].to(device)
    selector = BestCheckpoint(paths.exp/'best.pt', patience_checks=0)
    started = time.monotonic()
    for epoch in range(1, cfg['epochs']+1):
        tick = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, links = model(x,adjs,cfg['tau'])
        supervised = F.nll_loss(F.log_softmax(logits[train_idx],dim=-1),y_train)
        edge_loss = torch.stack(links).mean()
        loss = supervised - cfg['lamda'] * edge_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss epoch={epoch}')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f'Nonfinite gradient epoch={epoch}')
        optimizer.step()
        del logits, links
        model.eval()
        inference_rng = capture_rng()
        with torch.no_grad():
            val_logits,_ = model(x,adjs,cfg['tau'])
            val_acc = float((val_logits[val_idx].argmax(-1)==y_val).float().mean())
        improved = selector.consider(epoch,val_acc,{
            'model_state':model.state_dict(), 'optimizer_state':optimizer.state_dict(),
            'config_sha256':manifest['config_sha256'], 'inference_rng':inference_rng,
            'validation_scores':val_logits[val_idx].detach().cpu(),
            'projection_policy':'query_dependent_official_at_selection_frozen_for_replay'})
        record = {'epoch':epoch,'loss':float(loss.detach()),'supervised_nll':float(supervised.detach()),
            'mean_link_log_probability':float(edge_loss.detach()),'validation_acc':val_acc,
            'best_epoch':selector.best_epoch,'best_validation_acc':selector.best_acc,
            'improved':improved,'seconds':time.monotonic()-tick}
        with (paths.exp/'training.jsonl').open('a') as trace:
            trace.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        del val_logits, loss, edge_loss, supervised
    best = load_best(paths.exp/'best.pt')
    model.load_state_dict(best['payload']['model_state'],strict=True)
    freeze_projections(model)
    model.eval()
    restore_rng(best['payload']['inference_rng'])
    with torch.no_grad():
        scores,_ = model(x,adjs,cfg['tau'])
    torch.testing.assert_close(scores[val_idx].cpu(),best['payload']['validation_scores'],atol=1e-6,rtol=0)
    scores = scores.cpu()
    result = export_result(paths,data,checkpoint=paths.exp/'best.pt',best_epoch=best['epoch'],
        best_validation_acc=best['validation_acc'],node_ids=data.node_ids,
        predictions=scores.argmax(-1),scores=scores,score_kind='logits',
        timing={'total_seconds':time.monotonic()-started, 'epochs_completed':cfg['epochs'],
            'stopping':'author_fixed_budget_no_early_stopping',
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0})
    del model, optimizer
    if device.type=='cuda':
        torch.cuda.empty_cache()
    # New model object and loaded saved projection state; never recycle exported predictions.
    replay = build_model(x.shape[1],data.classes,cfg).to(device)
    replay.load_state_dict(best['payload']['model_state'],strict=True)
    freeze_projections(replay)
    replay.eval()
    restore_rng(best['payload']['inference_rng'])
    with torch.no_grad():
        fresh,_ = replay(x,adjs,cfg['tau'])
    verify_model_replay(paths.exp,node_ids=data.node_ids,predictions=fresh.argmax(-1),scores=fresh)
    result=json.loads((paths.exp/'result.json').read_text())
    print(json.dumps({'event':'complete_replayed','result':result}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--phase',choices=['formal','smoke','dev'],default='formal')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output-root',type=Path,default=METHOD/'exp')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--max-epochs',type=int)
    args=parser.parse_args()
    if args.output_root.resolve()!=(METHOD/'exp').resolve():
        parser.error('Artifacts must stay in this method directory')
    paths=RunPaths.create(METHOD,args.run_id,args.dataset,args.seed,args.phase)
    with (paths.log/'console.log').open('w',buffering=1) as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                run(args,paths)
            except BaseException:
                write_json(paths.exp/'failure.json',{'status':'failed','traceback':traceback.format_exc()})
                traceback.print_exc()
                raise

if __name__=='__main__':
    main()
