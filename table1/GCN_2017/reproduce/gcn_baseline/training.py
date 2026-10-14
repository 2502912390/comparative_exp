import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
import json
import time
import traceback
import torch
from shared.coe_benchmark import (
    RunPaths, seed_everything, BestCheckpoint, load_best, build_manifest,
    export_result, write_json, verify_model_replay, recount_result,
)
from .common import (METHOD_DIR, DATASETS, prepare, make_model, core_files,
                     read_config, configure_runtime, assert_source_snapshot)
from .model import TF1Adam


def log_event(event, **fields):
    print(json.dumps(dict(event=event, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **fields)), flush=True)


def execute(args, paths):
    started = time.monotonic()
    config = read_config(args.config)
    if args.phase == 'formal' and (config['epochs'] != 400 or config['patience_checks'] != 200):
        raise ValueError('Formal budget must be the frozen 400/200 protocol')
    seed_everything(args.seed)
    data, features, support, labels, masks, adaptation = prepare(args.dataset, args.device)
    model = make_model(config, data, args.device)
    optimizer = TF1Adam(model.parameters(), lr=config['learning_rate'])
    source = json.loads((METHOD_DIR / 'sources.json').read_text())
    if source['code_url'] != 'https://github.com/tkipf/gcn':
        raise ValueError('Unexpected author provenance')
    manifest = build_manifest(data, paths, method='GCN', config=config,
                              source_files=core_files(), adaptation=adaptation,
                              parameter_count=sum(p.numel() for p in model.parameters()),
                              device=args.device, author_commit=source['code_commit'])
    selection = BestCheckpoint(paths.exp / 'best.pt', config['patience_checks'])
    if torch.cuda.is_available() and str(args.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    log_event('training_started', dataset=args.dataset, seed=args.seed, phase=args.phase,
              nodes=data.nodes, edges=support._nnz(), config=config,
              parameters=manifest['parameters'], device=str(features.device),
              gpu_name=torch.cuda.get_device_name() if features.is_cuda else None,
              source_sha256=manifest['sources']['sha256'])
    train_start = time.monotonic()
    with (paths.exp / 'training.jsonl').open('w') as history:
        for epoch in range(1, config['epochs'] + 1):
            epoch_start = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, support)
            loss = model.loss(logits, labels, masks['train'], config['weight_decay'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite supervised objective at epoch {epoch}')
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite gradient at epoch {epoch}')
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_logits = model(features, support)
                val_acc = float((validation_logits[masks['val']].argmax(-1) == labels[masks['val']]).float().mean())
                train_acc = float((validation_logits[masks['train']].argmax(-1) == labels[masks['train']]).float().mean())
            saved = selection.consider(epoch, val_acc, {
                'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'config_sha256': manifest['config_sha256'],
                'sources_sha256': manifest['sources']['sha256'],
            })
            event = dict(epoch=epoch, train_loss=float(loss.detach()), train_acc=train_acc,
                         validation_acc=val_acc, checkpoint_saved=saved,
                         best_validation_acc=selection.best_acc, best_epoch=selection.best_epoch,
                         bad_checks=selection.bad_checks, seconds=time.monotonic()-epoch_start)
            history.write(json.dumps(event) + '\n')
            history.flush()
            log_event('epoch', **event)
            if selection.should_stop:
                log_event('early_stop', epoch=epoch, patience_checks=config['patience_checks'])
                break
    training_seconds = time.monotonic() - train_start
    checkpoint = load_best(paths.exp / 'best.pt')
    if checkpoint['payload']['config_sha256'] != manifest['config_sha256']:
        raise ValueError('Checkpoint/config identity mismatch')
    # Deliberately construct a fresh inference model, with no train-mode state.
    fresh = make_model(config, data, args.device)
    fresh.load_state_dict(checkpoint['payload']['model_state'], strict=True)
    fresh.eval()
    with torch.no_grad():
        scores = fresh(features, support)
    replay_val_acc = float((scores[masks['val']].argmax(-1) == labels[masks['val']]).float().mean())
    if replay_val_acc != selection.best_acc:
        raise ValueError('Restored checkpoint does not reproduce selected validation ACC')
    timing = dict(training_seconds=training_seconds, total_seconds=time.monotonic()-started,
                  epochs_completed=epoch, updates=epoch, best_epoch=selection.best_epoch,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated() if features.is_cuda else 0)
    if args.phase == 'dev':
        # No test predictions, test ACC, or export_result in development/HPO.
        record = dict(status='complete_development_validation_only', dataset=args.dataset,
                      seed=args.seed, phase='dev', best_epoch=selection.best_epoch,
                      best_validation_acc=selection.best_acc,
                      config_sha256=manifest['config_sha256'],
                      source_sha256=manifest['sources']['sha256'], timing=timing,
                      checkpoint_restored=True, test_evaluated=False)
        write_json(paths.exp / 'development.json', record)
        log_event('development_complete', **record)
    else:
        result = export_result(paths, data, checkpoint=paths.exp / 'best.pt',
                               best_epoch=selection.best_epoch,
                               best_validation_acc=selection.best_acc,
                               node_ids=data.node_ids, predictions=scores.argmax(-1),
                               scores=scores, score_kind='logits', timing=timing)
        log_event('single_seed_complete', test=result['test'], best_epoch=selection.best_epoch,
                  best_validation_acc=selection.best_acc, timing=timing,
                  replay_required=True)
    (paths.exp / 'TRAINING_COMPLETE').write_text('Training finished; formal runs still require replay.\n')


def run_cli():
    parser = argparse.ArgumentParser(description='Fixed-split, two-layer official GCN formula port')
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--phase', required=True, choices=('smoke', 'dev', 'formal'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', default=str(METHOD_DIR / 'exp'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--threads', default=2, type=int)
    args = parser.parse_args()
    if Path(args.output_root).resolve() != METHOD_DIR / 'exp':
        parser.error('output-root must be this method directory/exp')
    configure_runtime(args.threads)
    paths = RunPaths.create(METHOD_DIR, args.run_id, args.dataset, args.seed, args.phase)
    with (paths.log / 'console.log').open('w', buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                execute(args, paths)
            except Exception:
                failure = dict(status='failed', dataset=args.dataset, seed=args.seed,
                               phase=args.phase, traceback=traceback.format_exc())
                write_json(paths.exp / 'failure.json', failure)
                (paths.exp / 'FAILED').write_text(failure['traceback'])
                traceback.print_exc()
                raise


def replay_cli():
    parser = argparse.ArgumentParser(description='New inference from the selected GCN checkpoint')
    parser.add_argument('--job-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', default=2, type=int)
    args = parser.parse_args()
    configure_runtime(args.threads)
    job = args.job_dir.resolve()
    if METHOD_DIR / 'exp' not in job.parents:
        parser.error('Job must be inside this method exp directory')
    checked = recount_result(job)
    manifest = assert_source_snapshot(job)
    data, features, support, _, _, adaptation = prepare(manifest['dataset'], args.device)
    if adaptation != manifest['adaptation']:
        raise ValueError('Replay graph adaptation differs')
    model = make_model(manifest['config'], data, args.device)
    best = load_best(job / 'best.pt')
    if (best['payload']['config_sha256'] != manifest['config_sha256'] or
            best['payload']['sources_sha256'] != manifest['sources']['sha256']):
        raise ValueError('Checkpoint provenance differs')
    model.load_state_dict(best['payload']['model_state'], strict=True)
    model.eval()
    with torch.no_grad():
        scores = model(features, support)
    result = verify_model_replay(job, node_ids=data.node_ids, predictions=scores.argmax(-1), scores=scores)
    log_event('replay_verified', job_dir=str(job), test=checked['result']['test'], **result)

