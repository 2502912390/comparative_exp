"""Persistent sequential HPO -> freeze -> formal -> replay queue; no summaries."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback
from shared.coe_benchmark import SEEDS, plan_method, save_plan, write_json, recount_result
from shared.coe_benchmark.artifacts import source_manifest
from shared.coe_benchmark.protocol import json_digest, sha256_file
from gcn_baseline.common import METHOD_DIR, DATASETS, core_files, read_config, assert_source_snapshot


def event(kind, **fields):
    print(json.dumps(dict(event=kind, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **fields)), flush=True)


def main():
    parser = argparse.ArgumentParser(description='GCN full 12-candidate HPO + 25 formal jobs')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--phase', choices=['all'], default='all')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS or args.devices != ['cuda:0']:
        parser.error('Use the fixed five seeds and one process-bound cuda:0 GPU')
    batch = args.run_id
    if not batch.replace('_', '').replace('-', '').isalnum():
        parser.error('Unsafe run ID')
    root = METHOD_DIR / 'exp' / batch
    logs = METHOD_DIR / 'log' / batch
    if root.exists():
        raise FileExistsError('Use a new batch ID; old outputs cannot be overwritten')
    root.mkdir(parents=True)
    logs.mkdir(parents=True, exist_ok=True)
    initial_sources = source_manifest(core_files())
    candidates = json.loads((METHOD_DIR / 'configs/search_space.json').read_text())
    if len(candidates['candidates']) != 12:
        raise ValueError('Exactly 12 pre-registered candidates are required')
    declared = dict(method='GCN', datasets=list(DATASETS), training_seeds=list(SEEDS),
                    development_seed=1024, candidates_per_dataset=12,
                    source_manifest=initial_sources, total_development_jobs=60, total_formal_jobs=25,
                    automatic_summary=False, executor_model='inherited current session model')
    write_json(root / 'batch_plan.json', declared)
    write_json(root / 'status.json', dict(status='running', stage='development', completed=0))
    event('queue_started', run_id=batch, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
          python=sys.executable, **{k:v for k,v in declared.items() if k != 'source_manifest'})
    completed = 0

    def execute(argv, launcher_log):
        if source_manifest(core_files()) != initial_sources:
            raise RuntimeError('Core source changed after preregistration; stopping without loosening hashes')
        launcher_log.parent.mkdir(parents=True, exist_ok=True)
        with launcher_log.open('w') as stream:
            process = subprocess.run(argv, cwd=METHOD_DIR, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Subprocess exit {process.returncode}: {argv}; see {launcher_log}')

    frozen = {}
    try:
        for dataset in DATASETS:
            trials = []
            for order, item in enumerate(candidates['candidates']):
                candidate = item['id']
                config_path = METHOD_DIR / 'configs' / item['file']
                config = read_config(config_path)
                dev_run = f'dev_{batch}_{candidate}'
                argv = [sys.executable, str(METHOD_DIR / 'reproduce/run.py'),
                        '--dataset', dataset, '--seed', '1024', '--device', 'cuda:0',
                        '--phase', 'dev', '--run-id', dev_run, '--config', str(config_path),
                        '--threads', str(args.threads), '--output-root', str(METHOD_DIR / 'exp')]
                event('development_task_started', dataset=dataset, candidate=candidate,
                      completed=completed, console_log=str(METHOD_DIR / 'log' / dev_run / dataset / 'seed1024/console.log'))
                write_json(root / 'status.json', dict(status='running', stage='development',
                           dataset=dataset, candidate=candidate, completed=completed))
                execute(argv, logs / 'development' / dataset / candidate / 'launcher.log')
                job = METHOD_DIR / 'exp' / dev_run / dataset / 'seed1024'
                record = json.loads((job / 'development.json').read_text())
                manifest = assert_source_snapshot(job)
                if (record['status'] != 'complete_development_validation_only' or record['test_evaluated']
                        or record['config_sha256'] != json_digest(config)
                        or manifest['sources'] != initial_sources):
                    raise ValueError('Development record is invalid or contains inconsistent provenance')
                trial = dict(order=order, candidate=candidate, config=config,
                             config_sha256=json_digest(config), best_epoch=record['best_epoch'],
                             best_validation_acc=record['best_validation_acc'], job_dir=str(job))
                trials.append(trial)
                completed += 1
                event('development_task_complete', dataset=dataset, candidate=candidate,
                      best_validation_acc=record['best_validation_acc'], completed=completed)
            # max is stable: equal validation metrics retain registration order.
            chosen = max(trials, key=lambda record: record['best_validation_acc'])
            path = root / 'frozen_configs' / (dataset + '.json')
            write_json(path, chosen['config'])
            frozen[dataset] = path
            write_json(root / 'selection' / (dataset + '.json'), dict(
                dataset=dataset, selected_candidate=chosen['candidate'],
                selected_config_sha256=chosen['config_sha256'], selected_validation_acc=chosen['best_validation_acc'],
                tie_rule='highest_validation_acc_then_preregistration_order', test_evaluated=False,
                development_seed=1024, trials=trials))
            event('dataset_config_frozen', dataset=dataset, candidate=chosen['candidate'],
                  config_sha256=chosen['config_sha256'], validation_acc=chosen['best_validation_acc'])
        formal_id = 'formal_' + batch
        plan = plan_method('GCN', formal_id, devices=args.devices, seeds=SEEDS,
                           python=sys.executable, config_by_dataset=frozen)
        save_plan(plan)
        for index, job in enumerate(plan['jobs']):
            dataset, seed = job['dataset'], job['seed']
            job_dir = METHOD_DIR / 'exp' / formal_id / dataset / f'seed{seed}'
            event('formal_task_started', dataset=dataset, seed=seed, index=index+1, total=25,
                  console_log=str(METHOD_DIR / 'log' / formal_id / dataset / f'seed{seed}/console.log'))
            write_json(root / 'status.json', dict(status='running', stage='formal',
                       dataset=dataset, seed=seed, completed_formal=index, completed_development=completed))
            execute(job['argv'] + ['--threads', str(args.threads)],
                    logs / 'formal' / dataset / f'seed{seed}' / 'launcher.log')
            replay_argv = [sys.executable, str(METHOD_DIR / 'reproduce/replay.py'),
                           '--job-dir', str(job_dir), '--device', 'cuda:0', '--threads', str(args.threads)]
            execute(replay_argv, METHOD_DIR / 'log' / formal_id / dataset / f'seed{seed}/replay.log')
            checked = recount_result(job_dir)
            if not checked['result']['model_replay_verified']:
                raise ValueError('Checkpoint replay did not complete')
            (job_dir / 'COMPLETE').write_text('Formal training, fresh-model replay and prediction recount passed.\n')
            event('formal_task_replayed', dataset=dataset, seed=seed, completed_formal=index+1,
                  test=checked['result']['test'])
        status = dict(status='complete_awaiting_user_statistics', stage='finished',
                      completed_development=60, completed_formal=25,
                      replayed_formal=25, formal_root=str(METHOD_DIR / 'exp' / formal_id),
                      summary_generated=False)
        write_json(root / 'status.json', status)
        (root / 'COMPLETE').write_text('60 HPO + 25 formal + 25 checkpoint replays finished. Statistics deferred.\n')
        event('queue_complete', **status)
    except Exception:
        failure = dict(status='failed', completed_development=completed, traceback=traceback.format_exc())
        write_json(root / 'failure.json', failure)
        write_json(root / 'status.json', failure)
        (root / 'FAILED').write_text(failure['traceback'])
        event('queue_failed', **failure)
        raise


if __name__ == '__main__':
    main()
