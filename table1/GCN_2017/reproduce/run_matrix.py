"""Fixed author defaults -> 25 formal jobs -> new-model replay; no HPO."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from shared.coe_benchmark import SEEDS, plan_method, save_plan, write_json, recount_result
from shared.coe_benchmark.artifacts import source_manifest
from shared.coe_benchmark.protocol import json_digest
from gcn_baseline.common import METHOD_DIR, DATASETS, core_files, read_config


def event(kind, **fields):
    print(json.dumps(dict(event=kind, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **fields)), flush=True)


def main():
    parser = argparse.ArgumentParser(description='GCN author-default 25 formal jobs; no validation parameter search')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--phase', choices=['formal'], default='formal')
    parser.add_argument('--configuration-policy', choices=['author_default'], default='author_default')
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
    default_path = METHOD_DIR / 'configs/default.json'
    config = read_config(default_path)
    if config['epochs'] != 200 or config['early_stopping'] != 10:
        raise ValueError('Use official epochs200 and moving-mean early_stopping10')
    declared = dict(method='GCN', datasets=list(DATASETS), training_seeds=list(SEEDS),
                    configuration_policy='author_defaults_no_hpo_v1', fixed_config=config,
                    config_sha256=json_digest(config), config_source='configs/default.json',
                    author_default_source='code/gcn/train.py TF flags: hidden1=16, lr=.01, dropout=.5, weight_decay=5e-4, epochs=200, early_stopping=10',
                    source_manifest=initial_sources, total_development_jobs=0, total_formal_jobs=25,
                    automatic_summary=False, executor_model='inherited current session model',
                    model_id_status='not_exposed', Astra='not_run')
    write_json(root / 'batch_plan.json', declared)
    write_json(root / 'fixed_configuration.json', dict(config=config, config_sha256=json_digest(config),
               source='official tkipf/gcn train.py defaults; benchmark masks/union, author stopping, independent valAcc checkpoint selection',
               hyperparameter_search=False, shared_by_all_datasets=True))
    event('queue_started', run_id=batch, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
          python=sys.executable, **{k:v for k,v in declared.items() if k != 'source_manifest'})
    completed = 0

    def execute(argv, launcher_log):
        if source_manifest(core_files()) != initial_sources:
            raise RuntimeError('Core source changed after freeze; stopping without loosening hashes')
        launcher_log.parent.mkdir(parents=True, exist_ok=True)
        with launcher_log.open('w') as stream:
            process = subprocess.run(argv, cwd=METHOD_DIR, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Subprocess exit {process.returncode}: {argv}; see {launcher_log}')

    try:
        formal_id = 'formal_' + batch
        plan = plan_method('GCN', formal_id, devices=args.devices, seeds=SEEDS,
                           python=sys.executable, config_by_dataset={d:default_path for d in DATASETS})
        save_plan(plan)
        for index, job in enumerate(plan['jobs']):
            dataset, seed = job['dataset'], job['seed']
            job_dir = METHOD_DIR / 'exp' / formal_id / dataset / f'seed{seed}'
            event('formal_task_started', dataset=dataset, seed=seed, index=index+1, total=25,
                  hyperparameter_search=False, config_sha256=json_digest(config),
                  console_log=str(METHOD_DIR / 'log' / formal_id / dataset / f'seed{seed}/console.log'))
            write_json(root / 'status.json', dict(status='running', stage='formal',
                       dataset=dataset, seed=seed, completed_formal=completed,
                       configuration_policy='author_defaults_no_hpo_v1', hyperparameter_search=False))
            execute(job['argv'] + ['--threads', str(args.threads)],
                    logs / 'formal' / dataset / f'seed{seed}' / 'launcher.log')
            replay_argv = [sys.executable, str(METHOD_DIR / 'reproduce/replay.py'),
                           '--job-dir', str(job_dir), '--device', 'cuda:0', '--threads', str(args.threads)]
            execute(replay_argv, METHOD_DIR / 'log' / formal_id / dataset / f'seed{seed}/replay.log')
            checked = recount_result(job_dir)
            if not checked['result']['model_replay_verified']:
                raise ValueError('Checkpoint replay did not complete')
            (job_dir / 'COMPLETE').write_text('Fixed-default formal training, fresh-model replay and prediction recount passed.\n')
            completed += 1
            event('formal_task_replayed', dataset=dataset, seed=seed, completed_formal=completed,
                  test=checked['result']['test'])
        status = dict(status='complete_awaiting_user_statistics', stage='finished',
                      completed_development=0, completed_formal=25, hyperparameter_search=False,
                      configuration_policy='author_defaults_no_hpo_v1', replayed_formal=25,
                      formal_root=str(METHOD_DIR / 'exp' / formal_id), summary_generated=False)
        write_json(root / 'status.json', status)
        (root / 'COMPLETE').write_text('25 fixed-author-default formal runs and 25 replays finished. Statistics deferred.\n')
        event('queue_complete', **status)
    except Exception:
        failure = dict(status='failed', completed_formal=completed, hyperparameter_search=False,
                       traceback=traceback.format_exc())
        write_json(root / 'failure.json', failure)
        write_json(root / 'status.json', failure)
        (root / 'FAILED').write_text(failure['traceback'])
        event('queue_failed', **failure)
        raise


if __name__ == '__main__':
    main()
