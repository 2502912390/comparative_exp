"""Start the fixed GCN matrix under tmux; this process can exit immediately."""
import argparse
from pathlib import Path
import json
import shlex
import subprocess

METHOD_DIR = Path(__file__).resolve().parents[1]
PYTHON = '/home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    for value in (args.run_id, args.session):
        if not value.replace('_', '').replace('-', '').isalnum():
            parser.error('Use safe unique slugs')
    log_dir = METHOD_DIR / 'log' / args.run_id
    if log_dir.exists() or (METHOD_DIR / 'exp' / args.run_id).exists():
        raise FileExistsError('Fresh batch required; existing runs cannot be overwritten')
    log_dir.mkdir(parents=True)
    variables = dict(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                     PYTHONPATH='/home/chenzhichao/Research/other_exp',
                     COE_DATA_ROOT='/home/chenzhichao/Research/CoE2/data/CoE_dataset',
                     COE_MULTIVIEW_DATA_ROOT='/home/chenzhichao/Research/CoE2/data/Multi-view datasets',
                     CUBLAS_WORKSPACE_CONFIG=':4096:8')
    argv = ['env', *(name+'='+value for name,value in variables.items()),
            PYTHON, '-u', str(METHOD_DIR/'reproduce/run_matrix.py'),
            '--run-id', args.run_id, '--devices', 'cuda:0', '--seeds', '1024', '1025', '1026', '1027', '1028']
    shell_command = shlex.join(argv) + ' > ' + shlex.quote(str(log_dir/'console.log')) + ' 2>&1'
    command = ['/usr/bin/tmux', 'new-session', '-d', '-s', args.session,
               '-c', str(METHOD_DIR), shell_command]
    subprocess.run(command, check=True)
    record = dict(session=args.session, run_id=args.run_id, argv=command,
                  controller_log=str(log_dir/'console.log'),
                  status_path=str(METHOD_DIR/'exp'/args.run_id/'status.json'),
                  formal_root=str(METHOD_DIR/'exp'/('formal_'+args.run_id)))
    (log_dir/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
