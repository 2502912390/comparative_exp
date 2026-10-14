import argparse
from pathlib import Path
import json
import subprocess
import sys
import torch
from shared.coe_benchmark.io import write_json

METHOD_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    log_dir = METHOD_DIR / 'log' / args.run_id
    exp_dir = METHOD_DIR / 'exp' / args.run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=False)
    commands = {
        'tensorflow_import': [sys.executable, '-c', 'import tensorflow as tf; print(tf.__version__); print(tf.app.flags)'],
        'official_models_import': [sys.executable, '-c', 'from gcn.models import GCN; print(GCN)'],
    }
    import os
    environment = dict(os.environ, PYTHONPATH=str(METHOD_DIR / 'code'), PYTHONDONTWRITEBYTECODE='1')
    checks = {}
    for name, argv in commands.items():
        result = subprocess.run(argv, cwd=METHOD_DIR / 'code', env=environment,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (log_dir / (name + '.log')).write_text(result.stdout)
        checks[name] = dict(argv=argv, exit_code=result.returncode, output=result.stdout)
    torch.manual_seed(1024)
    if not torch.cuda.is_available():
        raise RuntimeError('Probe requires actual GPU access; sandbox visibility failure is not an environment failure')
    x = torch.randn(8, 8, device='cuda:0', requires_grad=True)
    y = x @ x
    y.square().mean().backward()
    report = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                  cuda=torch.version.cuda, gpu_name=torch.cuda.get_device_name(),
                  torch_gpu_forward_backward='passed', checks=checks,
                  clone_trigger='Official TF1 imports fail; formula-preserving PyTorch port will run in isolated clone',
                  executor_model='inherited current session model; no model switch/subagent')
    write_json(exp_dir / 'environment_probe.json', report)
    print(json.dumps(report), flush=True)
    if all(check['exit_code'] == 0 for check in checks.values()):
        raise RuntimeError('Official imports succeed: do not clone without an actual compatibility failure')


if __name__ == '__main__':
    main()
