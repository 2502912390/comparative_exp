# Isolated GCN environment

Reference `coe` is read-only. Save a real `coe` official-TF import failure using
`reproduce/environment_probe.py` before cloning. Serialize conda cloning with
`/tmp/coe_baseline_env_clone.lock`, then clone to `coe_baseline_gcn` and do not
modify the base. No extra pip/conda packages are required by the PyTorch port.

`env_spec.json` records the fixed process-bound physical GPU0 and two CPU
threads. Its hash is SHA256 of sorted-key, no-whitespace JSON (first8 hex in
the root compute ledger). Actual package versions and executable are also
saved in every run's `environment.json`. This document describes the expected
stack; logged probes and real-data smoke provide the actual witness.

From `/home/chenzhichao/Research/other_exp/table1/GCN_2017`:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python reproduce/environment_probe.py --run-id smoke_coe_env_NEW_ID
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python reproduce/check.py --device cuda:0 --run-id equations_NEW_ID
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python reproduce/run.py --dataset acm --seed 1024 --device cuda:0 --phase smoke --run-id acm_NEW_ID --config configs/smoke_acm.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python reproduce/run.py --dataset mag --seed 1024 --device cuda:0 --phase smoke --run-id mag_NEW_ID --config configs/smoke_mag.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python reproduce/replay.py --job-dir exp/smoke_mag_NEW_ID/mag/seed1024 --device cuda:0
```

All NEW_ID placeholders must be replaced with unused IDs. The first command
intentionally records unavailable official TensorFlow; GPU witness must still
pass. Replay can also be applied to the ACM smoke directory. No source or base
environment mutation is part of these runtime commands.
