# Fixed-default MRiemGNN commands

Run from this method directory. Set `PYTHONPATH=/home/chenzhichao/Research/other_exp`,
`COE_DATA_ROOT=/home/chenzhichao/Research/CoE2/data/CoE_dataset`, and
`COE_MULTIVIEW_DATA_ROOT=/home/chenzhichao/Research/CoE2/data/Multi-view datasets`
(quote the latter path's spaces). Use existing `/home/chenzhichao/miniconda3/envs/coe/bin/python`,
`PYTHONDONTWRITEBYTECODE=1`, and OMP/MKL/OPENBLAS_NUM_THREADS=2.

```bash
python -m reproduce.checks --device cpu
python -m reproduce.checks --device cuda:0
python -u -m reproduce.run --dataset acm --seed 1024 --device cuda:0 --phase smoke --run-id smoke_unique --smoke-epochs 3
python -u -m reproduce.run --dataset mag --seed 1024 --device cuda:0 --phase smoke --run-id smoke_full_mag_unique --smoke-epochs 1
python -m reproduce.replay --job-dir exp/smoke_full_mag_unique/mag/seed1024 --device cuda:0
python -u -m reproduce.run_matrix --run-id formal_mriemgnn_author_defaults_20261015_a --device cuda:0
python -m reproduce.summarize --run-root exp/formal_mriemgnn_author_defaults_20261015_a
```

`run.py` also supports direct script execution with the root PYTHONPATH. Fixed
formal seeds are 1024–1028; formal phase rejects shorter smoke budgets/config
overrides. Matrix runs in a detached tmux on a GPU verified free at launch.
The master `log/<run>/console.log` reports queue progress, while full per-epoch
logs are `log/<run>/<dataset>/seed<seed>/console.log`; replays have replay.log.
Status, checkpoint, full-node predictions, manifests/source snapshots and
training.jsonl stay under `exp/<run>/`. Derived input cache is exp/_cache.
Strict complete-five-seed summaries are automatically written as summary.json,
summary.csv and table_rows.tex after all 25 independent replays pass. No paper
file is edited. Frozen core sources/environment are checked before every job;
runtime artifacts are excluded from Git. Failed runs remain visible.
