# Run Mowst-GCN

Use coe_baseline_mowst (coe lacked PyG; the base environment was not modified).
Run from the other_exp repository root with the fixed default configuration.

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/chenzhichao/miniconda3/envs/coe_baseline_mowst/bin/python -u table1/Mowst_2024/reproduce/checks.py --device cuda:0
CUDA_VISIBLE_DEVICES=7 /home/chenzhichao/miniconda3/envs/coe_baseline_mowst/bin/python -u table1/Mowst_2024/reproduce/run.py --dataset acm --seed 1024 --device cuda:0 --phase smoke --run-id smoke_unique --config table1/Mowst_2024/configs/smoke.json
CUDA_VISIBLE_DEVICES=7 /home/chenzhichao/miniconda3/envs/coe_baseline_mowst/bin/python -u table1/Mowst_2024/reproduce/replay.py --job-dir table1/Mowst_2024/exp/smoke_unique/acm/seed1024 --device cuda:0
```

The complete background queue preregisters all 25 tasks and uses a single
fixed author-default configuration, with no ten-run/split loop or HPO:

```bash
/home/chenzhichao/miniconda3/envs/coe_baseline_mowst/bin/python -u table1/Mowst_2024/reproduce/run_matrix.py --run-id formal_mowst_defaults_unique --physical-gpu 7
```

Monitor log/<run_id>/pipeline.log or the individual dataset_seed.log files.
Weights and predictions use exp/<run_id>/<dataset>/seed<seed>. Both experts,
the learned gate, BN buffers and test RNG are in best.pt. pretrain_best.pt
also retains the selected strong-pretraining state. Every completed formal
job is replayed in a new process. The queue then writes summary.json,
summary.csv and table_rows.tex, refusing mixed/incomplete five-seed results.

Native Mowst accuracy averages 50 random hard-routing draws for each seed;
all 50 per-node predictions are saved. Recompute with this method's MC-aware
recount/summarize APIs, not the shared single-prediction recount command:

```bash
/home/chenzhichao/miniconda3/envs/coe_baseline_mowst/bin/python table1/Mowst_2024/reproduce/summarize.py --run-root table1/Mowst_2024/exp/formal_mowst_defaults_unique
```

The original wandb interface is not imported or called. The parameters and
their exact author sources are in configs/PARAMETER_SOURCES.json. No model
or agent was used for an independent reviewer verdict; it is not_run.
