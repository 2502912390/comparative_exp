# Fixed default NodeFormer reproduction

Run from `/home/chenzhichao/Research/other_exp`. Read ADAPTATION.md and
configs/author_sources.json for exact source and graph-prior choices.

```bash
# Unique IDs required. Smoke never enters formal summary.
/home/chenzhichao/miniconda3/envs/coe_baseline_nodeformer/bin/python -m table1.NodeFormer_2022.reproduce.run --dataset acm --seed 1024 --device cuda:0 --phase smoke --run-id native_smoke_unique --max-epochs 2
/home/chenzhichao/miniconda3/envs/coe_baseline_nodeformer/bin/python -m table1.NodeFormer_2022.reproduce.replay --job-dir table1/NodeFormer_2022/exp/smoke_native_smoke_unique/acm/seed1024 --device cuda:0
# All 5 datasets x seeds 1024–1028; threaded process dispatch, no model agents.
/home/chenzhichao/miniconda3/envs/coe_baseline_nodeformer/bin/python -u -m table1.NodeFormer_2022.reproduce.run_matrix --run-id formal_defaults_unique --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4
/home/chenzhichao/miniconda3/envs/coe_baseline_nodeformer/bin/python -m table1.NodeFormer_2022.reproduce.summarize --run-root table1/NodeFormer_2022/exp/formal_defaults_unique
```

Every training process saves `log/<run>/<dataset>/seed<seed>/console.log`.
The matrix also runs separate-process checkpoint replay per job, updates
queue_status.json and summary.json/csv/table_rows.tex under exp/<run>.
Any failure is preserved with traceback, and missing jobs keep summary incomplete.
No hyperparameter searches, test-based checkpoint/config choices or dataset edits.

The initial v1 batch is incomplete due to official Gumbel exponent overflow.
The corrected batch is `formal_nodeformer_author_defaults_stable_20261015_v2`;
see ADAPTATION.md for the reproduced failure and equivalent stable arithmetic.
`bash table1/NodeFormer_2022/reproduce/launch_formal.sh` launches the corrected
25-job queue on GPUs0/1/2/3/7. It automatically fills the NodeFormer row and
source/seed records in `EXPERIMENT_RESULTS.md` only after all25 jobs pass the
complete-five-seed and independent replay gates. That file is written at the
user's explicit request; all other runtime artifacts stay in method log/exp.
