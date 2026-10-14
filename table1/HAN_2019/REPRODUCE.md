# HAN reproduction usage

Read the root `REPRODUCTION_PROTOCOL_AND_PROMPTS.md` and `shared/README.md` first.
Core adaptations are detailed in `ADAPTATION.md`; all Table1 relations are preserved.
There is one method directory, five datasets, seeds1024–1028, and no author utils/model
path injection. Canonical data roots remain their original read-only locations.

Set the method cwd and namespace/environment once:

```bash
cd /home/chenzhichao/Research/other_exp/table1/HAN_2019
export PYTHONPATH=/home/chenzhichao/Research/other_exp
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
```

Sparse dense-formula/gradient validation:

```bash
/home/chenzhichao/miniconda3/envs/coe_baseline_han/bin/python -m reproduce.checks --device cuda:0
```

Single validation-only development task (400 epochs, no test evaluation):

```bash
/home/chenzhichao/miniconda3/envs/coe_baseline_han/bin/python -m reproduce.run --dataset acm --seed 1024 --device cuda:0 --phase dev --run-id dev_unique_id --candidate 0
```

Full background matrix entrypoint, use a new batch ID:

```bash
/home/chenzhichao/miniconda3/envs/coe_baseline_han/bin/python -u -m reproduce.run_matrix --run-id han_unique_batch --devices cuda:0 --phase formal --seeds 1024 1025 1026 1027 1028
```

The matrix executes 12 pre-registered candidates per dataset using only validation,
freezes the chosen config into `exp/formal_<batch>/frozen_configs/<dataset>.json`, then
starts each of its five formal seeds from fresh initialization. Total60 dev jobs+25
formal jobs; all stages sequential on physicalGPU1. No automatic final statistics,
LaTeX generation or paper edits. A task failure writes `FAILED.json` and preserves
its logs; it never silently substitutes a smaller graph or omits a candidate/seed.
An existing batch ID is rejected; resumed/manual retry policy must preserve old failures
and use mathematically identical seed/config. Controller source lock rejects edits
to any method/shared core file during the batch.

Independent checkpoint replay (works only with unchanged frozen sources):

```bash
/home/chenzhichao/miniconda3/envs/coe_baseline_han/bin/python -m reproduce.replay --job-dir exp/formal_han_unique_batch/acm/seed1024 --device cuda:0
```

Explicit later five-seed summary, only after the user requests statistics:

```bash
/home/chenzhichao/miniconda3/envs/coe_baseline_han/bin/python -m reproduce.summarize --run-root exp/formal_han_unique_batch --datasets acm dblp yelp amazon mag
```

Strict shared aggregation checks all five seeds, source/config/data identities,
and replay; sample standard deviation uses ddof1. This command produces JSON/CSV
and candidate table rows; it does not modify the paper.

## Monitoring and outputs

- Master controller stdout goes to `log/formal_<batch>/console.log` via launcher.
- Every subprocess has `log/formal_<batch>/tasks/dev_<dataset>_cNN.log`,
  `formal_<dataset>_seedNNNN.log`, or `replay_<dataset>_seedNNNN.log`.
- Each training job also writes its exact protocol path
  `log/<run_id>/<dataset>/seedNNNN/console.log` and `exp/.../training.jsonl`.
- `exp/formal_<batch>/status.json` (also `progress.json`) always identifies current task and counts.
- End markers: `COMPLETED.json` (awaiting user summary), or `FAILED.json`.
- Weights, source snapshots, predictions, HPO evidence and logs remain in method exp/log
  and are excluded from Git; only core code/config/docs/environment provenance are tracked.

`--output-root` must be exactly this method's `exp`; other locations are rejected.
Formal single tasks require the full valid frozen12-trial evidence. Smoke accepts
`--smoke-epochs`, while dev/formal reject shortened budgets. Test labels are only read
in final export/evaluation, never for loss, checkpoints or HPO choices.
