# MMDynamics fixed defaults

From `/home/chenzhichao/Research/other_exp`, use the existing coe interpreter.
No installation, author checkpoint or hyperparameter search is required.

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python -u table2/MMDynamics_2022/reproduce/preflight.py --run-id smoke_mmdynamics_defaults_20261015_a --device cuda:7
```

Preflight compares native loss/gradients and canonical identities, then trains
three full-batch updates and replays in a separate process for each dataset.
Use a fresh run ID if repeating. The preflight report is
`exp/smoke_mmdynamics_defaults_20261015_a/preflight.json`.

After that report passes and a GPU is free:

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python -u table2/MMDynamics_2022/reproduce/run_matrix.py --run-id formal_mmdynamics_defaults_20261015_a --gpu 7 --preflight-run table2/MMDynamics_2022/exp/smoke_mmdynamics_defaults_20261015_a
```

This serial queue binds physical GPU7 as cuda:0 for child processes; change
`--gpu` only before plan freeze. `--plan-only` freezes a ready plan without
starting training. An occupied GPU writes `awaiting_gpu`; rerun the identical
command after the GPU becomes free. Use tmux to persist a running queue.
Do not edit frozen source/configuration during a run. Historical job folders
are never overwritten; a worker failure is recorded for investigation.

Each fit saves canonical `log/<run>/<dataset>/seed<seed>/console.log` and
`exp/<run>/<dataset>/seed<seed>/` artifacts. The controller also saves
`log/<run>/pipeline.log`, per-worker logs and `exp/<run>/queue_state.json`.
Summary CSV/JSON/LaTeX rows are generated only after all 20 jobs and replays.

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python table2/MMDynamics_2022/reproduce/replay.py --job-dir table2/MMDynamics_2022/exp/formal_mmdynamics_defaults_20261015_a/iapr/seed1024 --device cuda:0
/home/chenzhichao/miniconda3/envs/coe/bin/python table2/MMDynamics_2022/reproduce/summarize.py --run-root table2/MMDynamics_2022/exp/formal_mmdynamics_defaults_20261015_a
```

See `ADAPTATION.md` and `configs/PARAMETER_SOURCES.json` for the full mechanism,
inclusive 2501-update budget, validation changes and parameter provenance.
