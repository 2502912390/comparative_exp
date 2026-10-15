# Actual coe execution evidence

- Existing environment: `/home/chenzhichao/miniconda3/envs/coe`; no clone or package changes.
- CPU ACM short training and fresh replay: `exp/smoke_dmg_acm_cpu_20261015_a/acm/seed1024/`.
- CUDA 4 asymmetric dense/sparse forward and gradient, S, topology and native objective checks: `exp/smoke_dmg_checks_20261015_a/checks.json`.
- Full MAG 113919 nodes, two relations, native hidden256, one unsupervised update and three probe updates: `exp/smoke_dmg_mag_cuda_20261015_a/mag/seed1024/`.
- Fresh-process MAG replay restored canonical features with saved column masks, backbone, S and selected classifier. `replay.json` is verified; both exact predictions and supplied logits at the unchanged 1e-6 absolute tolerance passed.
- All five formal seeds retain fixed author defaults/recommended overrides. No tuning or extra probes were scheduled.
- Executor: current inherited session model, exact identifier not exposed. Independent reviewer: not_run. Astra: not_run.

Development smoke snapshots predate this final environment/documentation record. Formal jobs freeze the final core source/config/environment hashes before launch.
