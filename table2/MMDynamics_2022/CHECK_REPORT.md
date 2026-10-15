# MMDynamics implementation checks

Executor: current session model (Codex); no model switch or subagents.
Independent reviewer: not_run.

Actual host coe environment, GPU7; no package changes or environment clone.

Preflight: `exp/smoke_mmdynamics_defaults_20261015_a/preflight.json`.

Passed native two-view formula/logit checks, all parameter gradients and label-free infer.
Passed all four canonical source/label/class/split/feature identity checks.
Three full-batch updates on each complete real dataset, followed by separate-process
checkpoint reconstruction and exact class/ACC replay; maximum logit difference 0.
No smoke score is eligible for formal statistics. No hyperparameter candidates compared.

| Dataset | Nodes | Train / val / test | Parameters |
|---|---:|---|---:|
| esp | 11032 | 5514 / 2754 / 2764 | 252223 |
| flickr | 12154 | 6076 / 3037 / 3041 | 248220 |
| iapr | 7855 | 3926 / 1961 / 1968 | 248220 |
| nus | 20000 | 10000 / 5000 / 5000 | 256226 |
