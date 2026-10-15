# Actual coe execution witness

- Existing `/home/chenzhichao/miniconda3/envs/coe` reused read-only. No clone, installation or package changes.
- CPU and CUDA literal released-model output/parameter/input-gradient, composite relation set, native CE/KL and geometric-helper inspection checks passed: `exp/dev_preflight_20261015_a/{cpu,gpu}_checks.json`; actual logs are `log/dev_preflight_20261015_a/{cpu,gpu}.log`.
- GPU asymmetric sparse/native logits matched; maximum parameter/input gradient difference 2.78e-17 on the small double-precision formula oracle. Actual dataset training and inference remain native FP32.
- All five canonical datasets passed source/split/label/class identity verification and full observed relation-pattern construction: `exp/dev_preflight_20261015_a/data_audit.json`.
- Complete ACM three-update short run and fresh-process replay: `exp/smoke_mriemgnn_acm_20261015_a/acm/seed1024/`.
- Complete MAG113919 nodes/four classes/two original relations, three native composite channels with 11205836 non-self-loop edges, hidden128/two layers/64 kernels; one full training update, validation, checkpoint export and fresh-process replay: `exp/smoke_mriemgnn_mag_20261015_a/mag/seed1024/`.
- Exact node predictions/ACC matched in both real-data replays. FP32 logit maximum differences were 4.76837158203125e-7 and were logged, not used as a stop condition.
- The released default never uses its curvature Parameters; their gradients are None in the literal author oracle as well. Unused Lorentz projection helper constraint passes. Its unused expmap default float-curvature call raises a native TypeError, recorded in the checks; this is not an execution dependency failure and is not wired into the model.
- PyG is absent in coe but unused by this method's actual classification path; optional conversion tools are not imported. No replacement dataset or reference checkpoint was loaded.
- Executor: inherited current session model, exact identifier not exposed. Independent reviewer/Astra: not_run; new agents/models are not authorized for this task.

Smoke snapshots precede these final provenance/environment records. Formal jobs freeze the final core/config/environment and restart all seeds from scratch.
