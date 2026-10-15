# Mowst implementation checks

- Fixed variant: Mowst-GCN (not Mowst*); author generic numeric defaults, no HPO.
- Base coe lacked the required PyG; clone added only torch-geometric2.5.3.
- CPU and native GPU checks passed: parser-default matching, directed union
  normalization/self-loops, author MC50 metric and RNG equivalence, alternating
  update boundaries, and non-training-label isolation.
- All five real datasets passed full-node two-update/two-round smoke training
  and independent fresh-model replay of all50 class predictions and routes
  per test node. These smoke metrics are not formal results.
- All MLP/GCN/gate parameters and BN buffers are restored together; no expert
  is pretrained or refit during replay. Unique nodes and MC observations are
  explicitly distinguished in saved predictions and counts.
- Original paper code, common benchmark, CoE2 reference, data and coe were not
  modified. Executor: current session GPT-6. Independent reviewer: not_run.

Evidence: [preflight](exp/smoke_mowst_native_20261015_a/preflight.json),
[native GPU checks](log/smoke_mowst_native_20261015_a/native_checks.log),
[parameter sources](configs/PARAMETER_SOURCES.json),
[package difference](environment/package_diff.json).
