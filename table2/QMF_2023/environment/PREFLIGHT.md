# QMF preflight evidence

Existing coe reused read-only. Spec hash bd1ef18a; actual Python 3.10.21, Torch 2.1.1+cu121, NumPy 1.26.4, SciPy 1.15.3. No dependency installs, upgrades or clone. Unused raw image/text packages are not imported. GPU6 is user-specified; existing other jobs remain in place, no eviction.

- CPU literal-author forward/gradient and core mechanism checks: exp/dev_preflight_20261015_a/cpu_checks.json.
- CUDA literal-author forward/gradient and all nine canonical source/split/label/class mapping checks: exp/dev_preflight_20261015_a/cuda_checks.json.
- Table2 ESP, Table3 five-view MV-ACM and unequal393/3/438 Prokaryotic complete two-epoch short training/export and separate-process checkpoint inference passed; exact node predictions/ACC, FP32 finite logits differences diagnostic only. These are necessary functional witnesses, not hyperparameter trials or formal results.
- Short-run artifact paths: exp/smoke_qmf_esp_20261015_a/esp/seed1024, exp/smoke_qmf_mv_acm_20261015_a/mv_acm/seed1024, exp/smoke_qmf_mv_prokaryotic_20261015_a/mv_prokaryotic/seed1024.
- These checks use the public feature/graph interface; no CCoE training, parameters, learned graphs or reference directory changes. Wikipedia test693 verified in nine-dataset audit. No data/label/mask assertions weakened.
- All recorded author source SHA256 hashes were compared against archived official commit fe6c4c6ef7cb23f0a89594ee413d485f1854268b and matched. Short-run snapshots precede the final environment documentation; formal jobs start from scratch with the final frozen source set.

Native QMF RGBD forward/gradients, History cumulative losses, rolled pair target/margins and ranking gradients matched. The literal author History expects CPU DataLoader indices; an initial GPU-only oracle incorrectly passed CUDA IDs and failed. The fixture now follows that native calling convention; failed fixture log retained as cuda_native_oracle_cpu_indices.log. Actual adapter explicitly maps train IDs and accepts either device. This was a fixture issue, not a coe dependency failure. Zero-spread history returns tie0 rather than 0/0; invalid nontrain updates are rejected.

Executor: current inherited session model; exact identifier not exposed. Independent reviewer and agent-follows-doc: not_run (not authorized). Documented commands were executed by the current executor; no independent review claimed.
