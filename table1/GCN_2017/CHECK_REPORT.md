# FP64 evaluation repair — 2026-10-14

Current training staysFP32 in coe with the complete author defaults. Validation,
final export and replay use an independent FP64 GPU model copy, disclosed in
config and adaptation metadata; all input/source/split/graph hashes stay exact.
The prior nativeFP32 formal batch c was retained as FAILED after strict logit
replay failed (1.9e-6–2.9e-6 discrepancy, no changed class prediction). Enabling
strict algorithms did not solve native CSR accumulation nondeterminism.

A native16 FP64 sparse evaluation copy agrees with the independent dense
formula at1e-12. Evidence: log/smoke_coe_env_20261014_gcn/coe_native16_fp64_eval.log
and exp/smoke_coe_native16_fp64_eval_20261014_gcn/equations.json.
The original failed checkpoint also passed two separate-process FP64 inference
checks at the original1e-6 tolerance, with unchanged predictions; this is
diagnostic evidence only and does not change the failed original result.
See its log/fp64_first.log, log/fp64_second.log and exp/fp64_diagnosis.json under
the original formal ACM1024 directory. No full extra model was trained.

Replacement queue d reruns all25 seeds after commit; no rounding, relaxed
assertions, missing-score replay or reused failed result. Statistics remain
deferred. The first fresh ACM run is the full training/replay regression.

---

# Current author-default/coe checks — 2026-10-14

Active execution directly uses `/home/chenzhichao/miniconda3/envs/coe/bin/python`.
Native hidden16/lr.01/dropout.5/L2.0005/epochs200/early_stopping10 are identical
for all five datasets. There is no HPO and no package/environment change.
The original clone and earlier400/200/width256 evidence below are historical
revisions; they do not describe the current training policy.

Current coe checks **passed** on physicalGPU0 (RTX4090, torch2.1.1+cu121):

- Native16 sparse/dense forward and gradients; asymmetric official graph
  normalization; masked CE/first-layer half-L2 and TF1Adam NumPy update.
- Literal author early stop: zero-based epoch10 cannot stop; epoch11 can;
  mean is previous10 validation losses, and strict equality does not stop.
- Canonical ACM3025-node/2211607-edge relation-union data identity matches
  registry; one full native16 train update and validation loss including
  first-layer L2 completed. Test was not evaluated.

Evidence: `log/smoke_coe_env_20261014_gcn/coe_native16_author.log` and
`exp/smoke_coe_native16_author_20261014_gcn/equations.json`. Environment-spec
canonical hash: `6a6802fc`. Full formal ACM default train/replay will be the
first new queue task after core commit/freeze. The other24 run automatically;
final mean/std remains deferred until user confirmation.

---

## Historical revisions retained for provenance (superseded)

# Fixed-default policy correction — 2026-10-14

The user's requested policy is now active: native hidden16, lr=.01, dropout=.5,
first-layer L2=.0005 from official train.py; identical for all five datasets.
No HPO or 256-width change is active. The earlier HPO process group3110130
(controller3110133 plus its own child) was verified and terminated; old logs
and weights were preserved. `exp/gcn_table1_20261014_a/status.json` and
`cancellation.json` record `canceled_by_user_policy_change`.

The **new hidden16** asymmetric normalization, sparse/dense forward/gradients,
masked CE/half-L2 and TF1Adam NumPy checks all passed on physicalGPU0.
Evidence: `log/smoke_coe_env_20261014_gcn/equations_defaults.log` and
`exp/smoke_equations_defaults_20261014_gcn/equations.json`.

The graph implementation, environment and masks were unchanged. Previous
full-MAG sparse correctness evidence remains relevant to that operator; its
smoke scores are not new paper data. New complete ACM default training and
independent replay are the first task of the fresh formal queue, only after
commit/freeze and launch authorization. The queue then continues the other24
tasks; no automatic summary. New paths are in REPRODUCTION.md.

---

## Historical implementation checks (width256 smoke; not current defaults)

# GCN implementation checks — 2026-10-14

Executor: inherited current session model; exact model identifier is not exposed
by the tool interface. No model override, no Astra call, no nested agents.

All checks below passed on process-bound physical GPU0, NVIDIA RTX4090, in
`coe_baseline_gcn` (Python3.10.21, torch2.1.1+cu121). The original `coe` was
not modified. Its real TensorFlow and official GCN-model import checks both
failed with `ModuleNotFoundError: No module named 'tensorflow'`, while CUDA
forward/backward passed. Only then was the isolated environment cloned.
No packages were installed or upgraded; clone spec hash is `c341d9aa`.

- Official asymmetric SciPy normalization expression equals the sparse port.
- Sparse CSR and independent dense equation agree on forward logits and
  gradients for both learned matrices.
- Masked supervised CE and first-layer-only `0.5 * L2` agree with the official
  objective; TF1 Adam agrees with an independent NumPy update expression.
- Real ACM: 3025 nodes, 2211607 union edges, four complete updates, validation
  selected epoch4, checkpoint restored, predictions exported, separate-process
  new-model replay/logit comparison and prediction recount passed.
- Real MAG: **113919 nodes**, 11319755 union edges, one complete update and
  validation, best state restored, predictions exported, separate-process
  new-model replay/logit comparison and prediction recount passed. Graph stayed
  sparse CSR. Peak CUDA allocation: 1377182720 bytes (about1.28GiB).

Evidence remains untracked under this method:

| Check | Log | Artifact |
|---|---|---|
| Real `coe` imports and GPU witness | `log/smoke_coe_env_20261014_gcn/*import.log` | `exp/smoke_coe_env_20261014_gcn/environment_probe.json` |
| Environment clone | `log/smoke_coe_env_20261014_gcn/conda_clone.log` | Environment `coe_baseline_gcn` |
| Equation/gradient checks | `log/smoke_coe_env_20261014_gcn/equations.log` | `exp/smoke_equations_20261014_gcn/equations.json` |
| ACM training and replay | `log/smoke_acm_check_20261014_gcn/acm/seed1024/{console,replay}.log` | `exp/smoke_acm_check_20261014_gcn/acm/seed1024/` |
| Full MAG training and replay | `log/smoke_mag_check_20261014_gcn/mag/seed1024/{console,replay}.log` | `exp/smoke_mag_check_20261014_gcn/mag/seed1024/` |

These are implementation smoke checks and are not eligible paper numbers.
No formal baseline mean/std or candidate table row has been generated.
All five registered Table1 datasets will independently undergo canonical
identity verification during every real development/formal task. Background
stage and completion conventions are in [REPRODUCTION.md](REPRODUCTION.md).

Sparse Tensor deprecation/CSR beta notices were emitted by the installed
framework; no functional failure, dense graph fallback or relaxed assertion
was used. Cross-framework bitwise agreement is not claimed.
