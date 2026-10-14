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
