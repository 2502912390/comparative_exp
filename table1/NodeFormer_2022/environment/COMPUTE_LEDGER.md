# Local environment ledger

Environment `coe_baseline_nodeformer@bede1929` is cloned from read-only coe, Python
3.10.21, torch 2.1.1+cu121, CUDA 12.1. Spec: env_spec.json. Package installation
and clone logs: `log/dev_setup/`. Original coe import failed on required
`torch_sparse`; short training cannot pass that import. Only matched scatter,
sparse, PyG and their missing runtime dependencies were installed in the clone.
The exact delta is recorded in `exp/dev_setup/environment_delta.json`.

2026-10-14/15 verification: RTX 4090 CUDA kernel witness passed, including
forward/gradient comparison to unmodified official implementation in train/eval
with relational orders 0 and 1 and directed unequal-degree graph. Output max
error=0; link-loss max error <=2.39e-7; input-gradient max error <=1.17e-8.
Evidence: `log/dev_setup/equations_host.log`. The earlier sandbox invocation
could not access GPU; this was a sandbox restriction, not environment failure.

Real ACM 2 updates, validation-selected best, restored validation scores and
fresh-object replay passed at `exp/smoke_defaults_20261014_acm/acm/seed1024`.
Full MAG (113919 nodes, 11319755 union edges) 2 updates and the same checks
passed at `exp/smoke_defaults_20261014_mag/mag/seed1024`; peak CUDA allocation
5710887936 bytes. Separate-process replay on GPU4 (ACM) and GPU5 (MAG) both
returned `status=verified` on 2026-10-15, including score comparisons.
No smoke score was used to choose parameters.

Executor: current session model (exact model identifier is not exposed here).
Independent reviewer and agent-follows-doc: `not_run`, user prohibits subagents.
Sparse/index_add CUDA operations may be nondeterministic; strict bitwise
training determinism is not claimed. Random projection and inference RNG state
are saved explicitly; selected validation forward and exported logits replayed.

After smoke validation, scheduling/logging changes include all shared sources
and registries in frozen_sources.json. Each queued job rejects source changes.
Formal dispatch uses currently free GPUs 4,5,7 and 25 preplanned jobs.
