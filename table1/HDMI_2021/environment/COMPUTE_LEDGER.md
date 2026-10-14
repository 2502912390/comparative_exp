# HDMI local compute ledger

Existing coe reused read-only; no packages changed.

Spec SHA256: f21f828b7022556e740ffe6689256e29749443e53172585a282a7fa503be8cf1

Import probe: passed (Python3.10.21 / torch2.1.1+cu121).
GPU kernel/native-output/native-gradient witness: passed on physical GPU7, RTX4090.
Five complete-node datasets, three-update smoke and independent checkpoint
replay: passed, all 11 checks, elapsed 85.41 seconds.

Evidence: ../exp/smoke_hdmi_preflight_20261014_2005/preflight.json

MAG peak allocated memory: 2389.7 MiB; all 113919 nodes and both relations kept.

Sandbox binding to GPU7 hid CUDA. This was rechecked successfully in the host
GPU execution context and was not treated as an environment/clone trigger.

Independent agent-follows-doc: not_run under user policy.
