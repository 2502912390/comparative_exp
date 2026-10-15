# Mowst local compute

Environment: coe_baseline_mowst, cloned from coe after actual missing
torch_geometric import. Added only torch-geometric==2.5.3; no base package
changes and no changes to original coe.

Spec SHA256: 9e105b3ab7b9b7977e1f35a766bc0038a62789685d27dc97437332f2ff51cb99

Native GPU kernel/update/default checks passed on physical GPU7 (RTX4090).
All five full-node short trainings and fresh-process MC50 replays passed.
MAG peak allocated GPU memory: 1570.97 MiB; all113919 nodes retained.

Evidence: ../exp/smoke_mowst_native_20261015_a/preflight.json.
Independent agent-follows-doc / review: not_run; user prohibits subagents.
