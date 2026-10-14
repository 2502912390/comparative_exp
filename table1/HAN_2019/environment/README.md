# HAN environment

Actual `coe` TensorFlow import failed (`No module named tensorflow`); see
`../exp/smoke_coe_environment_20261014/coe_probe.json`. The protocol permits a disclosed PyTorch equivalent port because
official TF1 uses removed `tf.contrib` and `tf.layers` APIs. Original coe remains read-only.
The clone `coe_baseline_han` is complete; **no dependency installation or upgrade**.
Pinned used versions: Python3.10.21, torch2.1.1+cu121, DGL2.4.0+cu121,
NumPy1.26.4, SciPy1.15.3. Exact per-run package list is exported by the shared manifest.

Clone command, only when environment does not exist:

```bash
flock /tmp/coe_baseline_env_clone.lock /home/chenzhichao/miniconda3/bin/conda create -y -n coe_baseline_han --clone /home/chenzhichao/miniconda3/envs/coe
```

Initial HAN clone began before the shared-lock coordination message; it completed
before GCN clone was allowed to start. Subsequent environment builds use this lock.

Direct host check command from the unique method directory:

```bash
cd /home/chenzhichao/Research/other_exp/table1/HAN_2019
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe_baseline_han/bin/python -m reproduce.checks --device cuda:0
```

Expected witness `HAN_CHECKS_PASSED` confirms forward, all model parameter gradients,
feature gradients, node-specific semantic attention, and directed two-relation sparse
equivalence. Check JSON/log are in method exp/log. Real small-dataset training and
MAG full update/validation are additionally required before launching a batch.
The parent records env_spec.json canonical SHA256 and tier1/2 evidence in its ledger.
Independent agent-follows-doc review is not run by this child; no nested subagent was
authorized. Caller always binds physicalGPU1 -> logical `cuda:0` with two CPU threads.
