# Current HAN environment: original coe

Latest instruction: use coe directly when the actual adapted method works there.
Current PyTorch/DGL implementation uses already installed torch2.1.1+cu121,
DGL2.4.0+cu121, NumPy1.26.4, SciPy1.15.3; no install/upgrade is required.
Original coe packages are unchanged. Per-run environment.json exports actual versions.
Historical TF import failure is preserved in exp/smoke_coe_environment_20261014/
and log/smoke_coe_environment_20261014/console.log. Previous unused no-package-difference
coe_baseline_han clone is retained; new commands intentionally use original coe.

Actual full native64 checks from the method directory:

```bash
cd /home/chenzhichao/Research/other_exp/table1/HAN_2019
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python -m reproduce.checks --device cuda:0
```

Expected HAN_CHECKS_PASSED includes directed two-relation FP32/FP64 outputs and
gradients, independent evaluation-copy immutability, TF1Adam update/state checks,
and official inclusive early stopping. Necessary complete MAG smoke follows.
FP32 training and FP64 GPU evaluation are both checked using installed DGL kernels.
PhysicalGPU1 maps to logicalcuda:0, twoCPU threads. No nested reviewer is authorized;
independent agent-follows-doc verdict is not_run. Parent maintains environment ledger.
