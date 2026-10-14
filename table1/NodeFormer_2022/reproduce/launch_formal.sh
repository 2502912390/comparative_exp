#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhichao/Research/other_exp
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
run_id=formal_nodeformer_author_defaults_20261015_v1
mkdir -p "table1/NodeFormer_2022/log/${run_id}"
exec /home/chenzhichao/miniconda3/envs/coe_baseline_nodeformer/bin/python -u -m table1.NodeFormer_2022.reproduce.run_matrix \
    --run-id "${run_id}" --devices cuda:4 cuda:5 cuda:7 \
    > "table1/NodeFormer_2022/log/${run_id}/matrix.log" 2>&1
