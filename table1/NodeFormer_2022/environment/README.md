# NodeFormer environment

Original coe is read-only. Actual import of code/nodeformer.py failed with
`ModuleNotFoundError: No module named 'torch_sparse'` under Python 3.10.21 /
torch 2.1.1+cu121. No forward or short training can execute past that import.
Local RTX 4090 GPUs and CUDA are accessible. Clone dedicated environment:

```bash
/home/chenzhichao/miniconda3/bin/conda create -y -n coe_baseline_nodeformer --clone /home/chenzhichao/miniconda3/envs/coe
/home/chenzhichao/miniconda3/envs/coe_baseline_nodeformer/bin/python -m pip install --no-index 'torch-scatter==2.1.2+pt21cu121' 'torch-sparse==0.6.18+pt21cu121' -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
/home/chenzhichao/miniconda3/envs/coe_baseline_nodeformer/bin/python -m pip install 'torch-geometric==2.5.3'
```

The [official PyG install matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
maps torch 2.1.* / cu121 to torch-2.1.0+cu121 wheels, including Python 3.10.
No torch, NumPy or unrelated audio/video/LLM dependency upgrade is needed.
Package differences and import/seeded CUDA witness are stored in log/exp.
The user forbids subagents: independent agent-follows-doc check is `not_run`.

From repository root, run the exact CUDA output/gradient witness:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/chenzhichao/miniconda3/envs/coe_baseline_nodeformer/bin/python -m table1.NodeFormer_2022.reproduce.checks --device cuda:0
```

Actual real-data two-step training, selected-checkpoint validation reproduction,
new-model test replay, and separate-process replay must pass before formal launch.
Full MAG smoke uses every node and union edge. No smoke score selects a config.
