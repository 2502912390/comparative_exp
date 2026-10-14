# Direct reuse of coe

The current PyTorch formula port requires only packages already in the
read-only `coe` environment. Use `/home/chenzhichao/miniconda3/envs/coe/bin/python`
directly after the native16 sparse forward/backward and canonical ACM check.
No pip/conda install, version change or new clone is needed. The earlier
`coe_baseline_gcn` clone and original failed TF import logs remain historical
evidence; TensorFlow is not a dependency of the selected port.

`env_spec.json` records this stack and physicalGPU0/two-CPU-thread binding.
Actual versions/executable are saved per run. From the GCN_2017 directory:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python reproduce/check.py --device cuda:0 --run-id coe_native16_NEW_ID --data-check
/home/chenzhichao/miniconda3/envs/coe/bin/python reproduce/launch.py --run-id gcn_author_defaults_coe_20261014_c --session baseline_gcn_author_coe_20261014_c
```

The first check verifies the literal official early-stop boundary, sparse/dense
normalization/forward/gradients, masked CE/half-L2, TF1Adam formulas and a real
canonical ACM update. It does not evaluate test. Use fresh IDs and commit/freeze
core code before launch. The launch entry immediately runs25 formal jobs with
the native author defaults and independent checkpoint replay; no HPO/statistics.

No cache, weights or logs are committed. All runtime outputs stay method-local
under log/exp.
