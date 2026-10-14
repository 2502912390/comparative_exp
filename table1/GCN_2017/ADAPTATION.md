# GCN fixed-split reproduction

Upstream: `tkipf/gcn`, commit `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`;
the original six `code/gcn/*.py` files remain unchanged. Implementation type:
**PyTorch formula-preserving port**, not unmodified official TensorFlow code.

## Mechanism correspondence

| Official implementation | Adapted implementation |
|---|---|
| `models.GCN._build`: two graph convolutions | `gcn_baseline.model.GCN`, exactly two learned matrices |
| `layers.GraphConvolution`: `support @ (X @ W)` | Sparse CSR `torch.sparse.mm(support, X @ W)` |
| First layer ReLU; final identity; default `bias=False` | ReLU first layer, raw class logits last layer, no bias |
| Input dropout before each matrix; sparse first-layer input | Bernoulli inverted dropout before both matrices; dense feature storage includes zero entries, whose masks do not alter outputs. Distribution agrees on nonzeros; random draws need not match TF bitwise. |
| `inits.glorot`: uniform ±sqrt(6/(fan_in+fan_out)) | `nn.init.xavier_uniform_(gain=1)` with same distribution |
| `masked_softmax_cross_entropy` with mask/mean(mask) | Mean CE on all and only fixed outer train-mask nodes |
| Weight decay only `layers[0]`, `tf.nn.l2_loss(W)` | Explicit `weight_decay * 0.5 * sum(W0²)`; W1 unregularized |
| `tf.train.AdamOptimizer`, beta1=.9 beta2=.999 eps=1e-8 | `TF1Adam` preserves TF1 epsilon placement and bias-corrected learning-rate formula; not AdamW/global optimizer weight decay |
| `utils.normalize_adj`: `(A D).transpose() D` | Exact `D Aᵀ D`, D=row-degree^-1/2, checked on a directed graph |

The official SciPy operator transposes the adjacency. This equals D A D on
undirected inputs; this port explicitly preserves the actual directed-input
operator rather than assuming every provided graph is symmetric. The public
graph convention remains row receiver / column sender before this native GCN
operator. We do not force symmetrization or select a relation using test ACC.

## Deliberate benchmark adaptations

- Canonical data, features, labels, class mapping and masks are loaded by
  `shared.coe_benchmark`; source/split/label hashes must match registry.
- All relation channels enter public `relation_union` (binary nonself union
  plus exactly one unit self loop). This is a **protocol input adaptation**,
  not a CCoE model mechanism. GCN applies its own operator afterwards and does
  not add another self loop. No CCoE training, graph learner or weights enter.
- Original Planetoid data/splits and original row-L1 feature normalization are
  replaced by canonical benchmark inputs; canonical feature processing is
  kept unchanged, including dataset-specific preprocessing. No row-L1/L2 is
  added by this GCN port.
- Hidden dimension is fixed at 256 instead of the original 16, following the
  benchmark's preferred representation width; exactly two layers remain.
  Effective dimension and exact trainable parameter count are in manifests.
- Train/val/test masks are distinct. The official validation-loss moving
  average stopping is replaced by strict validation-ACC improvement,
  earliest tie, at most 400 epochs, patience 200 checks. Every epoch validates.
  Development trials never compute test accuracy. Formal test is computed
  from the restored best checkpoint, followed by separate-process replay.
- Twelve candidates per dataset are preregistered in `configs/search_space.json`:
  lr {.01,.005,.001} × first-layer L2 {.0005,0} × dropout {.5,.3}, all width256.
  Seed1024 validation selects once; ties follow candidate order. All 60
  development trials precede frozen configs and 25 new formal trainings.
- Formal seeds are 1024–1028. Sparse graphs and all nodes are retained,
  including MAG's 113919 nodes. No minibatch truncation or dense NxN matrices.
- Python/NumPy/Torch/CUDA seeds and deterministic settings follow the public
  API. Sparse CUDA kernels are not claimed to be bitwise deterministic across
  hosts. Failed hash or replay assertions remain failures and are not relaxed.

## Environment and validation

Run `reproduce/environment_probe.py` with `coe` first. It actually attempts
TensorFlow/official model imports in an isolated subprocess and saves their
errors, then witnesses CUDA forward/backward. Only an official compatibility
failure permits cloning `coe_baseline_gcn`. `coe` is never installed into or
modified. The clone uses its existing PyTorch/NumPy/SciPy; no packages are
installed. `environment/env_spec.json` records this declared stack/resources.

Validation command checks an asymmetric normalization against the literal
official SciPy expression, sparse/dense forward and parameter gradients,
masked CE and half L2, and TF1 Adam against independent NumPy equations.
Real-data smoke includes a four-update ACM run and one **full MAG** update,
best-checkpoint restoration, export and fresh-process model replay. These
are smoke artifacts, never formal table values. Exact witness logs live in
method-local `log/`; machine-readable evidence is under `exp/`.

## Commands and artifacts

From this method directory, set the env vars declared in `env_spec.json`, then:

```bash
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python reproduce/check.py --device cuda:0 --run-id fresh_check
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python reproduce/run_matrix.py --run-id gcn_new_unique_id --devices cuda:0 --seeds 1024 1025 1026 1027 1028
```

Physical GPU0 is process-bound by `CUDA_VISIBLE_DEVICES=0`; `cuda:0` means
that bound GPU. Use fresh IDs; no command overwrites historical outputs.
The controller must be started in tmux (or another persistent supervisor).

- Controller: `log/<batch>/console.log`; current state:
  `exp/<batch>/status.json`.
- Development console: `log/dev_<batch>_cXX/<dataset>/seed1024/console.log`;
  validation-only records and checkpoints at the matching `exp/` path.
- Frozen configs: `exp/<batch>/frozen_configs/<dataset>.json`, with complete
  validation candidate records at `exp/<batch>/selection/<dataset>.json`.
- Formal console and replay: `log/formal_<batch>/<dataset>/seed<seed>/`;
  weights, predictions, source snapshots, manifest and hashes in matching exp.
- `exp/<batch>/COMPLETE` means all 60 development runs, 25 formal runs and 25
  new-model replays passed. `FAILED`, `failure.json` and controller traceback
  are retained on errors. The queue stops on failure without discarding seeds.
- The controller **does not summarize**. After user confirmation, run
  `reproduce/summarize.py --run-root exp/formal_<batch>` to invoke the public
  strict five-seed mean/sample-std (ddof=1) exporter. It will refuse incomplete
  or inconsistent results. No paper is modified.

Only method core implementation, preregistered configs, environment spec,
source metadata and this explanation are committed. Data, logs, weights,
predictions and run snapshots remain untracked method-local artifacts.
