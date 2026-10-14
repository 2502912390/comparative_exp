# GCN fixed-author-default reproduction

Upstream: `tkipf/gcn`, commit `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`.
The original six `code/gcn/*.py` files remain unchanged. This is a **PyTorch
formula-preserving port**, not unmodified official TensorFlow code.

The authoritative configuration is `configs/default.json`, identical for
acm/dblp/yelp/amazon/mag: **hidden16, learning rate .01, dropout .5,
first-layer L2 coefficient .0005, epochs200, early_stopping10**, exactly the flags in official `train.py`.
There is **no validation hyperparameter search**. Validation selects the best
training checkpoint only. The former width256/HPO queue was canceled by the
user; its complete/partial artifacts remain under their original log/exp IDs
and are not the new comparison results.

## Mechanism correspondence

| Official implementation | Adapted implementation |
|---|---|
| `models.GCN._build`: two graph convolutions, hidden1=16 | `gcn_baseline.model.GCN`, exactly two matrices, native hidden16 |
| `layers.GraphConvolution`: `support @ (X @ W)` | Sparse CSR `torch.sparse.mm(support, X @ W)` |
| First ReLU; final identity; default `bias=False` | First ReLU, final raw class logits, no bias |
| Input dropout before each matrix; sparse first input | Bernoulli inverted dropout before both matrices. Dense feature storage also samples zero entries, whose masks do not affect outputs. Same nonzero distribution; TF/PyTorch RNG is not bitwise equivalent. |
| `inits.glorot`: uniform ±sqrt(6/(fan_in+fan_out)) | `xavier_uniform_(gain=1)` |
| `masked_softmax_cross_entropy` with mask/mean(mask) | Mean CE on all fixed outer training nodes only |
| Regularization only `layers[0]`, `tf.nn.l2_loss` | Explicit `.0005 * .5 * sum(W0²)`; W1 unregularized |
| TF1 Adam, beta1=.9 beta2=.999 eps=1e-8 | `TF1Adam` preserves TF1 epsilon placement and corrected learning-rate formula; no AdamW/global weight decay |
| `utils.normalize_adj`: `(A D).transpose() D` | Exact `D Aᵀ D`, D=row-degree^-1/2, verified on an asymmetric graph |

The actual SciPy operator transposes adjacency; this equals D A D for an
undirected graph. This port preserves the actual directed-input operator,
without assuming all data are symmetric or forcibly symmetrizing them. The
public input graph convention is row receiver / column sender before the
native GCN operator.

## Benchmark input and evaluation adaptations

- Namespaced `shared.coe_benchmark` supplies canonical features, all nodes,
  labels, class mapping and fixed masks with strict registry fingerprints.
  No author same-name utils/model import contaminates this interface.
- Every provided relation enters public `relation_union`: binary nonself
  union plus one unit self loop. This is an **input adapter**, not a CCoE
  mechanism. GCN normalizes the union afterwards without another self loop.
  No CCoE learner, training or weights are used.
- Official Planetoid data/splits and row-L1 feature normalization are replaced
  by canonical observations. No extra normalization is applied by this port.
- Train uses all outer train labels. Validation ACC strictly improves best
  (ties retain earliest epoch); test is evaluated only after restoring best.
  Author stopping is retained: at most200 epochs; after zero-based epoch>10,
  stop when current validation loss exceeds the mean of the preceding10 losses.
  Validation loss includes the official first-layer half-L2 term. Checkpoint
  selection by validation ACC is independent of this loss-based stopping rule.
- All five datasets use the single fixed default config and formal seeds
  1024–1028. Formal run.py rejects any differing config or missing default-policy
  declaration. No development trials, per-dataset config selection or test-based
  relation/parameter choice is performed.
- Sparse CSR retains all113919 MAG nodes and all union edges; no dense NxN,
  cropped graph, or altered masks. Parameter count is reported per dataset.
- Python/NumPy/Torch/CUDA seed settings use the public helper; CUDA sparse
  kernels are not claimed to be cross-host bitwise deterministic. Failed
  hashes, nonfinite objectives or replay differences remain errors.

## Environment and checks

Real `coe` TensorFlow and official GCN imports failed with ModuleNotFoundError;
GPU forward/backward passed. The earlier isolated clone is retained as history. The PyTorch port
requires only packages already available in `coe`, so current training directly
uses `/home/chenzhichao/miniconda3/envs/coe/bin/python` after a native16 sparse
kernel and real-data check. No packages are installed/upgraded and `coe` is
unchanged. The current environment spec records direct reuse of `coe`.

The equation check uses **hidden16** and independently checks asymmetric SciPy
normalization, sparse/dense forward and parameter gradients, masked CE/half-L2,
and TF1 Adam's NumPy formula. The native16 coe check verifies exact official stopping boundaries and
validation loss, and performs a canonical ACM update without evaluating test.
The first new formal ACM task uses the full author200/early10 schedule and
separately replays its selected validation-ACC checkpoint. Previous
full-MAG sparse implementation checks remain applicable to the same graph
operator; their width256 scores are smoke history and are never new table data.

## Running and monitoring

From this method directory after the core is committed/frozen:

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python reproduce/launch.py --run-id gcn_author_defaults_coe_20261014_c --session baseline_gcn_author_coe_20261014_c
```

PhysicalGPU0 is process-bound with CUDA_VISIBLE_DEVICES=0; two CPU threads.
The queue runs25 formal trainings plus25 independent checkpoint replays. It
never automatically generates mean/std or modifies the paper.

- Total log: `log/<batch>/console.log`; state: `exp/<batch>/status.json`.
- Per-seed console/replay: `log/formal_<batch>/<dataset>/seed<seed>/`.
- Checkpoints, full predictions, provenance/snapshots and results: corresponding
  `exp/formal_<batch>/<dataset>/seed<seed>/`.
- `exp/<batch>/COMPLETE` means all25 tasks and replays passed. `FAILED` and
  `failure.json` retain errors and stop the queue rather than dropping seeds.
- After user confirmation, manually run `reproduce/summarize.py --run-root
  exp/formal_<batch>`; the public exporter requires five complete matching seeds
  and reports mean/sample std (ddof=1), without paper modification.

Historical `configs/c00..c11.json` and `search_space.json` are retained only as
old-queue provenance; current CLI does not run them. All data/logs/weights and
runtime snapshots are ignored by Git. Only core code, defaults, source metadata
and documentation are committed.
