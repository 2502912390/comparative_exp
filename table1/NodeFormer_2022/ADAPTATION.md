# NodeFormer reproduction

Official source: qitianwu/NodeFormer commit `5390b6908326905b2eb840f65a70b0f4082aef43`.
Executor: current session model; no subagents; independent reviewer `not_run`.
Protocol: `coe1005_fixedsplit_v1.1`. No hyperparameter search.

`configs/default.json` freezes the general numeric defaults in `code/parse.py`
and model-constructor boolean defaults in `code/nodeformer.py`. Gumbel,
LayerNorm and residual flags are explicitly on, as in official graph-present
`run.sh` recipes. None of these five benchmark variants has an official
specific configuration. The source of each value is in `configs/author_sources.json`.
Width=32, four 32-dimensional heads per convolution, two layers, M=30 random
features, K=10 Gumbel samples, tau=.25, dropout=0, Adam lr=.01/wd=.005,
edge-loss coefficient=.1. The official general relational-bias order is 0:
additive bias is inactive by fixed author default; its original implementation
remains available. The graph still provides an active edge-regularization prior.
This is disclosed rather than selecting a graph switch using benchmark scores.

Training budget is the official general 500 epochs; `main.py` has no early
stopping and we do not add one. Every epoch checks validation ACC, strict
improvement only, retaining earliest ties. All outer training labels enter
NLL(log_softmax(logits)) minus lambda times mean layer link loss. Test labels
are only used for final export after best restoration. The official repeated
per-epoch test evaluation is removed.

The shared verified loader supplies all five datasets, fixed masks, unmodified
features and all relations. Common unweighted `relation_union` merges positive
non-self edges and inserts one self loop. Its row=receiver convention is flipped
to official (sender,receiver) edge_index. No extra symmetrization, normalization,
CCoE context/weights, feature reduction or test-based relation choice is applied.
MAG retains all 113919 nodes and every union edge.

`reproduce/model.py` loads the official model through a unique importlib name,
without adding author directories to sys.path. Official random feature mapping,
all-pair kernelized softmax/Gumbel softmax, K samples, value aggregation,
LayerNorm, activations, residual and output layers are reused. Attention uses
activation checkpointing with RNG preservation, recomputing the same Gumbel
samples on backward. Edge regularization computes the identical
mean(log(dot(q'_i,k'_j)/den_i)/indegree_i) over every edge/head, in bounded chunks.
Its custom first derivative is the literal derivative of log dot and log
normalizer. Small directed-graph forward and gradients are compared against
unmodified official code. CUDA index_add and sparse operations can be
nondeterministic; no bitwise determinism is claimed.

Official random projections are generated from the current query sum on every
forward, including validation. Each layer saves its actual validation projection
matrix and generation seed as checkpoint buffers. At best-checkpoint test and
replay the exact saved matrices are used; Gumbel is disabled by official eval.
The pre-validation inference RNG is saved too. Restored validation logits must
match the selected forward before final test export. Replay constructs a fresh
model and compares all node class predictions and logits. `replay.py` additionally
supports an independent process and validates live source/config hashes.

Environment: actual official import in read-only coe fails on torch_sparse.
Clone `coe_baseline_nodeformer`, keep torch 2.1.1+cu121, install matched
pt21cu121 scatter/sparse wheels and PyG 2.5.3. See environment documentation.
All checks, logs, failures, source snapshots and results stay in this method's
log/exp. Core source/config/documentation only is committed.
