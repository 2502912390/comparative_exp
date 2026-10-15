# InfoMGF canonical Tables 1 and 2 reproduction

Upstream: `zxlearningdeep/InfoMGF`, commit
`27962685731a7cde96fb839c3fcf1b45e6cfca2f`. The original checkout is preserved.
Executor: current session model; exact model identifier not exposed. Independent
external review: not run. No hyperparameter search is implemented or authorized.

## Fixed configuration and input

`configs/author_defaults.json` copies every default in `params.py::{acm,dblp,yelp,mag}_params`.
ACM defaults are the first documented/general launch and are fixed for Amazon and
the four Table 2 datasets, for which authors provide no config. Dataset changes
only input width, relation count, class count and masks. MAG retains hidden 256,
3 encoder/GCN layers, 200 encoder epochs, contrast batches 2560, evaluation every
50 encoder epochs, downstream hidden 256 and maximum 500 epochs. There is no
model-capacity or training-budget reduction. `graph_block=1000` follows upstream;
`contrast_block=256` is a memory chunk, not a negative-sampling or model parameter.

Canonical namespaced `shared.coe_benchmark.load_dataset` checks all reference
source, dataset, split, label and class-mapping hashes. All nodes/views and native
feature scaling are preserved; graphs stay sparse. Original author loader is
not imported. CCoE model graph hashes are unrelated to InfoMGF learned graphs.

Table 1: shared features equal each relation's canonical feature matrix, native
`AGG(A_v^r X)` row-L2 contexts plus raw shared feature concatenate feed the fusion
learner. All 2/3 provided relations are used. These relations are not unioned.

Table 2 is an explicitly disclosed interface extension, not an original author
experiment: per-view features create unlabelled cosine top-(k+1) graphs using
the author's dense graph processing formula (symmetrize, ReLU, degree normalize).
Placeholder empty graphs are never used. Each specific learner sees its own
100-dimensional `row_l2(A_v^r X_v)`. The one shared GCL encoder and evaluation GCN
both see concatenated text/image 200-dimensional features. Fused learner input is
100+100+200=400 dimensions. No trainable alignment or independent view encoders
are added. Flickr is the canonical six-class dataset, not another Flickr release.

## Preserved mechanisms and bounded memory

`mechanism.py::Attentive` is exactly `x @ diag(w)` (implemented by elementwise
multiplication); two layers initialized at w=1 and first ReLU/tanh match
`layers.py::Attentive` / `graph_learners.py::ATT_learner`.

The dense default graph path remains normalized cosine → row top-(k+1) →
`(S+S.T)/2` → ReLU → symmetric degree normalization → edge dropout. Its tensors
are stored in COO and multiplication is sparse, but the operator is unchanged.

MAG preserves the author's distinct `knn_fast` sparse path: outgoing+incoming
signed kNN degrees, normalization before ReLU, duplicate forward/reverse DGL
edges, independent edge dropout, **no extra symmetric renormalization**.
Discrete exact topk indices are selected in row blocks with gradients disabled,
then selected cosines and degrees are recomputed with gradients. Gradients still
reach both endpoints and the learner weights. Selection is still exact across all
nodes, not approximate kNN, sampled nodes, truncated nodes, or an NxN allocation.
Topology decisions are nondifferentiable upstream as well. Duplicate DGL edges
are not collapsed before dropout. Dense graph path coalesces before dropout as
upstream's dense path requires. COO zero-entry RNG consumption differs from
upstream dense dropout; distribution and positive-edge dropout mechanism agree.

Native graph convolution is `A Linear(X)` including bias before aggregation;
the DGL sparse path retains upstream rows=source/cols=destination behavior.
Canonical input rows=receive is documented; for asymmetry MAG augmented provided
graphs are therefore applied with the official DGL transpose, while AGG retains
canonical matrix multiplication. Learned graphs are symmetric. This deliberate
upstream path difference is not silently changed. DGL u_mul_e/sum kernels work in
current coe; old DGL requirements are not blindly installed.

Shared encoder has native input/hidden dropout, ReLU intermediate layers, final
row-L2. Three separate projection families preserve specific/specific,
specific/augmented and fused/specific symmetric InfoNCE terms with temperature
0.2 and upstream EPS=1e-10. All view pairs are used. Official random augmentation
(default for all datasets) drops non-diagonal provided edges and preserves loops.
The optional non-default generative discriminator is not executed; no defaults
enable it. No supervised CE enters unsupervised encoder training.

Contrast uses the full original within-batch negative denominator. For MAG the
upstream shuffled disjoint batches of 2560 and len(batch)/N weighting are kept.
Else full all-node loss is retained. Recomputed row-block autograd checkpointing
avoids NxN identity/similarity storage and does not change the objective.
`checks.py` tests dense/sparse graph forward and input gradients against literal
official formulas, plus contrast loss and gradients on a small exact oracle.

## Evaluation and replay

Official `main.py` trains a fresh downstream native GCN at each encoder eval
checkpoint, selects that head by validation accuracy every 10 head epochs, and
checks patience after 10 non-improving checks. These budgets/frequencies stay.
The official code also computes/prints test at each evaluation and ultimately
reports the **last evaluated** graph/head, rather than an outer globally selected
checkpoint. This adaptation removes all intermediate test evaluations and
selects the outer encoder/graph/head chain strictly by validation ACC (earliest
tie), as required by the public protocol. This is training-checkpoint selection,
not configuration search. No inner repeated-probe averaging is added.

Each best.pt includes unsupervised encoder, specific/fused learners, all projection
families, optimizer state, frozen learned graph with duplicate edges, selected
downstream GCN and optimizer state, selected head epoch, config and RNG. Final
test uses the frozen graph and selected head once. Replay reconstructs both model
states and runs fresh inference through canonical features, frozen graph and GCN;
it never retrains graph learners or the evaluation GCN. Public prediction/count/
identity assertions remain strict.

## Commands and output

Run in this method directory with `PYTHONPATH=/home/chenzhichao/Research/other_exp`,
`PYTHONDONTWRITEBYTECODE=1`, fixed `COE_DATA_ROOT` and `COE_MULTIVIEW_DATA_ROOT`.
Use `/home/chenzhichao/miniconda3/envs/coe/bin/python` (coe remains read-only).

```bash
python -m reproduce.checks --device cpu
python -u -m reproduce.run --dataset acm --seed 1024 --device cuda:0 --phase smoke --run-id unique_smoke --smoke-epochs 1
python -u -m reproduce.run_matrix --run-id formal_infomgf_author_defaults_20261015_a --devices cuda:0
python -m reproduce.replay --job-dir exp/formal_infomgf_author_defaults_20261015_a/acm/seed1024 --device cuda:0
python -m reproduce.summarize --run-root exp/formal_infomgf_author_defaults_20261015_a
```

Smoke budgets are explicit and rejected for formal phase. Formal matrix covers
9 datasets × exact seeds 1024–1028, stops on the first failure, and checks frozen
source hashes before each job. Logs are `log/<run>/console.log` and per-job
`log/<run>/<dataset>/seed<seed>/{console,replay}.log`. Weights/predictions and
status are under `exp/<run>`. Full and Table-grouped sample-std(ddof=1) summaries
are written only for replayed results. Paper is not modified. All artifacts/data
are ignored by root Git; only core adaptation files are committed.

## Frozen Table 2 inputs and verified environment

Table 2 initial, untrained per-view feature graphs use the same exact cosine
top-(k+1), symmetrization, ReLU and degree-normalization formula, built once on
CPU and stored under `exp/_cache/initial_graphs/`. CPU construction avoids CUDA
scatter rounding changing a fixed input's byte fingerprint between seeds. Cache
identity includes canonical data identity, k, dtype/PyTorch version and adapter/
mechanism source hashes; cached graph states are independently hash-checked.
All formal seeds reuse these fixed input graphs. This is a disclosed input
adapter and data freeze; learned specific/fused graph training retains its
native device/formula/gradients. Learned graph fingerprints live with individual
checkpoints, separately from the fixed input and canonical data fingerprints.

Existing coe passed CPU ACM short execution, CUDA literal graph/gradient/contrast
oracles, full 113919-node MAG one-update execution and independent replay, plus
complete 12154-node/six-class Flickr short training and independent replay.
Two fresh processes with different seeds produced identical Flickr fixed-graph
and adaptation fingerprints. No clone or package changes were needed. Evidence
is indexed in `environment/PREFLIGHT.md`. Smoke runs never enter formal means.
