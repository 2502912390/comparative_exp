# MRiemGNN: canonical Table1 reproduction

Upstream is `lx970414/MRiemGNN`, commit `6fc9b96122c6d202516b56ade3806b1cbcb937f1`.
The local paper itself links this repository (paper.txt near line50). Author
code/ is preserved. `reproduce/native` retains the released model, kernel and
manifold classes with package-relative imports; layer aggregation additionally
accepts an equivalent sparse operator to avoid full E-by-feature intermediates.
This is an official-code reproduction with disclosed canonical-data/evaluation
adaptations, not a claim of reproducing the paper's original dataset experiments.

## Fixed defaults and evaluation

`main.py::parse_args` is authoritative: hidden128, two layers, 64 RFF kernels,
dropout0.5, relation_dropout0.2, Adam lr0.001/weight_decay0.0005, epochs300,
patience100, beta0.6, mutual coefficient0.01 and temperature2.0. Fixed random
kernel omega/bias remain in the checkpoint. Class-balanced CE uses training
counts exactly as train.py. Symmetric embedding KL also uses only train nodes,
as the released node-classification trainer does. All five datasets use these
generic defaults, without HPO. `config.yaml` says hidden64 but is never loaded
by main.py. The recommended MAG script targets a different 100k/120-class/three-
relation OGB subset and sets lambda_mutual=0; it is not silently applied to our
113919-node/four-class/two-relation MAG. Amazon is 11944 nodes/two classes.

The native CLI defaults merge validation into training and monitor test macro-F1.
Those evaluation choices are explicitly replaced: original train labels only,
strict validation ACC checkpoint selection/earliest tie, patience100 on the
same monitor, restore best then one test. No logit-prior search, saved-logit
ensemble, extra probe, label smoothing, focal modification or link prediction
is executed. Main's inactive options remain inactive. Model state still includes
native unused auxiliaries/link decoder; those have no executed task loss.

Native default feature_norm=standard is retained, but fitted on canonical train
features only (unbiased std and native clamp1e-6), not the whole dataset. Mean/std
are stored and reused during replay, without fitting them again. Canonical
features, labels/classes, masks and supplied graphs are never modified. This
method-specific standardization is separately fingerprinted.

## Relations, kernels and bounded memory

Each canonical relation is read separately. The native loader removes relation
diagonals and the layer has its own learned self-loop transformation; this rule
is retained. Positive supports supply the native unweighted edge interface.
Native reverse-edge addition and observed composite relation patterns are kept:
each pair's relation bitmask determines a channel. All raw relations can be
reconstructed exactly from those masks, checked by the adapter; no view is lost.
The vectorized set/bitwise-OR construction matches data.py::build_adj_dict and
does not form an NxN tensor. Only observed combinations are instantiated. Cache
and graph fingerprints are separate from canonical and CCoE model graph hashes.

For each space/relation, phi(h)=sqrt(2/D)*cos(h omega^T+b), with independent native
RFF state and learned W. Original sum_j[phi(h_j) W]/degree equals
`sparse_mm(A, phi(H) @ W)/degree`. This distributes the same linear transform,
preserves duplicates/count degrees and relation-softmax aggregation, and avoids
E-by-D/E-by-hidden allocations on full MAG. No width, node, relation or edge
truncation, nonlinear-kernel removal, neighbor sampling or GCN substitution is
used. Small asymmetric graphs compare original released forward/parameter/input
gradients against the sparse path, including empty and duplicate-edge cases.

## Actual geometric mechanism in the released code

Both named Euclidean and Hyperbolic branches, their separate relation kernels,
self-loop transforms, global relation weights, symmetric KL and weighted fusion
are retained. The released default sets hyperbolic_proj=False; its curvature
Parameters and manifold methods are declared but never used in layer.forward.
The adapter does not invent curvature gradients, geodesic message passing or
manifold-aware optimization: native optimizer is ordinary Adam. This limitation
is checked and reported, rather than claiming that curvature is being trained.
The optional model projection is a norm clip, while manifold helpers are separate
Lorentz functions; optional unused helpers are not silently wired into the model.
Paper equations1-9 use the released RFF/aggregation/KL/fusion/classifier form;
introductory geometric terminology does not establish an active curvature path.

## Precision, source isolation and replay

Training, validation, export and replay remain FP32. Fresh-process replay restores
all fixed kernels/weights, saved train-only feature transform, canonical input and
derived graph identity. Exact node predictions and ACC are verified; finite
logit differences are separately logged under the revised user policy and do
not halt training. No FP64 workaround is introduced. No reference1005 training,
weights or learned graph is imported. Public functions and this adapter never
import author modules under generic utils/model/data names.

The actual model/data path needs torch/numpy/scipy/sklearn. PyG and ogb listed in
author requirements belong to unused dataset-preparation tools, so their absence
does not force a clone. coe is tested with actual kernels/short runs; it is never
modified. Only a required execution failure authorizes a minimally adjusted
coe_baseline_mriemgnn clone. Evidence is kept in method-local log/exp.

Executor: inherited current session model, exact identifier not exposed.
Independent reviewer/Astra: not_run; no new agent/model authorized for this task.
