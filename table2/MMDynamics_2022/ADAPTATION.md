# MMDynamics on canonical text/image vectors

Author source: TencentAILabHealthcare/mmdynamics, commit
`06a2bb8a66953be77c53e6a027bc04193c5bdca4`. The author copyright statement in
`code/readme.md` is preserved. The upstream checkout does not supply a separate
LICENSE file. Executor: current session model (Codex); no model switch or
subagents. Independent reviewer: `not_run`.

## Fixed configuration and budget

The official entrypoint selects BRCA, and its train branch supplies hidden
dimensions `[1000]`, dropout 0.5, Adam lr=1e-4 and weight_decay=1e-4, StepLR
step_size=500 and gamma=0.2. No configuration for ESP/Flickr/IAPR/NUS exists.
We therefore fix that entrypoint's numeric training profile for all four,
without validation-based comparisons to the shorter ROSMAP profile. This
transfers numeric defaults only: no omics data or author checkpoint is loaded.
Classes and input widths come from the canonical dataset, never BRCA/ROSMAP.

Author `num_epoch=2500` and `range(num_epoch+1)` mean **2501 optimizer updates**.
We retain this inclusive bound and scheduler.step() after optimizer.step().
Saved `update=1..2501` corresponds to author epoch indices `0..2500`. Training
is full-batch on the entire outer training split. The author has no early
stopping; no additional patience or shortened budget is introduced. Smoke
checks use three updates and cannot enter formal statistics. Exact sources
are recorded in `configs/PARAMETER_SOURCES.json` before formal training.

## Native mechanism retained without editing model.py

The actual author `MMDynamic` is imported as
`table2.MMDynamics_2022.code.model.MMDynamic`; model.py is hash-pinned and
unmodified. Its `len(in_dim)` already supports two views. We retain:

- Feature information `sigmoid(FeatureInforEncoder(x))`, elementwise feature
  gating and its mean-information penalty for every view.
- Native Linear/ReLU/dropout encoders, per-view TCP classifiers and raw
  scalar confidence regressors. No extra sigmoid, clamp or normalization is
  placed on TCPConfidence. The softmax true-class confidence target keeps its
  native gradients; it is not detached.
- Confidence-weighted concatenation and the native fused classifier.
- Fused CE plus each view's information penalty, confidence MSE and view CE,
  with all original coefficients and Xavier-normal initialization unchanged.

`reproduce/checks.py` compares native logits, the complete loss and every
parameter gradient to an explicit formula using two unequal view dimensions.
It also checks inference with no label argument.

## Input and label boundaries

The shared loader checks canonical source/features, labels, class mapping and
fixed split fingerprints. Data roots are explicit. The two views are text
then image, each 100 dimensions, float32. Their public row-L2 processing is
already complete: we add no normalization, scaling, PCA or graph construction.
The author's omics CSV reader and its train-min/max transform are bypassed.
All canonical samples and both views remain present; no new split is made.

Only train features and train labels enter `model(train_x, train_y)` and its
auxiliary confidence targets. Validation and final test call `model.infer(x)`
without labels. Validation labels are used outside the model to compute ACC;
test labels are used only after checkpoint restoration to export final ACC.
No author pretrained model is read. Module imports do not execute main.py or
the generic train_test.py imports, avoiding same-name module collisions.

## Validation, persistence and replay changes

The original script checks test every 50 source epochs and saves its final
state. We remove all training-time test evaluation. An independent fixed
validation split is checked after each update, and strict validation-ACC
improvements save best.pt; ties retain the earliest update. The increased
validation frequency is an evaluation adaptation, not parameter search.

BestCheckpoint saves the complete native model state_dict (information
encoders, feature encoders, TCP classifiers, confidence regressors and fused
classifier), optimizer/scheduler states, RNG and shape/configuration metadata.
The same strict state_dict loading path is used for final test and fresh
process replay, with hash-checked selection metadata. Test is evaluated after
restoring that best state. Inference uses native FP32 and deterministic eval
dropout; no inference sampling is introduced. Every run exports unique test
IDs, labels, logits, predictions, full masks and identities through the shared
artifact API. Replay reconstructs a new model, loads saved weights and checks
new predicted classes/ACC exactly; finite logits differences are diagnostic.
It never fits a head or repeats training.

## Environment and outputs

First choice is the existing read-only `coe`. Required core dependencies are
torch, NumPy, SciPy and sklearn; no PyG, DGL, wandb, pretrained backbone or omics
package is required. Environment specifications and probe evidence are in
`environment/`. A clone is permitted only after actual required-path failure.
Real full-node short training and separate-process replay cover all four
datasets in preflight; their outputs remain separate from formal seeds.

All logs, weights, source snapshots and predictions stay in this method's
`log/` and `exp/`. The formal queue freezes four datasets x five seeds before
training, refuses occupied GPUs and source changes, and stops with a failure
record if any worker fails. It runs a fresh-process replay after each fit.
The shared strict summary requires all five seeds per dataset with identical
configuration/provenance and verified replay. ACC mean and sample std use
five training seeds, 1024–1028, ddof=1. No literature numbers or smoke scores
are substituted. Papers and other method directories are untouched.
