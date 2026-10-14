# HDMI adaptation to coe1005_fixedsplit_v1.1

Executor: current session GPT-6; no model switch or subagent. Independent
reviewer / agent-follows-doc verdict: not_run (user spending policy).

## Author implementation retained

Author: https://github.com/baoyujing/HDMI, commit
`00b1bbdf37a530e2ec5e1e836f25eb25ef1cd94e`.
The actual author `GCN`, `InterDiscriminator`, `modeler`, and `LogReg`
ClassDefs are compiled from the pinned local source files into an isolated
namespace. `configs/author_sources.json` verifies SHA256 before loading.
Author code is unchanged; the old generic imports, data loader, training
entrypoint and evaluator are never executed.

The single-layer relation-specific linear GCN + ReLU, graph multiplication,
per-node tanh/softmax relation attention, mean global readout, shuffled-row
corruption, all three relation MI objectives and all three fusion MI
objectives are retained. Relations' losses are summed, not averaged. The
joint negative keeps the positive local embedding and corrupts attributes.
Native discriminator sharing is configurable; the official main.py default
of separate relation/fusion discriminators is used. The native implementation's
sigmoid/project/sigmoid sequence and biases are preserved exactly.

## Inputs and scale

Only shared.coe_benchmark loads canonical acm/dblp/yelp/amazon/mag features,
all supplied relations, class identity and frozen outer masks. There is no
additional feature normalization or adjacency normalization; graph input
uses the public loader's provided normalized graphs. The author's original
feature preprocessing and weighted self-connections are replaced by this
common input contract and are not searched. Current DBLP/Amazon versions
are different from the paper's versions. No original paper ACC is copied.

Every provided graph stays sparse COO. All MAG 113919 nodes and both relations
remain present. Scalar-output bilinear layers use the algebraic identity
`B(x,y) = sum((x @ W) * y) + b` in float32. Node chunks of 4096 and activation
checkpointing bound discriminator activation memory; they preserve global
readout, node weights, parameterization, feature corruption and the complete
loss. No subgraph/node sampling or truncated relation is introduced. CUDA
sparse reductions may not be bitwise deterministic. TF32 is disabled.

`reproduce/checks.py` compares native/adapted outputs, every parameter
gradient and embeddings on two/three directed relations with shared/separate
discriminators, an uneven final chunk, and dense/sparse GCN consistency.
The check must pass before launch; its report is stored in method-local log.

## Two-stage evaluation

The backbone receives only features and graphs. It trains for a fixed 400
unsupervised updates and uses the final representation; no label CE is added,
and no validation/test label chooses an unsupervised update. This fixed
representation policy replaces the original loss-based early stop and its
10000-epoch upper bound under the common default budget. The linear head then
fits all outer training labels for up to 400 updates, with patience 200.

Exactly one probe is initialized at `training_seed + 100000`, with native
Xavier initialization and Adam. Validation accuracy strictly selects the
complete backbone+head checkpoint; ties retain the earliest probe epoch.
No gradient from the head reaches the unsupervised backbone. Formal inference
restores both selected states and computes test accuracy once. The original
50 probes, per-epoch test metrics, KMeans and similarity-search evaluation
are removed from the classification protocol. Replay reconstructs both
modules and performs new inference without fitting anything.

Width is 256 instead of the author's 128 default, following the benchmark's
preferred representation width. Actual parameters, both stages' update
counts and timing are recorded. These runs are not described as having the
same compute budget as a single-stage supervised model.

## Development and formal matrix

`configs/search_space.json` preregisters 12 candidates per dataset, including
the official learning rate/probe rate/six MI coefficients. The grid varies
backbone learning rate {0.001,0.0005,0.003}, probe rate {0.1,0.01}, and a joint
coefficient multiplier {1,10}. Development uses seed1024, validation only.
The earliest candidate wins ties. All five dataset configs are frozen before
formal jobs start. Formal seeds are 1024--1028 on unchanged outer splits.
Formal jobs start from scratch; development weights are never reused.

The full queue performs 60 development runs then 25 formal runs, independent
checkpoint replay, strict recount and mean/sample-std (ddof=1) export. A
pre-registered plan records the full scope before execution. Failure attempts
are kept; only equivalent same-seed OOM retries are allowed. Missing/failed
seeds make the summary incomplete and do not produce paper rows.

All runtime outputs remain under HDMI_2021/log and HDMI_2021/exp. The author
saved_model path is unused. Reference CoE2 code, data, coe environment and
paper files are not modified.
