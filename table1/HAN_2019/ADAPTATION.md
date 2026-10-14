# HAN: fixed author defaults and public benchmark data

Official source: [Jhy1993/HAN](https://github.com/Jhy1993/HAN), commit
`71bac29a07fb8fab908d50a806a7bc38aa6c6611`, verified against sources.json.
The code is a disclosed PyTorch/DGL port of **HeteGAT_multi**, which is selected by
`code/ex_acm3025.py`; it is not the ordinary GAT or an unchanged TensorFlow execution.
Executor: user-authorized HAN child agent inheriting current session model; no model
override, Astra/reviewer not_run, no nested subagent. Exact model identifier is not
exposed by the available tools and is not guessed.

## Fixed configuration, with no HPO

`configs/default.json` records the official source and exact defaults: learning rate
.005, L2 coefficient .001, feature and attention dropout .6, one relation-attention
layer, eight heads of eight channels (**64** total), semantic attention128, one dense
output head, ELU, no residual, TF1Adam. Source is ex_acm3025.py lines22–35 and its
training feed with attn_drop=.6/ffd_drop=.6. All five datasets and five seeds use
this fixed configuration, without validation-based configuration selection.

The latest user instruction restores original **epochs200 and patience100**. Native
stopping resets curr_step when validation Acc >= its historical maximum **or**
validation CE <= its historical minimum, including ties; otherwise it increments,
and stops at100. Validation CE excludes L2, as official evaluation does. This rule
is kept separately from checkpoint selection: the public best checkpoint rule is
strictly improving validation Acc, earliest tie. Original combined Acc-and-loss
checkpoint save is replaced only for this common evaluation requirement. These
native budgets supersede the previous400/200 benchmark default for this run.

Previously generated 256-width HPO runs are preserved under their existing exp/log
folders; the old formal queue failed on score replay and is **not successful**.
Its configurations and scores are not used to choose or initialize the new fixed64
runs. Historical search_space.json is archived as an ignored artifact. No summaries
or candidate paper values are generated automatically by the new queue.

## Correspondence to the released implementation

| Official module | Port |
|---|---|
| attn_head no-bias 1x1 convolution | independent per-head Linear(input,8,bias=False), Xavier uniform |
| f_1/f_2 convolutions with bias | receiver/sender Linear(8,1), Xavier uniform, zero bias |
| leaky_relu(f1_i+f2_j), masked neighbor softmax | slope .2, positive edge support, DGL destination edge_softmax |
| coefs times projected features | fused u_mul_e_sum with column sender and row receiver |
| bias_add, ELU, no residual | zero head bias, ELU, no residual |
| feature dropout before projection and after projection | two independent inverted Bernoulli masks per head |
| coefficient dropout after softmax | separate inverted coefficient dropout, no truncated neighbors |
| SimpleAttLayer w/b/u normal(std=.1), tanh | identical trainable shapes and initialization |
| final one-head dense classifier | Xavier uniform Linear(64,C), zero bias, no extra dropout |
| masked softmax cross-entropy | mean single-label CE over all outer train nodes |
| TF1Adam | explicit bias-corrected step with epsilon in uncorrected second-moment denominator |
| tf.nn.l2_loss filtering actual variable names | .5*l2*sum(p^2) including every bias tensor |

BaseGAttN checks exact names bias/gamma/b/g/beta, whereas actual TF variable names
include scope and :0; thus its filter regularizes all trainable tensors. This actual
behavior is preserved. TF1Adam uses m=.9m+.1g, v=.999v+.001g²,
lr_t=lr*sqrt(1-.999^t)/(1-.9^t), update -= lr_t*m/(sqrt(v)+1e-8).
Ordinary PyTorch Adam places epsilon differently, so an independent literal update
oracle checks the custom implementation and state restoration.

SimpleAttLayer receives [N,V,D] and computes relation softmax **per node**, without
averaging across nodes. The paper describes global semantic-score averaging; this
port explicitly retains the released ACM HeteGAT_multi implementation. It does not
claim to implement a global semantic-attention variant or a sampled-batch substitute.

For V relations with dimensions d_v and C classes, trainable parameter count is
sum_v(64*d_v + 8*(2*8+2+8)) + 64*128 + 2*128 + 64*C + C.
Actual counts are exported in each manifest; evaluation copies have no trainable
parameters and do not change the native architecture.

## Data, graph direction and sparse computation

Namespaced shared.coe_benchmark provides canonical features/labels/masks and every
relation. All sample/split/class/feature fingerprints are verified. No author data,
new meta-path, remapped node, omitted view, or reference model/weight is used.
Each relation remains separate. Official adj_to_bias(nhood=1) uses edge existence;
the port uses positive provided-adj edges with exactly one self-loop. Existing edge
normalization weights are not extra attention weights. There is no relation_union,
symmetrization or test-based relationship selection. Support hash is stored separately
from canonical source and CCoE learned-graph hashes.

DGL src=column,dst=row preserves the row-receiver direction. Sparse u_add_v,
edge_softmax(dst), and fused u_mul_e_sum avoid NxN attention and E*heads*hidden
message tensors. Full MAG uses all113919 nodes, four classes, both complete relations.

## Actual replay failure and precision repair

The old formal ACM seed1024 run failed shared verify_model_replay with
`ValueError: Model replay scores differ`. From its selected epoch35 checkpoint,
repeated identical FP32 DGL CUDA evaluations showed maximal logit differences
1.43e-6 to1.91e-6, with identical class predictions. This exceeds the existing1e-6
absolute score assertion. Diagnosis is preserved in
exp/smoke_replay_diagnosis_20261014/diagnosis.json; old failure logs remain intact.

With the same weight values, formula and full graph evaluated in FP64, repeated
logit differences were at most3.55e-15. The new runner keeps training/optimizer in
FP32. **Every validation, restored-best evaluation, final export, fresh-object replay
and independent-process replay uses the same FP64 GPU evaluation path.** The
inference model and feature tensors are independent dtype copies; training model
and canonical features/hashes are never cast in place. Logits are saved as FP64.
There is no rounding, tolerance relaxation or prediction-only score bypass. Shared
verify_model_replay keeps its original atol1e-6 and receives newly computed scores.

GPU atomic reductions still do not guarantee bitwise determinism; FP64 reduces
roundoff well below the existing assertion rather than claiming exact CUDA bitwise
identity. Manifest config/adaptation record both precisions and actual requested
GPU. Trajectories record validation time and peak GPU allocation, result timing
records final inference time, so additional precision cost is visible.

## Environment and checks

Original coe lacks TensorFlow; that historical failure and clone remain documented.
The current port needs only installed torch2.1.1+cu121/DGL2.4.0+cu121, so per latest
user instruction it is checked and run **directly in coe**, without changing packages.
Historical coe_baseline_han clone is retained unused. PhysicalGPU1 is allocated.

Necessary checks are native64 asymmetric two-relation dense/sparse forward and all
parameter/input gradients in FP32 and FP64, independent evaluation-copy immutability,
TF1Adam and native stopping rules, and a complete64-width MAG update/validation/replay.
The first formal ACM default run trains under its full native budget, then must pass
fresh-object and independent-process score replay before the matrix continues.
All artifacts stay under this method exp/log, excluded from Git. No paper edits.
