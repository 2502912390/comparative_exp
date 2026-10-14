# HAN Table 1 adaptation

Source: official [Jhy1993/HAN](https://github.com/Jhy1993/HAN), commit
`71bac29a07fb8fab908d50a806a7bc38aa6c6611`, recorded in `sources.json`.
This is a disclosed PyTorch/DGL equivalent port, not an unmodified TF execution.
Executor: user-authorized HAN child agent inheriting the current Codex session model.
No model override; Astra and independent reviewer are not_run; no nested subagent.
Exact session model identifier is not exposed by the available tools and is not guessed.

## Exact official variant and mechanisms

`code/ex_acm3025.py` selects `model = HeteGAT_multi`, not the plain GAT class.
For every supplied relation it creates eight independent `attn_head` functions,
concatenates their outputs, calls `SimpleAttLayer`, then one dense classifier.
The public benchmark supplies its own five network datasets; no ACM3025 MAT,
author masks, synthetic meta-paths, reference model, or reference weights enter.

| Official implementation | Equivalent port |
|---|---|
| `utils/layers.py:attn_head`, no-bias size-1 convolution | independent per-head `Linear(d,32,bias=False)`; Xavier uniform |
| `f_1` and `f_2` size-1 convolution with bias | independent receiver/sender `Linear(32,1,bias=True)`, Xavier uniform, zero bias |
| `softmax(leaky_relu(f1_i+f2_j)+bias_mat)` | binary positive-edge support, leaky slope .2, DGL `edge_softmax(norm_by='dst')` |
| `tf.matmul(coefs,seq_fts)` | fused `u_mul_e_sum`, row receiver / column sender |
| `tf.contrib.layers.bias_add` and ELU | zero initialized head bias and ELU; no residual |
| original feature dropout before projection AND after projection | separate inverted Bernoulli masks at both places for every head |
| original coefficient dropout after softmax | inverted coefficient dropout; no neighbor truncation |
| `SimpleAttLayer`, w/b/u normal(std=.1), tanh | identical trainable w/b/u shapes and normal initialization |
| final dense output and one output head | Xavier uniform `Linear(256,C)`, zero bias; no extra classifier dropout |
| masked softmax CE | mean single-label CE on exactly all outer train nodes |
| Adam default parameters | custom TF1Adam beta(.9,.999), epsilon1e-8 in uncorrected denominator; no optimizer weight_decay |
| `tf.nn.l2_loss` summed by BaseGAttN | explicit `.5*l2*sum(p²)` over every trainable tensor |

Official `BaseGAttN.training` excludes exact variable names `bias/gamma/b/g/beta`;
TF actual names include scopes and `:0`, so the check does not exempt bias tensors.
The port preserves this actual behavior, including semantic and classifier biases.

**Semantic detail:** `SimpleAttLayer` receives `[N,V,D]` and applies `softmax`
to the relation dimension for each node. It does not average scores across all
nodes. The paper's semantic-level formula describes global node averaging;
we retain the released ACM-entrypoint implementation and explicitly name this
variant `official_HeteGAT_multi`. This distinction is not hidden by calling it
a global semantic-attention implementation. Full graph inputs are used; there
is no local batch approximation of semantic aggregation.

TF1 Adam update is implemented explicitly: m_t=.9*m_(t−1)+.1*g,
v_t=.999*v_(t−1)+.001*g², lr_t=lr*sqrt(1−.999^t)/(1−.9^t),
parameter -= lr_t*m_t/(sqrt(v_t)+1e−8). Ordinary torch.optim.Adam places
epsilon after bias correction; this port retains TF1 placement. A literal
independent update oracle tests small gradients and optimizer-state restoration.

## Public input and graph support

All `shared.coe_benchmark` canonical features, all relations, labels and masks
are preserved and verified before use. The public features receive no additional
normalization. HAN's own `adj_to_bias(...,nhood=1)` uses edge existence, so each
provided relation is converted to positive support with exactly one self-loop.
Original normalization weights are not used as extra attention weights; this is
disclosed model-native support conversion, not `relation_union`. Relations remain
separate. No symmetrization, removed relation, remapped node, or selected relation.
The resulting support hash is separate from canonical and CCoE learned-graph hashes.

Sparse DGL graph edges have `src=column,dst=row`. `u_add_v`, destination softmax,
and fused `u_mul_e_sum` preserve the dense directed formula. Heads execute separately;
no dense NxN matrix and no E×heads×hidden message tensor are materialized. All MAG
113919 nodes, four classes, and both full relations are used. Sparse GPU reductions
can use atomics; seeds/deterministic settings do not promise bitwise identical training.

## Capacity and training changes required by the benchmark

Official ACM uses 8 heads × 8 channels =64. The registered primary comparison uses
8 ×32 =256, the protocol-preferred width; eight-head structure and two-level attention
remain unchanged. Semantic attention width128 and one output head remain official.
For V relations, input dimensions d_v, C classes, total trainable parameters are
`sum_v(256*d_v + 8*(2*32+2+32)) + 256*128 + 2*128 + 256*C + C`.
Exact actual counts are written to each manifest; no unused capacity padding.

Training budget is400 epochs, validation every epoch, patience200, replacing
official200 epochs/patience100. Original script prints test results during training
and has combined loss/Acc selection; the benchmark selects strictly improving
validation Acc, earliest tie, then restores best. Development never evaluates test.
Twelve ordered candidates (lr .005/.001/.01, l2 .001/.0001, dropout .6/.3) are registered
before the batch. Each dataset develops only seed1024; all twelve must complete,
then the winning validation configuration is frozen and five formal seeds restart
from scratch. Each formal seed exports test once and performs fresh-object plus
independent-process checkpoint replay. No formal test score influences tuning.

## Environment and evidence

Actual coe TensorFlow import failed with `ModuleNotFoundError: No module named tensorflow`.
Before clone, environment spec and concrete probe were saved under `environment/`.
`coe_baseline_han` is a clone of coe; no packages were installed, removed, or upgraded.
PyTorch2.1.1+cu121 / DGL2.4.0+cu121 already passed a physical-GPU1 sparse forward/backward
kernel witness. Old TF1/tf.contrib is not force-installed into Python3.10. The original
coe and reference1005 directories are unchanged. Environment spec canonical digest
is used by the parent's compute ledger. No independent agent-follows-doc review
was run by this agent, because a further subagent was not authorized.

`reproduce/checks.py` compares full two-relation HAN outputs, semantic weights,
all parameter and input gradients against the independent dense formula on asymmetric
small graphs. Real ACM short training and full MAG update/validation are smoke-only;
their artifacts are excluded from formal summaries. Failure logs remain on disk.
