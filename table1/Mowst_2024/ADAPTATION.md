# Mowst-GCN fixed-default reproduction

Author source: facebookresearch/mowst-gnn, commit
2e3569962d2388bfda4535cdd1fc0b6eaec88a28 (MIT; code/LICENSE retained).
Parameter policy: author_default_no_hpo. Executor: current session GPT-6;
independent reviewer not_run. No subagents or model switches.

## Fixed variant and parameters

The main-table method is **Mowst-GCN**, method=mowst, separate expert losses,
native multi/random-route inference. It is not Mowst*. The first explicit
Mowst-GCN command in code/README.md supplies pretrain_model2 and
original_data=false (variance/entropy-only learnable gate). The generic
parser's original_data=none does not construct a gate, so this documented
executable variant is fixed before any training. No validation comparison
of gate features, variants or parameter configurations is performed.

Our five dataset IDs have no author-specific recommended configuration.
All numeric settings therefore use code/main.py's generic defaults:
MLP/GCN/gate hidden width64, two layers, lr=0.001, weight_decay=0.0005,
dropout=0.5, dropout_gate=0.5, AdamW, NLLLoss. epoch=20000, big_epoch=20000,
patience=100, big_patience=10; the source's strict '> patience' stop is kept.
lr_gate=0.001 is recorded but the native Mowst alternating path uses the
lr-based weak+gate and strong optimizers; its unused joint optimizer is not
silently substituted. See configs/default.json and PARAMETER_SOURCES.json.

## Native mechanism and input

The author GCN, MLP, GateMLP, compute_confidence, train_vanilla and the two
train_mowst*_simple_gate functions are executed from their actual ClassDef /
FunctionDef sources without edits. Hashes are pinned in author_sources.json.
Isolated namespaces avoid importing the original dataset loaders, wandb,
generic utils/model names or training entrypoint.

All five datasets use shared.coe_benchmark's exact identities, outer masks,
labels, classes and common node features. The public relation_union includes
every supplied positive relation edge, binarizes off-diagonal support and
adds one unit self-loop. PyG GCNConv then performs its native normalization.
Sparse adjacency row=receiver, column=sender is preserved and checked against
PyG edge_index. The author's optional torch_sparse input fills the diagonal,
whereas PyG's built-in torch COO path adds to an existing diagonal. The
adapter removes the union's unit diagonal immediately before GCNConv adds
it back, preserving exactly one unit loop and avoiding extra compiled
dependencies. Directed sparse and edge_index outputs are numerically checked.
No CCoE graph/encoder/weights, relation selection, feature
truncation, sampling or node deletion is introduced; MAG retains 113919 nodes.

The strong GCN is pretrained first. Mowst then alternates an MLP+gate turn
and a GCN turn. The native gate is sigmoid(MLP(variance, normalized entropy))
and the loss is the per-training-node gate-weighted sum of the experts' NLLs.
Native no_grad boundaries and training-mode BN/dropout behavior are kept.

## Required evaluation/checkpoint adaptations

The author vanilla validation-loss function incorrectly uses train_mask;
the wrapper uses actual validation labels for this stop signal. Pretraining
stopping keeps the author val_loss policy, independently saving the strict
best validation-ACC state. Alternating turns and outer rounds keep the
author validation-ACC stop/acceptance logic. No test metrics are evaluated
inside training. Final best checkpoint is selected at outer round validation.

Native Mowst save/rollback code omits the learned gate. Every adapted snapshot
includes MLP, GCN, gate and all BN buffers together, including inner rollback
and the final selected state. Optimizer moments follow the native behavior
(weight rollback does not roll optimizer moments back). The final checkpoint
also includes optimizer states and independent evaluation RNG state.

## Native stochastic ACC and saved evidence

The author m_times=50 is retained for validation and final test. Each draw
generates CPU torch.rand for ALL nodes, uses q < gate_confidence, and accepts
the selected expert's top-1 prediction. No probability average or majority
vote replaces this random routing. ACC for one training seed is the native
mean of 50 draw accuracies, equivalently sum(correct)/(50*N_test). Sample std
is calculated across five independent training seeds, not across 50 draws.
Validation/test CPU RNG seeds are training_seed+1000000/+2000000, fixed in
advance and isolated from training dropout. The saved RNG recreates all draws.

The common single-prediction exporter cannot represent 50 predictions/node.
The method-local mowst_bernoulli_mc_v1 extension retains the common loader,
manifest/snapshot, checkpoint selector, five_seed_statistics and write_summary
APIs without changing shared code. predictions.npz keeps unique node_ids and
labels/masks, y_pred[50,N_test], hard routes[50,N_test], expert probabilities
and gate confidence. result.json explicitly distinguishes N_test unique
nodes from 50*N_test stochastic observations. Recount independently verifies
every routed class and correct count; fresh-process replay reconstructs all
three modules and checks every draw's classes/routes, without fitting.

## Environment and outputs

Actual coe import failed: ModuleNotFoundError torch_geometric. It was cloned
to coe_baseline_mowst; only torch-geometric==2.5.3 was added, keeping existing
torch2.1.1+cu121 and other base packages. No optional PyG extension, wandb or
original-dataset package is required by the selected path. Environment
evidence/specification is under environment/ and log/dev_setup/.

All logs, checkpoints, source snapshots, MC predictions and summaries stay in
this method's log/exp directories. Smoke uses two inner epochs/two rounds
only for implementation checks and is never included in the 25 formal jobs.
