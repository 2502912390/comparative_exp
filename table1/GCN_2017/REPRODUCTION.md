# GCN author-default runs and monitoring

Use one fixed author configuration for every registered dataset:
`configs/default.json` — native hidden16, lr=.01, dropout=.5, first-layer
L2=.0005, epochs200 and early_stopping10. No validation hyperparameter search; val selects checkpoints only.
[ADAPTATION.md](ADAPTATION.md) documents the exact PyTorch/TF1 correspondence
and the common data/mask/relation-union adaptations. Official val-loss moving-mean stopping is preserved; checkpoint
selection still independently uses strict valAcc improvement and earliest tie.

From this method directory after commit/freeze, start the persistent GPU0 queue:

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python reproduce/launch.py --run-id gcn_author_defaults_coe_20261014_c --session baseline_gcn_author_coe_20261014_c
```

It immediately runs acm/dblp/yelp/amazon/mag × seeds1024–1028 (25 formal jobs),
rebuilding each best checkpoint for independent inference/recount. No HPO,
development configs, mean/std export or paper edits occur in the queue.

- Total log: `log/gcn_author_defaults_coe_20261014_c/console.log`.
- Current status: `exp/gcn_author_defaults_coe_20261014_c/status.json`.
- Fixed config/provenance: `exp/gcn_author_defaults_coe_20261014_c/fixed_configuration.json`.
- Per-seed console and replay logs:
  `log/formal_gcn_author_defaults_coe_20261014_c/<dataset>/seed<seed>/`.
- Corresponding checkpoints, predictions, manifests and source snapshots:
  `exp/formal_gcn_author_defaults_coe_20261014_c/<dataset>/seed<seed>/`.
- Completion: `exp/gcn_author_defaults_coe_20261014_c/COMPLETE` (all25 replayed).
  Failure: `FAILED` and `failure.json` in the same batch directory.

Old HPO batch `gcn_table1_20261014_a` has been stopped and marked
`canceled_by_user_policy_change`; its entire old log/exp tree remains intact.
It must not be mixed with new native-width16 fixed-default results.

After user confirmation only:

```bash
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python reproduce/summarize.py --run-root exp/formal_gcn_author_defaults_coe_20261014_c
```

The public exporter requires all five matching complete/replayed seeds and
uses sample std (`ddof=1`). It generates candidate rows without editing papers.
Single-run and replay commands follow the public [shared API](../../shared/README.md)
and [environment instructions](environment/README.md). `run.py --phase formal`
requires the exact default configuration; output-root must be this method/exp.
