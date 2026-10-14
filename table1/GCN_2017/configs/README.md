# GCN configuration policy

`default.json` is the single active formal configuration for all five datasets:
hidden16, lr=.01, dropout=.5, first-layer L2=.0005. These match the author's
`code/gcn/train.py` flags, including epochs200 and early_stopping10.
Official zero-based epoch>10/current-val-loss>mean(previous10) stopping is
preserved. The independent validation-ACC best checkpoint choice remains the
benchmark evaluation adaptation. Validation loss includes first-layer half-L2. No HPO is performed.

`smoke_acm.json` and `smoke_mag.json` have the same native hyperparameters with
shortened check-only budgets; phase smoke is ineligible for paper statistics.

`c00..c11.json` and `search_space.json` are historical provenance from the
canceled earlier assistant-created HPO/width256 queue. They are not active;
current formal CLI rejects these configs and the matrix entry has no search
mode. Preserve old log/exp artifacts as canceled history.
