# GCN reproduction and monitoring

The original author repository is preserved in `code/`; the isolated,
formula-preserving implementation is in `reproduce/gcn_baseline`. See
[ADAPTATION.md](ADAPTATION.md) for all deliberate input, optimizer, width and
validation changes. Reference data and the `coe` environment remain unchanged.

Start a persistent, physical-GPU0 queue from this method directory (fresh IDs):

```bash
/home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python reproduce/launch.py --run-id gcn_table1_20261014_a --session baseline_gcn_20261014_a
```

It runs twelve seed1024 validation-only candidates on each of acm/dblp/yelp/
amazon/mag, freezes all five configurations, then trains 1024–1028 from scratch
and separately replays each selected checkpoint. No final mean/std is computed
until the user requests it. This command must only be used after the real-data
smoke checks pass and the core implementation is committed/frozen.

Monitor these stable paths:

- `log/gcn_table1_20261014_a/console.log`: queue stage, candidate/job transitions,
  validation choices, formal completion and failures.
- `exp/gcn_table1_20261014_a/status.json`: latest machine-readable status.
- `log/dev_gcn_table1_20261014_a_cXX/<dataset>/seed1024/console.log`: each
  development epoch; `XX=00..11` in preregistered order.
- `log/formal_gcn_table1_20261014_a/<dataset>/seed<seed>/console.log` and
  `replay.log`: each formal epoch and independent new-model inference proof.
- `exp/gcn_table1_20261014_a/COMPLETE`: all 60 development, 25 formal and 25
  replay tasks passed. It does not mean a statistics export was produced.
- `exp/gcn_table1_20261014_a/FAILED` and `failure.json`: retained failure detail;
  the queue stops instead of silently accepting partial seeds.
- All per-run model artifacts are under the matching method-local `exp/` tree.

After user confirmation, the manual statistics command is:

```bash
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe_baseline_gcn/bin/python reproduce/summarize.py --run-root exp/formal_gcn_table1_20261014_a
```

The public exporter requires all five seeds and hashes, re-counts predictions,
checks new-model replay and uses sample std (`ddof=1`). It emits candidate
LaTeX rows without editing the paper. Do not run it during background execution.

Single-run and replay APIs are documented in [shared README](../../shared/README.md)
and [environment witness commands](environment/README.md). Method arguments
follow `--dataset --seed --device --phase --run-id --output-root --config`.
Output-root is restricted to this method's `exp`; logs/weights/data are ignored
by the root Git repository.
