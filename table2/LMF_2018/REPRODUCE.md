# LMF reproduction

Use coe directly, from this method directory; set PYTHONPATH to /home/chenzhichao/Research/other_exp and both COE data roots as in the root protocol. Author fixed parameters/source hashes are configs/author_defaults.json and configs/author_sources.json; differences in ADAPTATION.md.

```bash
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python -B -m reproduce.checks --device cuda:0
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python -B -m reproduce.run --dataset esp --seed 1024 --device cuda:0 --phase smoke --run-id smoke_unique_id --smoke-epochs 2
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python -B -m reproduce.replay --job-dir exp/smoke_unique_id/esp/seed1024 --device cuda:0
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python -B -m reproduce.run_matrix --run-id formal_unique_id --devices cuda:0
```

Always new IDs. Matrix does nine datasets × five seeds, separate-process replay, then strict summary.csv/json/LaTeX and table2/table3 grouped outputs. Monitor log/<run_id>/console.log for queue, log/<run_id>/<dataset>/seed<seed>/console.log for epochs, exp/<run_id>/status.json for failure/completion. Start matrix in durable tmux bound to the assigned GPU. Formal run cannot accept shortened budgets or alternate seeds/configs. Runtime outputs ignored by Git; commit core only.

MV-ACM public sparse graph-to-feature processing can take several minutes on CPU before epoch1; the initial data_prepare_begin event and queue status identify this stage. Formal source snapshots include final environment documentation; short functional witnesses retain their earlier complete snapshots.
