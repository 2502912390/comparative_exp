# Fixed-default HAN reproduction

Read root REPRODUCTION_PROTOCOL_AND_PROMPTS.md, shared/README.md and ADAPTATION.md.
Current comparison uses released HeteGAT_multi defaults:64 width, eight8-channel
heads, semantic128, lr.005, L2.001, both dropouts.6, **200epochs/patience100**.
Native inclusive validation-Acc-or-CE stopping is preserved; selected checkpoint
still follows strict validation Acc improvement with earliest tie. No HPO.

Run from the unique method directory, using original coe without installing packages:

```bash
cd /home/chenzhichao/Research/other_exp/table1/HAN_2019
export PYTHONPATH=/home/chenzhichao/Research/other_exp
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/chenzhichao/miniconda3/envs/coe/bin/python -m reproduce.checks --device cuda:0
```

Single full formal task using the fixed authored source defaults:

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python -m reproduce.run --dataset acm --seed 1024 --device cuda:0 --phase formal --run-id formal_unique_id --config configs/default.json
```

New matrix; use a fresh batch ID, preserve all older failed exp/log:

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python -u -m reproduce.run_matrix --run-id han_defaults_unique_id --devices cuda:0 --phase formal --seeds 1024 1025 1026 1027 1028
```

Matrix executes25 formal jobs, no development candidates. Each seed starts from
fresh initialization, saves validation-selected best, exports FP64 full-node logits,
and performs a fresh-object then independent-process score replay. The first complete
ACM seed1024 is a regression gate; only after its full training/replay succeeds does
the queue continue the remaining24. It never averages scores or modifies the paper.

FP32 training and FP64 GPU validation/export/replay retain all graph edges and model
operations. This repairs real FP32 sparse rounding errors without changing the public
1e-6 score assertion, rounding logits, or casting canonical features in place.
No author directory is added to sys.path; only the method namespace and shared root.

Independent replay and explicit later user-triggered statistics:

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python -m reproduce.replay --job-dir exp/formal_han_defaults_unique_id/acm/seed1024 --device cuda:0
/home/chenzhichao/miniconda3/envs/coe/bin/python -m reproduce.summarize --run-root exp/formal_han_defaults_unique_id --datasets acm dblp yelp amazon mag
```

Summary requires all five exact seeds with matched code/config/data/source/precision
and replay; sample std uses ddof1. The matrix does not call summary automatically.

## Persistent monitoring

Launch the matrix in detached tmux, redirect stdout/stderr to
log/formal_<batch>/console.log. Task logs are
log/formal_<batch>/tasks/formal_<dataset>_seed<seed>.log and replay_<dataset>_seed<seed>.log.
Each job also writes log/<run_id>/<dataset>/seed<seed>/console.log and exp/.../training.jsonl.
Stable state: exp/formal_<batch>/status.json (also progress.json).
Initial complete-ACM gate marker: ACM_REPLAY_GATE_PASSED.json.
Final markers: COMPLETED.json (awaiting user summary), or FAILED.json.
Failures preserve full evidence and stop; there is no smaller-graph fallback.

Source lock refuses changes during the formal matrix. output-root must stay in this
method exp. Config is always the complete fixed default specification; arbitrary
candidate/HPO config and dev phase are rejected. Short budgets are allowed only for
explicit smoke jobs. Old HPO queue failed and is not mixed into current results.
