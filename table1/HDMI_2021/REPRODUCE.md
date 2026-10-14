# Run HDMI

Use the existing coe environment. No installation is required by the isolated
HDMI path. Limit CPU threads; CUDA_VISIBLE_DEVICES binds each worker to one
physical GPU. A physical GPU must be idle before allocating it.

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/chenzhichao/miniconda3/envs/coe/bin/python -u table1/HDMI_2021/reproduce/checks.py --device cuda:0
```

Smoke (three unsupervised updates + four probe updates; this is not a formal
paper result), followed by a separate reconstruction/inference process:

```bash
CUDA_VISIBLE_DEVICES=7 /home/chenzhichao/miniconda3/envs/coe/bin/python -u table1/HDMI_2021/reproduce/run.py --dataset acm --seed 1024 --device cuda:0 --phase smoke --run-id smoke_unique --config table1/HDMI_2021/configs/smoke.json
CUDA_VISIBLE_DEVICES=7 /home/chenzhichao/miniconda3/envs/coe/bin/python -u table1/HDMI_2021/reproduce/replay.py --job-dir table1/HDMI_2021/exp/smoke_unique/acm/seed1024 --device cuda:0
```

The complete pipeline takes a fresh unique run-id. The controller interprets
--devices as physical GPU numbers, runs child jobs with CUDA_VISIBLE_DEVICES
set to that GPU and local --device cuda:0, and waits for idle resources.

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python -u table1/HDMI_2021/reproduce/run_matrix.py --run-id formal_hdmi_unique --devices cuda:7 --phase formal --seeds 1024 1025 1026 1027 1028
```

The controller preregisters 12 candidates on every dataset, executes all
60 validation-only development runs, freezes five configurations, then
executes all 25 formal jobs from scratch. Each formal job gets an independent
replay process. Do not edit core sources/configuration during the queue.
--resume uses the recorded plan and retained outputs; it does not infer scope
from successes or ignore failed/low-scoring seeds.

Monitor log/<formal_run_id>/pipeline.log and exp/<formal_run_id>/queue_state.json.
Individual logs use log/<run_id>/<dataset>/seed<seed>/console.log; development
run-ids include the candidate. Formal results and final summary.json,
summary.csv, table_rows.tex use exp/<formal_run_id>.

Summary can be regenerated after completion:

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python table1/HDMI_2021/reproduce/summarize.py --run-root table1/HDMI_2021/exp/formal_hdmi_unique
```

The source/config/environment hashes, checkpoint selection and exact test
class predictions must agree. Independent subagent review was not run under
the user's model spending policy; local numerical checks and fresh-process
checkpoint replay are recorded separately.
