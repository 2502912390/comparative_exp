"""Direct fixed-author-configuration 25-job Mowst queue with independent replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shared.coe_benchmark import plan_method, save_plan, write_json
from shared.coe_benchmark.protocol import SEEDS
from table1.Mowst_2024.reproduce.native import METHOD_DIR, DATASETS
from table1.Mowst_2024.reproduce.artifacts import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    run_id = args.run_id if args.run_id.startswith("formal_") else "formal_" + args.run_id
    root = METHOD_DIR / "exp" / run_id
    root.mkdir(parents=True, exist_ok=False)
    log = METHOD_DIR / "log" / run_id
    log.mkdir(parents=True, exist_ok=True)
    plan = plan_method("Mowst", run_id, devices=("cuda:0",), python=sys.executable,
        config_by_dataset={dataset: METHOD_DIR / "configs/default.json" for dataset in DATASETS})
    for job in plan["jobs"]:
        job["environment"] = {"CUDA_VISIBLE_DEVICES": str(args.physical_gpu)}
    save_plan(plan)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    state = {"status": "running", "stage": "formal", "method": "Mowst", "variant": "Mowst-GCN",
        "controller_pid": os.getpid(), "run_id": run_id, "physical_gpu": args.physical_gpu,
        "parameter_policy": "author_default_no_hpo", "expected_jobs": 25, "completed_jobs": 0,
        "current_worker": None, "executor_model": "current session GPT-6", "independent_reviewer": "not_run"}

    def record(event, **details):
        now = datetime.now(timezone.utc).isoformat()
        state["updated_at"] = now
        write_json(root / "queue_state.json", state)
        print(json.dumps({"time": now, "event": event, **details}), flush=True)

    def worker(command, path, description):
        with path.open("x", buffering=1) as output:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            state["current_worker"] = {"pid": process.pid, "description": description, "log": str(path)}
            record("worker_started", **state["current_worker"])
            for line in process.stdout:
                output.write(line)
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"{description} failed (exit={code}); inspect {path}")

    record("author_defaults_queue_started", datasets=list(DATASETS), seeds=list(SEEDS), jobs=25,
           inference="50 native Bernoulli-route draws, mean ACC per training seed")
    try:
        for job in plan["jobs"]:
            label = f"{job['dataset']}_seed{job['seed']}"
            worker([sys.executable, "-u", *job["argv"][1:]], log / (label + ".log"), label)
            directory = root / job["dataset"] / f"seed{job['seed']}"
            worker([sys.executable, "-u", str(METHOD_DIR / "reproduce/replay.py"),
                "--job-dir", str(directory), "--device", "cuda:0"],
                log / (label + "_replay.log"), label + " replay")
            state["completed_jobs"] += 1
            state["current_worker"] = None
            record("job_complete", dataset=job["dataset"], seed=job["seed"])
        summary = summarize(root)
        state["status"] = summary["status"]
        record("MOWST_DEFAULTS_COMPLETE", summary=str(root / "summary.json"))
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        record("MOWST_ERROR", error=repr(error))
        raise


if __name__ == "__main__":
    main()
