"""Persisted native checks and five full-node short trainings/fresh replays."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shared.coe_benchmark import write_json
from table1.Mowst_2024.reproduce.native import METHOD_DIR, DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_id = args.run_id if args.run_id.startswith("smoke_") else "smoke_" + args.run_id
    log = METHOD_DIR / "log" / run_id
    log.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    commands = [("native_checks", [sys.executable, "-u", str(METHOD_DIR / "reproduce/checks.py"), "--device", "cuda:0"])]
    for dataset in DATASETS:
        commands += [(dataset + "_train", [sys.executable, "-u", str(METHOD_DIR / "reproduce/run.py"),
            "--dataset", dataset, "--seed", "1024", "--device", "cuda:0", "--phase", "smoke",
            "--run-id", run_id, "--config", str(METHOD_DIR / "configs/smoke.json")]),
            (dataset + "_replay", [sys.executable, "-u", str(METHOD_DIR / "reproduce/replay.py"),
            "--job-dir", str(METHOD_DIR / "exp" / run_id / dataset / "seed1024"), "--device", "cuda:0"])]
    started = time.monotonic()
    report = {"status": "running", "scope": "Native mechanism/default/label checks; two-update/two-round full-node fixtures, MC50; not formal results",
              "physical_gpu": args.physical_gpu, "executor_model": "current session GPT-6", "checks": []}
    for name, command in commands:
        print("PREFLIGHT_START", name, flush=True)
        check_start = time.monotonic()
        with (log / (name + ".log")).open("x", buffering=1) as output:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
        report["checks"].append({"name": name, "returncode": completed.returncode,
            "log": str(log / (name + ".log")), "seconds": time.monotonic() - check_start})
        if completed.returncode:
            report["status"] = "failed"
            write_json(METHOD_DIR / "exp" / run_id / "preflight.json", report)
            raise RuntimeError(f"Preflight {name} failed; inspect {log / (name + '.log')}")
        print("PREFLIGHT_PASSED", name, flush=True)
    report.update(status="passed", elapsed_seconds=time.monotonic() - started)
    write_json(METHOD_DIR / "exp" / run_id / "preflight.json", report)
    print("MOWST_PREFLIGHT_ALL5_PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
