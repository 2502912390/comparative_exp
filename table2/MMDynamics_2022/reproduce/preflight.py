"""Actual coe mechanism checks and full-node short training/replay on all four datasets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from shared.coe_benchmark import write_json
from shared.coe_benchmark.artifacts import environment_metadata
from table2.MMDynamics_2022.reproduce.common import METHOD_DIR, REPO_ROOT, DATASETS, EXECUTOR_MODEL
from table2.MMDynamics_2022.reproduce.checks import check_mechanism, check_data


def main():
    torch.set_num_threads(2)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.run_id.startswith("smoke_"):
        parser.error("preflight run-id must start with smoke_")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Host CUDA witness required for requested GPU; no fallback")
    root = METHOD_DIR / "exp" / args.run_id
    if root.exists():
        raise FileExistsError("Do not reuse a smoke run-id")
    start = time.monotonic()
    report = {"executor_model": EXECUTOR_MODEL, "independent_reviewer": "not_run",
              "environment": environment_metadata(), "status": "running"}
    root.mkdir(parents=True)
    try:
        report["mechanism"] = check_mechanism(args.device)
        report["canonical_data"] = check_data()
        for dataset in DATASETS:
            print("SHORT_TRAIN_AND_REPLAY", dataset, flush=True)
            subprocess.run([sys.executable, "-u", str(METHOD_DIR / "reproduce/run.py"),
                "--dataset", dataset, "--seed", "1024", "--device", args.device,
                "--phase", "smoke", "--run-id", args.run_id, "--smoke-updates", "3"],
                cwd=REPO_ROOT, check=True)
            subprocess.run([sys.executable, "-u", str(METHOD_DIR / "reproduce/replay.py"),
                "--job-dir", str(root / dataset / "seed1024"), "--device", args.device],
                cwd=REPO_ROOT, check=True)
        report.update(status="passed", short_training_and_fresh_process_replay=list(DATASETS),
                      elapsed_seconds=time.monotonic() - start)
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(root / "preflight.json", report)
    print("MMDYNAMICS_PREFLIGHT_PASSED", json.dumps({"datasets": list(DATASETS),
        "device": args.device, "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
