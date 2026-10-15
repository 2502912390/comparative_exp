"""Freeze and execute 20 native-default jobs; stop with evidence on any failure."""
import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.coe_benchmark import plan_method, save_plan, write_json
from shared.coe_benchmark.artifacts import source_manifest
from shared.coe_benchmark.protocol import json_digest
from table2.MMDynamics_2022.reproduce.common import (
    METHOD_DIR, REPO_ROOT, EXECUTOR_MODEL, read_config, source_files)
from table2.MMDynamics_2022.reproduce.run import Tee
from table2.MMDynamics_2022.reproduce.summarize import summarize


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU index")
    parser.add_argument("--phase", choices=("formal",), default="formal")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1024, 1029)))
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--preflight-run", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.devices != ["cuda:0"]:
        parser.error("Serial queue binds one physical GPU as cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    preflight = json.loads((args.preflight_run / "preflight.json").read_text())
    if preflight["status"] != "passed":
        raise ValueError("Required actual short training/replay did not pass")
    frozen_sources = source_manifest(source_files())
    config = read_config()
    plan = plan_method("MMDynamics", args.run_id, devices=args.devices, seeds=tuple(args.seeds),
        python=sys.executable, config_by_dataset={ds: METHOD_DIR / "configs/default.json"
        for ds in ("esp", "flickr", "iapr", "nus")})
    plan.update(parameter_policy="author_default_no_hpo", config=config,
        config_sha256=json_digest(config), sources=frozen_sources,
        executor_model=EXECUTOR_MODEL, independent_reviewer="not_run",
        preflight=str(args.preflight_run.resolve()), physical_gpu=args.gpu)
    root = METHOD_DIR / "exp" / plan["run_id"]
    plan_path = root / "planned_jobs.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing plan differs from requested frozen run")
    else:
        save_plan(plan)
    log_root = METHOD_DIR / "log" / plan["run_id"]
    log_root.mkdir(parents=True, exist_ok=True)
    state = {"status": "ready" if args.plan_only else "running", "method": "MMDynamics",
        "stage": "formal", "run_id": plan["run_id"], "expected_jobs": len(plan["jobs"]),
        "completed_jobs": 0, "physical_gpu": args.gpu, "controller_pid": os.getpid(),
        "current_worker": None, "executor_model": EXECUTOR_MODEL,
        "independent_reviewer": "not_run", "parameter_policy": "author_default_no_hpo"}
    def update(**values):
        state.update(values, updated_at=utc_now())
        write_json(root / "queue_state.json", state)
    if args.plan_only:
        update()
        print(plan_path)
        return
    # Refuse a duplicate controller or an accidental restart into historical jobs.
    if any((root / job["dataset"] / f"seed{job['seed']}").exists() for job in plan["jobs"]):
        raise FileExistsError("Historical job directories exist; inspect them instead of overwriting")
    used = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    if int(used) >= 500:
        update(status="awaiting_gpu", memory_used_mib=int(used))
        print("GPU occupied; frozen plan retained. Rerun this command when available.")
        return
    with (log_root / "pipeline.log").open("a") as pipeline:
        with contextlib.redirect_stdout(Tee(sys.stdout, pipeline)), contextlib.redirect_stderr(Tee(sys.stderr, pipeline)):
            try:
                update()
                for job in plan["jobs"]:
                    if source_manifest(source_files()) != frozen_sources:
                        raise ValueError("Core sources changed after plan freeze")
                    dataset, seed = job["dataset"], job["seed"]
                    for label, argv in (("train", job["argv"]), ("replay", [sys.executable, "-u",
                        str(METHOD_DIR / "reproduce/replay.py"), "--job-dir",
                        str(root / dataset / f"seed{seed}"), "--device", "cuda:0"])):
                        argv = list(argv)
                        if label == "train":
                            argv.insert(1, "-u")
                        worker_log = log_root / f"{dataset}_seed{seed}_{label}.log"
                        with worker_log.open("x") as output:
                            process = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
                            update(current_worker={"pid": process.pid, "description": f"{dataset}_seed{seed} {label}", "log": str(worker_log)})
                            print(json.dumps({"event": "worker_started", "time": utc_now(), **state["current_worker"]}), flush=True)
                            for line in process.stdout:
                                output.write(line)
                                output.flush()
                                print(line, end="", flush=True)
                            code = process.wait()
                            if code:
                                raise RuntimeError(f"{dataset} seed{seed} {label} exited {code}; see {worker_log}")
                    update(completed_jobs=state["completed_jobs"] + 1, current_worker=None)
                    print(json.dumps({"event": "job_complete", "dataset": dataset, "seed": seed, "time": utc_now()}), flush=True)
                summary = summarize(root)
                if summary["status"] != "complete":
                    raise ValueError("All workers finished but summary failed completion gates")
                update(status="complete", current_worker=None)
                print(json.dumps({"event": "MMDYNAMICS_DEFAULTS_COMPLETE", "summary": str(root / "summary.json"), "time": utc_now()}), flush=True)
            except Exception as error:
                update(status="failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
