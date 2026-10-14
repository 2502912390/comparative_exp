"""Durable validation-only HPO -> frozen five-seed formal/replay queue."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from shared.coe_benchmark import plan_method, save_plan, summarize_run, write_summary, write_json
from shared.coe_benchmark.protocol import json_digest, sha256_file, PROTOCOL_ID, SEEDS
from table1.HDMI_2021.reproduce.model import METHOD_DIR
from table1.HDMI_2021.reproduce.run import core_sources, DATASETS

PYTHON = "/home/chenzhichao/miniconda3/envs/coe/bin/python"


class Queue:
    def __init__(self, args):
        self.args = args
        self.run_id = args.run_id if args.run_id.startswith(args.phase + "_") else args.phase + "_" + args.run_id
        self.root = METHOD_DIR / "exp" / self.run_id
        self.log = METHOD_DIR / "log" / self.run_id
        self.frozen = METHOD_DIR / "configs" / self.run_id
        self.state_file = self.root / "queue_state.json"
        self.event_file = self.root / "queue_events.jsonl"
        self.search = json.loads((METHOD_DIR / "configs/search_space.json").read_text())
        self.devices = [int(device.split(":")[-1]) for device in args.devices]
        if len(set(self.devices)) != len(self.devices) or any(device < 0 for device in self.devices):
            raise ValueError("Physical devices must be distinct nonnegative GPU indices")
        self.lock = None
        self.state = None

    def event(self, kind, **details):
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
        with self.event_file.open("a", buffering=1) as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def save(self):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["completed_development"] = sum(job["status"] == "complete" for job in self.state["development"])
        self.state["completed_formal"] = sum(job["status"] == "complete" for job in self.state["formal"])
        write_json(self.state_file, self.state)

    def initialize(self):
        self.log.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists():
            if not self.args.resume:
                raise FileExistsError("Queue exists; use --resume, never overwrite")
            self.state = json.loads(self.state_file.read_text())
            if self.state["search_space_sha256"] != json_digest(self.search):
                raise ValueError("Cannot resume a changed development search")
        else:
            if self.args.resume:
                raise FileNotFoundError("No queue state to resume")
            self.root.mkdir(parents=True, exist_ok=False)
            dev = []
            for dataset in DATASETS:
                for candidate in self.search["candidates"]:
                    dev.append({"dataset": dataset, "candidate_id": candidate["candidate_id"],
                        "run_id": "dev_" + self.run_id + "_" + candidate["candidate_id"],
                        "seed": 1024, "phase": "dev", "config": candidate["config"],
                        "status": "pending", "attempts": []})
            plan = plan_method("HDMI", self.run_id, devices=self.args.devices,
                config_by_dataset={dataset: self.frozen / (dataset + ".json") for dataset in DATASETS})
            for job in plan["jobs"]:
                physical = int(job["device"].split(":")[-1])
                job["physical_device"] = physical
                job["environment"] = {"CUDA_VISIBLE_DEVICES": str(physical)}
                job["argv"][job["argv"].index("--device") + 1] = "cuda:0"
                import shlex
                job["command"] = shlex.join(job["argv"])
            save_plan(plan)
            self.state = {"schema_version": 1, "method": "HDMI", "protocol_id": PROTOCOL_ID,
                "run_id": self.run_id, "stage": "development", "status": "running",
                "search_space_sha256": json_digest(self.search), "expected_datasets": list(DATASETS),
                "formal_seeds": list(SEEDS), "executor_model": "current session GPT-6",
                "independent_reviewer": "not_run", "development": dev,
                "formal": [{"dataset": job["dataset"], "seed": job["seed"],
                    "phase": "formal", "run_id": self.run_id, "status": "pending", "attempts": []}
                    for job in plan["jobs"]], "failures": []}
            from shared.coe_benchmark.artifacts import source_manifest
            self.state["development_source_lock"] = source_manifest(core_sources())["files"]
        self.lock = (self.root / "controller.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state["controller_pid"] = os.getpid()
        self.save()
        self.event("queue_started", run_id=self.run_id, pid=os.getpid(), devices=self.devices,
                   development_jobs=60, formal_jobs=25, resume=self.args.resume)

    def source_guard(self):
        expected_sources = self.state.get("formal_source_lock", self.state.get("development_source_lock", {}))
        for name, expected in expected_sources.items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError(f"Core source changed during {self.state['stage']} queue: {name}")

    def idle_gpu(self):
        last_notice = 0
        while True:
            for index in self.devices:
                raw = subprocess.check_output(["nvidia-smi", f"--id={index}",
                    "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                memory, utilization = [int(x.strip()) for x in raw.strip().split(",")]
                if memory < 500 and utilization < 5:
                    return index
            if time.monotonic() - last_notice > 60:
                self.event("waiting_for_idle_gpu", physical_devices=self.devices)
                last_notice = time.monotonic()
            time.sleep(5)

    def archive_attempt(self, job):
        # Keep failed artifacts under the same method/run. No reset, deletion,
        # model change, seed change or small-data fallback.
        suffix = Path(job["dataset"]) / f"seed{job['seed']}"
        count = len(job["attempts"])
        destinations = {}
        for kind in ("exp", "log"):
            source = METHOD_DIR / kind / job["run_id"] / suffix
            destination = METHOD_DIR / kind / job["run_id"] / "_failures" / suffix / f"attempt{count}"
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                destinations[kind] = str(destination)
        if job["attempts"]:
            job["attempts"][-1]["archived_outputs"] = destinations

    def worker(self, command, environment, log_file, description):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("x", buffering=1) as stream:
            child = subprocess.Popen(command, env=environment, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            self.state["current_worker"] = {"pid": child.pid, "description": description, "worker_log": str(log_file)}
            self.save()
            self.event("worker_started", pid=child.pid, description=description,
                       visible_gpu=environment["CUDA_VISIBLE_DEVICES"], log=str(log_file))
            last_notice = time.monotonic()
            while child.poll() is None:
                if time.monotonic() - last_notice >= 30:
                    tail = log_file.read_text(errors="replace").splitlines()[-1:]
                    self.event("worker_heartbeat", pid=child.pid, description=description, last_line=tail)
                    last_notice = time.monotonic()
                time.sleep(2)
            return child.returncode

    def run_job(self, job):
        if job["status"] == "complete":
            return
        self.source_guard()
        exp = METHOD_DIR / "exp" / job["run_id"] / job["dataset"] / f"seed{job['seed']}"
        if exp.exists():
            raise FileExistsError(f"Unfinished retained job requires inspection before retry: {exp}")
        if job["phase"] == "dev":
            config_file = self.root / "development_configs" / (job["candidate_id"] + ".json")
            if not config_file.exists():
                write_json(config_file, job["config"])
        else:
            config_file = self.frozen / (job["dataset"] + ".json")
        for retry in range(2):
            physical_gpu = self.idle_gpu()
            environment = dict(os.environ)
            environment.update(CUDA_VISIBLE_DEVICES=str(physical_gpu), OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
                COE_DATA_ROOT="/home/chenzhichao/Research/CoE2/data/CoE_dataset",
                COE_MULTIVIEW_DATA_ROOT="/home/chenzhichao/Research/CoE2/data/Multi-view datasets")
            attempt = {"index": len(job["attempts"]) + 1, "physical_gpu": physical_gpu,
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
            job["attempts"].append(attempt)
            job["status"] = "running"
            self.save()
            description = f"{job['phase']} {job['dataset']} seed{job['seed']} {job.get('candidate_id','')}"
            worker_log = self.log / "workers" / f"{job['phase']}_{job['dataset']}_{job.get('candidate_id','formal')}_seed{job['seed']}_attempt{attempt['index']}.log"
            command = [PYTHON, "-u", str(METHOD_DIR / "reproduce/run.py"),
                "--dataset", job["dataset"], "--seed", str(job["seed"]), "--device", "cuda:0",
                "--phase", job["phase"], "--run-id", job["run_id"],
                "--output-root", str(METHOD_DIR / "exp"), "--config", str(config_file)]
            returncode = self.worker(command, environment, worker_log, description)
            attempt["returncode"] = returncode
            if returncode:
                output = worker_log.read_text(errors="replace")
                attempt["status"] = "failed"
                job["status"] = "failed"
                self.state["failures"].append({"description": description, "returncode": returncode,
                    "worker_log": str(worker_log), "error_tail": output[-4000:]})
                self.save()
                self.event("worker_failed", description=description, returncode=returncode, log=str(worker_log))
                if retry == 0 and ("out of memory" in output.lower() or "cudaerroroutofmemory" in output.lower()):
                    self.archive_attempt(job)
                    self.save()
                    self.event("equivalent_oom_retry", description=description, same_config=True, same_seed=True)
                    continue
                raise RuntimeError(f"{description} failed; inspect {worker_log}")
            if job["phase"] == "formal":
                replay_command = [PYTHON, "-u", str(METHOD_DIR / "reproduce/replay.py"),
                                  "--job-dir", str(exp), "--device", "cuda:0"]
                replay_log = worker_log.with_name(worker_log.stem + "_replay.log")
                returncode = self.worker(replay_command, environment, replay_log, description + " replay")
                attempt["replay_log"] = str(replay_log)
                if returncode:
                    attempt["status"] = "replay_failed"
                    job["status"] = "replay_failed"
                    self.save()
                    raise RuntimeError(f"Checkpoint replay failed: {replay_log}")
            selected = json.loads((exp / "validation_result.json").read_text())
            job["best_validation_acc"] = selected["best_validation_acc"]
            job["config_sha256"] = selected["config_sha256"]
            job["status"] = "complete"
            attempt["status"] = "complete"
            attempt["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.save()
            self.event("job_complete", description=description, validation_acc=job["best_validation_acc"])
            return

    def freeze(self):
        choices = {}
        self.frozen.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            jobs = [job for job in self.state["development"] if job["dataset"] == dataset]
            if len(jobs) != 12 or any(job["status"] != "complete" for job in jobs):
                raise ValueError("Cannot freeze incomplete development scope")
            winner = max(jobs, key=lambda job: job["best_validation_acc"])
            path = self.frozen / (dataset + ".json")
            if path.exists():
                if json.loads(path.read_text()) != winner["config"]:
                    raise ValueError("Previously frozen configuration differs")
            else:
                write_json(path, winner["config"])
            choices[dataset] = {"candidate_id": winner["candidate_id"],
                "best_validation_acc": winner["best_validation_acc"],
                "config_sha256": winner["config_sha256"], "config_path": str(path),
                "development_run_id": winner["run_id"], "test_used": False}
        write_json(self.root / "configuration_selection.json", choices)
        # Shared registry/source hashes participate in every run's immutable
        # snapshot and therefore in the source guard as well.
        from shared.coe_benchmark.artifacts import source_manifest
        self.state["formal_source_lock"] = source_manifest(core_sources())["files"]
        write_json(self.root / "formal_source_lock.json", self.state["formal_source_lock"])
        self.state["stage"] = "formal"
        self.save()
        self.event("all_configs_frozen", choices=choices, fresh_formal_jobs=25)

    def execute(self):
        self.initialize()
        try:
            if self.state["stage"] == "development":
                for job in self.state["development"]:
                    self.run_job(job)
                self.freeze()
            for job in self.state["formal"]:
                self.run_job(job)
            summary = summarize_run(self.root, expected_datasets=DATASETS)
            write_summary(self.root, summary)
            if summary["status"] != "complete":
                raise ValueError("Formal summary is incomplete; no completion claim")
            self.state.update(stage="complete", status="complete", current_worker=None)
            self.save()
            self.event("HDMI_MATRIX_COMPLETE", summary=str(self.root / "summary.json"), formal_jobs=25)
        except BaseException as error:
            self.state.update(status="blocked_error", error=repr(error))
            self.save()
            self.event("QUEUE_ERROR", error=repr(error), state=str(self.state_file))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--devices", nargs="+", required=True, help="Physical GPU indices, e.g. cuda:7")
    parser.add_argument("--phase", choices=("formal",), default="formal")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS:
        parser.error("Formal scope is exactly seeds 1024--1028")
    Queue(args).execute()


if __name__ == "__main__":
    main()
