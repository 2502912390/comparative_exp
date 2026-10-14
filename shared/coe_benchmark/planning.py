"""Job planning only: method runners own their architecture and training."""
from pathlib import Path
import json
import shlex
from .io import validate_run_id, write_json
from .protocol import REPO_ROOT, SEEDS, PROTOCOL_ID

COE_PYTHON = "/home/chenzhichao/miniconda3/envs/coe/bin/python"


def plan_method(method, run_id, *, devices=("cpu",), phase="formal", seeds=SEEDS,
                python=COE_PYTHON, config_by_dataset=None):
    registry = json.loads((REPO_ROOT / "registry/methods.json").read_text())
    if registry["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Method registry protocol mismatch")
    item = next((record for record in registry["methods"] if record["method"] == method), None)
    if item is None:
        raise ValueError(f"Unregistered method: {method}")
    validate_run_id(run_id)
    if phase not in ("smoke", "dev", "formal") or not devices:
        raise ValueError("Explicit phase/devices required")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if phase == "formal" and seeds != SEEDS:
        raise ValueError("Formal plan must use exactly 1024-1028 in order")
    if not run_id.startswith(phase + "_"):
        run_id = phase + "_" + run_id
    method_dir = REPO_ROOT / item["directory"]
    jobs = []
    for dataset in item["datasets"]:
        for seed in seeds:
            device = devices[len(jobs) % len(devices)]
            command = [str(python), str(method_dir / "reproduce/run.py"),
                       "--dataset", dataset, "--seed", str(seed), "--device", device,
                       "--phase", phase, "--run-id", run_id,
                       "--output-root", str(method_dir / "exp")]
            if config_by_dataset:
                if dataset not in config_by_dataset:
                    raise ValueError(f"Missing frozen configuration: {dataset}")
                command.extend(["--config", str(config_by_dataset[dataset])])
            jobs.append({"dataset": dataset, "seed": seed, "device": device,
                         "argv": command, "command": shlex.join(command)})
    return {"protocol_id": PROTOCOL_ID, "method": method,
            "method_dir": str(method_dir), "run_id": run_id, "phase": phase,
            "datasets": item["datasets"], "seeds": list(seeds), "jobs": jobs,
            "status": "plan_only_not_executed",
            "run_entrypoint_exists": (method_dir / "reproduce/run.py").is_file()}


def save_plan(plan):
    root = Path(plan["method_dir"]) / "exp" / plan["run_id"]
    path = root / "planned_jobs.json"
    if path.exists():
        raise FileExistsError("A frozen plan cannot be overwritten")
    write_json(path, plan)
    return path

