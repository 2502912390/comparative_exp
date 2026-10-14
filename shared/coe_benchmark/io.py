"""Atomic metadata writes and method-local, non-overwriting run directories."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import tempfile
from .protocol import DATASETS, SEEDS

_PHASES = ("smoke", "dev", "formal")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id must be a safe nonempty slug")
    return run_id


@dataclass(frozen=True)
class RunPaths:
    method_dir: Path
    run_id: str
    dataset: str
    seed: int
    phase: str
    exp: Path
    log: Path

    @classmethod
    def create(cls, method_dir, run_id, dataset, seed, phase="formal"):
        method_dir = Path(method_dir).resolve()
        validate_run_id(run_id)
        if phase not in _PHASES or dataset not in DATASETS:
            raise ValueError("Unknown phase or dataset")
        if not isinstance(seed, int) or seed < 0 or (phase == "formal" and seed not in SEEDS):
            raise ValueError("Formal training seed must be one of the frozen five")
        prefix = phase + "_"
        if not run_id.startswith(prefix):
            run_id = prefix + run_id
        suffix = Path(run_id) / dataset / f"seed{seed}"
        exp, log = method_dir / "exp" / suffix, method_dir / "log" / suffix
        if exp.exists() or log.exists():
            raise FileExistsError(f"Run cannot overwrite existing outputs: {suffix}")
        exp.mkdir(parents=True, exist_ok=False)
        log.mkdir(parents=True, exist_ok=False)
        return cls(method_dir, run_id, dataset, seed, phase, exp, log)

