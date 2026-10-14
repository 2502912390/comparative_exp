"""Validation-only strict-improvement checkpoint selection; earliest ties survive."""
from pathlib import Path
import math
import os
import tempfile
import torch
from .io import write_json
from .protocol import sha256_file
from .seed import capture_rng

SELECTION_RULE = "outer_validation_accuracy_strict_improvement_earliest_tie"


def write_selection_record(path, selector):
    """Native serializers call this after writing their selected checkpoint file."""
    path = Path(path)
    if selector.best_epoch is None or not path.is_file():
        raise ValueError("A selected checkpoint file is required")
    write_json(path.with_suffix(path.suffix + ".selection.json"),
               {"best_epoch": selector.best_epoch,
                "best_validation_acc": selector.best_acc,
                "selection_rule": SELECTION_RULE,
                "checkpoint_sha256": sha256_file(path)})


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


class ValidationSelector:
    """Framework-independent selection decision, with no test-metric argument."""
    def __init__(self, patience_checks=200):
        if not isinstance(patience_checks, int) or patience_checks < 0:
            raise ValueError("patience_checks must be nonnegative")
        self.patience_checks = patience_checks
        self.best_acc = -1.0
        self.best_epoch = None
        self.last_epoch = 0
        self.bad_checks = 0

    def consider(self, epoch, validation_acc):
        if not isinstance(epoch, int) or epoch <= self.last_epoch:
            raise ValueError("Validation epochs must be positive and strictly increasing")
        if not math.isfinite(validation_acc) or not 0 <= validation_acc <= 1:
            raise ValueError("Validation ACC must be a finite fraction in [0,1]")
        self.last_epoch = epoch
        improved = validation_acc > self.best_acc
        if improved:
            self.best_acc, self.best_epoch, self.bad_checks = float(validation_acc), epoch, 0
        else:
            self.bad_checks += 1
        return improved

    @property
    def should_stop(self):
        return bool(self.patience_checks and self.bad_checks >= self.patience_checks)


class BestCheckpoint(ValidationSelector):
    """PyTorch serializer layered on the shared selection rule.

    Non-PyTorch methods can use ValidationSelector with their native serializer.
    """
    def __init__(self, path, patience_checks=200):
        super().__init__(patience_checks)
        self.path = Path(path)
        if self.path.exists() or self.path.with_suffix(self.path.suffix + ".selection.json").exists():
            raise FileExistsError("A new selector cannot overwrite a historical best checkpoint")

    def consider(self, epoch, validation_acc, payload):
        if not super().consider(epoch, validation_acc):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".best.", dir=self.path.parent)
        os.close(descriptor)
        try:
            torch.save({"payload": cpu_tree(payload), "epoch": self.best_epoch,
                        "validation_acc": self.best_acc, "selection_rule": SELECTION_RULE,
                        "rng_state": capture_rng()}, temporary)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        write_selection_record(self.path, self)
        return True


def load_best(path):
    """Load only a caller's trusted local checkpoint, never a reference weight."""
    path = Path(path)
    import json
    selected = json.loads(path.with_suffix(path.suffix + ".selection.json").read_text())
    if sha256_file(path) != selected["checkpoint_sha256"]:
        raise ValueError("Best checkpoint hash differs from its selection record")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint["selection_rule"] != SELECTION_RULE
            or checkpoint["epoch"] != selected["best_epoch"]
            or checkpoint["validation_acc"] != selected["best_validation_acc"]):
        raise ValueError("Checkpoint selection metadata mismatch")
    return checkpoint

