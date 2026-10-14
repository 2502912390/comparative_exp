"""Result export, identity recounting and caller-supplied model replay verification."""
from pathlib import Path
import json
import platform
import subprocess
import sys
from importlib import metadata as package_metadata
import numpy as np
import torch

from .data import verify_identity
from .fingerprints import label_identity, split_fingerprint
from .io import write_json
from .protocol import (PROTOCOL_ID, DATA_PROTOCOL_ID, IMPLEMENTATION_VERSION, REPO_ROOT,
                       sha256_file, json_digest, read_registry, dataset_record)
from .selection import SELECTION_RULE


def source_manifest(source_files):
    """Caller lists all method sources; shared implementation and registries are automatic."""
    package = Path(__file__).resolve().parent
    files = {Path(path).resolve() for path in source_files}
    if not files:
        raise ValueError("List the method's core source files explicitly")
    files.update(package.glob("*.py"))
    files.add(package / "provenance.json")
    files.add(REPO_ROOT / "registry/reference_datasets.json")
    files.add(REPO_ROOT / "registry/methods.json")
    result = {}
    for path in sorted(files):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError("Method sources must belong to the local baseline repository") from exc
        if any(part in {"exp", "log", "logs", ".git", ".upstream_git"} for part in relative.parts):
            raise ValueError("Run artifacts cannot be declared as core source")
        result[str(relative)] = sha256_file(path)
    return {"files": result, "sha256": json_digest(result)}


def environment_metadata():
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in package_metadata.distributions()
                if distribution.metadata.get("Name")}
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "packages": dict(sorted(packages.items())), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available()}


def build_manifest(data, paths, *, method, config, source_files, adaptation,
                   parameter_count, device, author_commit=None):
    verify_identity(data)
    if paths.dataset != data.dataset:
        raise ValueError("Run directory and data identity differ")
    sources = source_manifest(source_files)
    def git(*args):
        return subprocess.check_output(["rtk", "git", "-C", str(REPO_ROOT), *args],
                                       text=True).strip()
    metadata = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "data_protocol_id": DATA_PROTOCOL_ID,
        "shared_implementation_version": IMPLEMENTATION_VERSION,
        "method": method, "dataset": data.dataset, "seed": paths.seed,
        "phase": paths.phase, "run_id": paths.run_id, "data": data.identity(),
        "config": config, "config_sha256": json_digest(config), "sources": sources,
        "adaptation": adaptation, "parameters": int(parameter_count), "device": str(device),
        "author_commit": author_commit, "selection_rule": SELECTION_RULE,
        "environment": environment_metadata(),
        "git": {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short")},
        "paper_eligible": paths.phase == "formal", "executor": "local method runner",
    }
    write_json(paths.exp / "manifest.json", metadata)
    snapshot = paths.exp / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    for name, expected in sources["files"].items():
        source, destination = REPO_ROOT / name, snapshot / name
        if sha256_file(source) != expected:
            raise ValueError("Source changed while creating run snapshot")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    write_json(snapshot / "source_manifest.json", sources)
    write_json(paths.exp / "environment.json", metadata["environment"])
    return metadata


def _numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def export_result(paths, data, *, checkpoint, best_epoch, best_validation_acc,
                  node_ids, predictions, scores=None, score_kind=None, timing=None):
    """Export test or all-node predictions, always retaining full label/mask identity."""
    verify_identity(data)
    manifest_path = paths.exp / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["dataset"], manifest["seed"], manifest["phase"]) != (data.dataset, paths.seed, paths.phase):
        raise ValueError("Manifest does not identify this run")
    if best_epoch < 1 or not np.isfinite(best_validation_acc) or not 0 <= best_validation_acc <= 1:
        raise ValueError("Invalid validation checkpoint selection")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or checkpoint.parent != paths.exp.resolve():
        raise ValueError("Checkpoint must be a file inside this run's exp directory")
    selected_path = checkpoint.with_suffix(checkpoint.suffix + ".selection.json")
    selected = json.loads(selected_path.read_text())
    if (selected["best_epoch"] != best_epoch
            or selected["best_validation_acc"] != best_validation_acc
            or selected["selection_rule"] != SELECTION_RULE
            or selected["checkpoint_sha256"] != sha256_file(checkpoint)):
        raise ValueError("Export must use the checkpoint selected on validation ACC")
    ids, predicted = _numpy(node_ids), _numpy(predictions)
    if ids.dtype.kind not in "iu" or predicted.dtype.kind not in "iu":
        raise ValueError("Node IDs and predictions must be integer arrays")
    ids, predicted = ids.astype(np.int64), predicted.astype(np.int64)
    test_ids = data.indices("test").cpu().numpy()
    if (ids.ndim != 1 or predicted.shape != ids.shape
            or len(np.unique(ids)) != len(ids) or len(ids) == 0
            or ids.min() < 0 or ids.max() >= data.nodes
            or predicted.min() < 0 or predicted.max() >= data.classes):
        raise ValueError("Invalid/misaligned node IDs or class predictions")
    order = np.argsort(ids)
    ids, predicted = ids[order], predicted[order]
    if not (np.array_equal(ids, test_ids) or np.array_equal(ids, np.arange(data.nodes))):
        raise ValueError("Export exactly the test nodes or all nodes; do not drop test samples")
    labels = data.labels.cpu().numpy()
    artifact = {"node_ids": ids, "y_true": labels[ids], "y_pred": predicted,
                "labels": labels, "class_ids": np.arange(data.classes),
                "class_mapping_json": np.asarray(json.dumps(list(data.class_mapping), ensure_ascii=False)),
                "canonical_source_fingerprint": np.asarray(data.fingerprints["source"]),
                "split_fingerprint": np.asarray(data.fingerprints["splits"]),
                "labels_fingerprint": np.asarray(data.fingerprints["labels_sha256"]),
                "class_mapping_fingerprint": np.asarray(data.fingerprints["class_mapping_sha256"])}
    artifact.update({name + "_mask": mask.cpu().numpy() for name, mask in data.masks.items()})
    if scores is not None:
        scores = _numpy(scores)
        if scores.shape != (len(ids), data.classes) or not np.isfinite(scores).all():
            raise ValueError("Scores must be finite [exported_nodes, classes]")
        scores = scores[order]
        if score_kind not in {"logits", "probabilities", "evidence"}:
            raise ValueError("Declare the original meaning of scores")
        if not np.array_equal(scores.argmax(-1), predicted):
            raise ValueError("Score argmax and supplied predictions differ")
        if score_kind == "probabilities" and (
                (scores < 0).any() or not np.allclose(scores.sum(-1), 1, atol=1e-5)):
            raise ValueError("Invalid probability simplex")
        if score_kind == "evidence" and (scores < 0).any():
            raise ValueError("Evidence must be nonnegative; it is not a logit/probability")
        artifact["scores"], artifact["score_kind"] = scores, np.asarray(score_kind)
    elif score_kind is not None:
        raise ValueError("score_kind without scores")
    prediction_path = paths.exp / "predictions.npz"
    if prediction_path.exists() or (paths.exp / "result.json").exists():
        raise FileExistsError("Results cannot overwrite a previous final test evaluation")
    np.savez_compressed(prediction_path, **artifact)
    is_test = data.masks["test"].cpu().numpy()[ids]
    correct, total = int((predicted[is_test] == labels[ids[is_test]]).sum()), len(test_ids)
    result = {"schema_version": 1, "status": "complete_single_seed",
              "method": manifest["method"], "dataset": data.dataset, "seed": paths.seed,
              "phase": paths.phase, "protocol_id": PROTOCOL_ID,
              "test": {"correct": correct, "total": total, "acc": correct / total},
              "best_epoch": int(best_epoch), "best_validation_acc": float(best_validation_acc),
              "selection_rule": SELECTION_RULE,
              "manifest_sha256": sha256_file(manifest_path),
              "checkpoint": checkpoint.name, "checkpoint_sha256": sha256_file(checkpoint),
              "selection_record_sha256": sha256_file(selected_path),
              "predictions_sha256": sha256_file(prediction_path), "timing": timing or {},
              "model_replay_verified": False}
    write_json(paths.exp / "result.json", result)
    recount_result(paths.exp)
    return result


def recount_result(job_dir, *, registry=None):
    """Recompute saved test ACC and all label/split identities; does not run a model."""
    job_dir = Path(job_dir)
    registry = read_registry() if registry is None else registry
    manifest = json.loads((job_dir / "manifest.json").read_text())
    result = json.loads((job_dir / "result.json").read_text())
    record = dataset_record(manifest["dataset"], registry)
    if manifest["protocol_id"] != PROTOCOL_ID or result["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Mixed or outdated protocol")
    if result["status"] != "complete_single_seed":
        raise ValueError("Only completed single-seed records can be recounted")
    for key in ("method", "dataset", "seed", "phase"):
        if result[key] != manifest[key]:
            raise ValueError(f"Result/manifest mismatch: {key}")
    for name, expected in (
        ("manifest.json", result["manifest_sha256"]),
        (result["checkpoint"], result["checkpoint_sha256"]),
        (result["checkpoint"] + ".selection.json", result["selection_record_sha256"]),
        ("predictions.npz", result["predictions_sha256"]),
    ):
        if Path(name).name != name or sha256_file(job_dir / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    identity = manifest["data"]
    selected = json.loads((job_dir / (result["checkpoint"] + ".selection.json")).read_text())
    if (selected["best_epoch"] != result["best_epoch"]
            or selected["best_validation_acc"] != result["best_validation_acc"]
            or selected["checkpoint_sha256"] != result["checkpoint_sha256"]
            or selected["selection_rule"] != SELECTION_RULE):
        raise ValueError("Validation selection record mismatch")
    expected = {"source": record["canonical_source_fingerprint"],
                "splits": record["split_fingerprint"], **record["diagnostic_identity"]}
    for key, value in expected.items():
        if identity["fingerprints"].get(key) != value:
            raise ValueError(f"Manifest canonical identity mismatch: {key}")
    if manifest["config_sha256"] != json_digest(manifest["config"]):
        raise ValueError("Manifest configuration digest mismatch")
    sources = manifest["sources"]
    if sources["sha256"] != json_digest(sources["files"]):
        raise ValueError("Source list digest mismatch")
    snapshot = job_dir / "source_snapshot"
    for name, expected_hash in sources["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid source snapshot path")
        if sha256_file(snapshot / name) != expected_hash:
            raise ValueError(f"Source snapshot differs: {name}")
    with np.load(job_dir / "predictions.npz", allow_pickle=False) as saved:
        labels = saved["labels"]
        if labels.dtype != np.int64 or labels.shape != (record["nodes"],):
            raise ValueError("Saved full label vector schema mismatch")
        mapping = json.loads(str(saved["class_mapping_json"].item()))
        if label_identity(torch.from_numpy(labels), mapping) != record["diagnostic_identity"]:
            raise ValueError("Saved labels/class mapping identity mismatch")
        masks = {name: torch.from_numpy(saved[name + "_mask"].copy()) for name in ("train", "val", "test")}
        if split_fingerprint(masks) != record["split_fingerprint"]:
            raise ValueError("Saved outer split identity mismatch")
        if str(saved["canonical_source_fingerprint"].item()) != record["canonical_source_fingerprint"]:
            raise ValueError("Prediction source identity mismatch")
        for key, expected_hash in (
            ("split_fingerprint", record["split_fingerprint"]),
            ("labels_fingerprint", record["diagnostic_identity"]["labels_sha256"]),
            ("class_mapping_fingerprint", record["diagnostic_identity"]["class_mapping_sha256"]),
        ):
            if str(saved[key].item()) != expected_hash:
                raise ValueError(f"Prediction identity mismatch: {key}")
        if not np.array_equal(saved["class_ids"], np.arange(record["classes"])):
            raise ValueError("Class ID order mismatch")
        ids, predicted = saved["node_ids"], saved["y_pred"]
        test_ids = masks["test"].nonzero(as_tuple=False).flatten().numpy()
        if ids.dtype != np.int64 or predicted.dtype != np.int64 or ids.ndim != 1 or predicted.shape != ids.shape:
            raise ValueError("Saved prediction schema mismatch")
        if not (np.array_equal(ids, test_ids) or np.array_equal(ids, np.arange(record["nodes"]))):
            raise ValueError("Saved predictions omit/misorder test nodes")
        if (predicted < 0).any() or (predicted >= record["classes"]).any():
            raise ValueError("Saved prediction class ID out of range")
        if not np.array_equal(saved["y_true"], labels[ids]):
            raise ValueError("Saved y_true differs from frozen labels")
        selected = masks["test"].numpy()[ids]
        correct, total = int((predicted[selected] == labels[ids[selected]]).sum()), len(test_ids)
        if "scores" in saved.files:
            scores = saved["scores"]
            kind = str(saved["score_kind"].item())
            if (kind not in {"logits", "probabilities", "evidence"}
                    or scores.shape != (len(ids), record["classes"])
                    or not np.isfinite(scores).all()
                    or not np.array_equal(scores.argmax(-1), predicted)):
                raise ValueError("Saved score semantics/schema mismatch")
            if kind == "probabilities" and ((scores < 0).any() or not np.allclose(scores.sum(-1), 1, atol=1e-5)):
                raise ValueError("Saved probability simplex mismatch")
            if kind == "evidence" and (scores < 0).any():
                raise ValueError("Saved evidence must be nonnegative")
    if result["selection_rule"] != SELECTION_RULE or manifest["selection_rule"] != SELECTION_RULE:
        raise ValueError("Wrong checkpoint selection rule")
    if (result["test"]["correct"], result["test"]["total"]) != (correct, total):
        raise ValueError("Saved test event counts differ from predictions")
    if not np.isclose(result["test"]["acc"], correct / total, atol=1e-12, rtol=0):
        raise ValueError("Saved ACC differs from predictions")
    return {"manifest": manifest, "result": result, "acc": correct / total,
            "correct": correct, "total": total}


def verify_model_replay(job_dir, *, node_ids, predictions, scores=None, atol=1e-6):
    """Method's own replay supplies freshly inferred outputs from its saved model."""
    job_dir = Path(job_dir)
    checked = recount_result(job_dir)
    ids, predicted = _numpy(node_ids), _numpy(predictions)
    if ids.dtype.kind not in "iu" or predicted.dtype.kind not in "iu":
        raise ValueError("Replay IDs/predictions must be integer arrays")
    order = np.argsort(ids)
    ids, predicted = ids[order], predicted[order]
    with np.load(job_dir / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(ids, saved["node_ids"]) or not np.array_equal(predicted, saved["y_pred"]):
            raise ValueError("Model replay differs from exported predictions")
        if scores is not None:
            fresh = _numpy(scores)[order]
            if "scores" not in saved.files or not np.allclose(fresh, saved["scores"], atol=atol, rtol=0):
                raise ValueError("Model replay scores differ")
    result = checked["result"]
    result["model_replay_verified"] = True
    write_json(job_dir / "result.json", result)
    replay = {"status": "verified", "kind": "caller_supplied_model_inference",
              "checkpoint_sha256": result["checkpoint_sha256"],
              "predictions_sha256": result["predictions_sha256"],
              "manifest_sha256": result["manifest_sha256"],
              "scope": "Exact class predictions; score comparison only if supplied."}
    write_json(job_dir / "replay.json", replay)
    return replay

