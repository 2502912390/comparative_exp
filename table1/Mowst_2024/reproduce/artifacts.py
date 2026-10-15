"""Method-local extension for native Mowst's 50 stochastic predictions/node.

The common exporter accepts one prediction/node. We keep its identity,
checkpoint, snapshot and five-seed conventions while storing an explicit
draw axis; unique nodes and stochastic observations are never conflated.
"""
from pathlib import Path
import json

import numpy as np
import torch
from shared.coe_benchmark import verify_identity, write_json, load_best, five_seed_statistics, write_summary
from shared.coe_benchmark.protocol import (PROTOCOL_ID, SEEDS, dataset_record, sha256_file, json_digest)
from shared.coe_benchmark.fingerprints import label_identity, split_fingerprint
from table1.Mowst_2024.reproduce.native import DATASETS

SCHEMA = "mowst_bernoulli_mc_v1"


def export_result(paths, data, output, timing):
    verify_identity(data)
    path = paths.exp
    if (path / "result.json").exists() or (path / "predictions.npz").exists():
        raise FileExistsError("Cannot overwrite a completed Mowst result")
    manifest = json.loads((path / "manifest.json").read_text())
    selected = load_best(path / "best.pt")
    if selected["payload"]["config_sha256"] != manifest["config_sha256"]:
        raise ValueError("Selected model configuration mismatch")
    ids = data.indices("test").cpu().numpy()
    labels = data.labels.cpu().numpy()
    artifact = {"node_ids": ids, "y_true": labels[ids], "y_pred": output["predictions"],
        "route_to_weak": output["routes"], "expert_probabilities": output["expert_probabilities"],
        "gate_probabilities": output["gate_probabilities"], "labels": labels,
        "class_mapping_json": np.asarray(json.dumps(list(data.class_mapping))),
        "prediction_axes": np.asarray("monte_carlo_draw,unique_test_node"),
        "canonical_source_fingerprint": np.asarray(data.fingerprints["source"])}
    artifact.update({name + "_mask": mask.cpu().numpy() for name, mask in data.masks.items()})
    np.savez_compressed(path / "predictions.npz", **artifact)
    test = {key: output[key] for key in ("acc", "correct", "total", "unique_test_nodes", "mc_samples", "per_draw_correct")}
    result = {"schema_version": SCHEMA, "status": "complete_single_seed", "method": "Mowst",
        "variant": "Mowst-GCN", "protocol_id": PROTOCOL_ID, "dataset": data.dataset,
        "seed": paths.seed, "phase": paths.phase, "test": test,
        "acc_definition": "mean of 50 native hard-routing draw ACCs = sum(correct)/(50*N_test)",
        "best_epoch": selected["epoch"], "best_validation_acc": selected["validation_acc"],
        "selection_rule": selected["selection_rule"], "checkpoint": "best.pt",
        "checkpoint_sha256": sha256_file(path / "best.pt"),
        "selection_record_sha256": sha256_file(path / "best.pt.selection.json"),
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "predictions_sha256": sha256_file(path / "predictions.npz"),
        "timing": timing, "model_replay_verified": False}
    write_json(path / "result.json", result)
    recount(path)
    return result


def recount(job_dir):
    path = Path(job_dir)
    result = json.loads((path / "result.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    if result["schema_version"] != SCHEMA or result["status"] != "complete_single_seed":
        raise ValueError("Wrong Mowst MC result schema/status")
    for key in ("method", "dataset", "seed", "phase", "protocol_id"):
        if result[key] != manifest[key]:
            raise ValueError("Manifest/result identity mismatch: " + key)
    if result["method"] != "Mowst" or result["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Wrong method/protocol")
    for file, field in (("best.pt", "checkpoint_sha256"), ("best.pt.selection.json", "selection_record_sha256"),
                        ("manifest.json", "manifest_sha256"), ("predictions.npz", "predictions_sha256")):
        if sha256_file(path / file) != result[field]:
            raise ValueError("Artifact hash mismatch: " + file)
    selection = json.loads((path / "best.pt.selection.json").read_text())
    if (selection["checkpoint_sha256"] != result["checkpoint_sha256"] or
        selection["best_epoch"] != result["best_epoch"] or
        selection["best_validation_acc"] != result["best_validation_acc"] or
        selection["selection_rule"] != result["selection_rule"]):
        raise ValueError("Checkpoint selection mismatch")
    if manifest["config_sha256"] != json_digest(manifest["config"]):
        raise ValueError("Configuration digest mismatch")
    sources = manifest["sources"]
    if sources["sha256"] != json_digest(sources["files"]):
        raise ValueError("Source manifest digest mismatch")
    for name, expected in sources["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid snapshot path")
        if sha256_file(path / "source_snapshot" / relative) != expected:
            raise ValueError("Source snapshot mismatch: " + name)
    record = dataset_record(result["dataset"])
    if manifest["data"]["fingerprints"]["source"] != record["canonical_source_fingerprint"]:
        raise ValueError("Manifest source identity mismatch")
    with np.load(path / "predictions.npz", allow_pickle=False) as saved:
        labels = saved["labels"]
        mapping = json.loads(saved["class_mapping_json"].item())
        if labels.shape != (record["nodes"],) or labels.dtype != np.int64:
            raise ValueError("Label vector identity/shape mismatch")
        if label_identity(torch.from_numpy(labels), mapping) != record["diagnostic_identity"]:
            raise ValueError("Label/class identity mismatch")
        masks = {name: torch.from_numpy(saved[name + "_mask"].copy()) for name in ("train", "val", "test")}
        if split_fingerprint(masks) != record["split_fingerprint"]:
            raise ValueError("Frozen split mismatch")
        if saved["canonical_source_fingerprint"].item() != record["canonical_source_fingerprint"]:
            raise ValueError("Saved source fingerprint mismatch")
        ids, predicted, routes = saved["node_ids"], saved["y_pred"], saved["route_to_weak"]
        draws = manifest["config"]["m_times"]
        expected_ids = np.flatnonzero(saved["test_mask"])
        if ids.dtype != np.int64 or not np.array_equal(ids, expected_ids):
            raise ValueError("Missing, reordered or duplicated test nodes")
        if predicted.dtype != np.int64 or predicted.shape != (draws, len(ids)):
            raise ValueError("Expected [MC_draw, unique_test_node] predictions")
        if predicted.min() < 0 or predicted.max() >= record["classes"]:
            raise ValueError("Invalid predicted class")
        if routes.dtype != np.bool_ or routes.shape != predicted.shape:
            raise ValueError("Native hard-route decisions missing")
        if not np.array_equal(saved["y_true"], labels[ids]):
            raise ValueError("Test label identities differ")
        experts, confidence = saved["expert_probabilities"], saved["gate_probabilities"]
        if experts.shape != (2, len(ids), record["classes"]) or confidence.shape != (len(ids),):
            raise ValueError("Expert/gate shapes differ")
        if (not np.isfinite(experts).all() or not np.isfinite(confidence).all() or
            (experts < 0).any() or not np.allclose(experts.sum(-1), 1, atol=1e-5) or
            (confidence < 0).any() or (confidence > 1).any()):
            raise ValueError("Invalid expert probabilities/confidence")
        expert_predictions = experts.argmax(-1)
        if not np.array_equal(predicted, np.where(routes, expert_predictions[0], expert_predictions[1])):
            raise ValueError("Saved classes are not the hard-routed expert predictions")
        counts = (predicted == labels[ids][None, :]).sum(axis=1).tolist()
        correct, total = sum(counts), draws * len(ids)
        test = result["test"]
        if (test["correct"], test["total"], test["unique_test_nodes"], test["mc_samples"], test["per_draw_correct"]) != (correct, total, len(ids), draws, counts):
            raise ValueError("MC sample/node counts differ from predictions")
        if abs(test["acc"] - correct / total) > 1e-12:
            raise ValueError("Recorded ACC differs from saved MC predictions")
    return {"manifest": manifest, "result": result, "acc": correct / total}


def verify_replay(job_dir, fresh):
    path = Path(job_dir)
    checked = recount(path)
    with np.load(path / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(fresh["predictions"], saved["y_pred"]):
            raise ValueError("Fresh Mowst MC class predictions differ")
        if not np.array_equal(fresh["routes"], saved["route_to_weak"]):
            raise ValueError("Fresh Mowst Bernoulli routes differ")
        score_difference = float(np.max(np.abs(fresh["expert_probabilities"] - saved["expert_probabilities"])))
        gate_difference = float(np.max(np.abs(fresh["gate_probabilities"] - saved["gate_probabilities"])))
    result = checked["result"]
    result["model_replay_verified"] = True
    write_json(path / "result.json", result)
    replay = {"status": "verified", "kind": "fresh_model_inference_with_checkpoint_rng",
        "exact_all_MC_class_predictions": True, "exact_all_MC_routes": True,
        "mc_samples": fresh["mc_samples"], "maximum_expert_probability_difference": score_difference,
        "maximum_gate_probability_difference": gate_difference,
        **{key: result[key] for key in ("checkpoint_sha256", "predictions_sha256", "manifest_sha256")}}
    write_json(path / "replay.json", replay)
    return replay


def summarize(run_root):
    root = Path(run_root)
    plan = json.loads((root / "planned_jobs.json").read_text())
    if plan["method"] != "Mowst" or plan["datasets"] != list(DATASETS) or plan["seeds"] != list(SEEDS):
        raise ValueError("Summary requires the preregistered five-dataset/five-seed Mowst plan")
    rows = []
    for dataset in DATASETS:
        values, signatures, unverified = {}, set(), []
        for seed in SEEDS:
            path = root / dataset / f"seed{seed}"
            if not (path / "result.json").exists():
                continue
            checked = recount(path)
            manifest, result = checked["manifest"], checked["result"]
            if (result["phase"] != "formal" or result["dataset"] != dataset or result["seed"] != seed or
                manifest["run_id"] != root.name or not manifest["paper_eligible"]):
                raise ValueError("Formal directory/identity mismatch")
            signatures.add(json_digest({key: manifest[key] for key in ("protocol_id", "data_protocol_id", "method",
                "dataset", "data", "config_sha256", "sources", "adaptation", "parameters", "selection_rule", "environment")}))
            values[seed] = checked["acc"]
            replay_path = path / "replay.json"
            ready = result["model_replay_verified"] and replay_path.exists()
            if ready:
                replay = json.loads(replay_path.read_text())
                ready = replay["status"] == "verified" and all(replay[key] == result[key]
                    for key in ("checkpoint_sha256", "predictions_sha256", "manifest_sha256"))
            if not ready:
                unverified.append(seed)
        if len(signatures) > 1:
            raise ValueError(f"{dataset}: mixed data/config/source/environment/parameter signatures")
        row = {"dataset": dataset, "observed_seeds": sorted(values), **five_seed_statistics(values),
               "unverified_replay_seeds": unverified}
        if unverified:
            row.update(status="incomplete", mean_percent=None, sample_std_percent=None)
        rows.append(row)
    summary = {"method": "Mowst", "variant": "Mowst-GCN", "protocol_id": PROTOCOL_ID,
        "status": "complete" if all(row["status"] == "complete" for row in rows) else "incomplete",
        "expected_datasets": list(DATASETS), "rows": rows,
        "scope": "Native 50-draw Bernoulli-route mean ACC per seed; std ddof=1 across five training seeds; fixed outer split."}
    write_summary(root, summary)
    return summary
