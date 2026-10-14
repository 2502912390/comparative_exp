"""Complete fixed-five-seed summaries with strict provenance and replay gates."""
from pathlib import Path
import csv
import json
import statistics
from .artifacts import recount_result
from .protocol import SEEDS, DATASETS, PROTOCOL_ID, json_digest
from .io import write_json


def five_seed_statistics(seed_accuracy):
    if set(seed_accuracy) - set(SEEDS):
        raise ValueError("Unexpected seed in five-seed statistics")
    missing = sorted(set(SEEDS) - set(seed_accuracy))
    if missing:
        return {"status": "incomplete", "missing_seeds": missing,
                "mean_percent": None, "sample_std_percent": None}
    values = [float(seed_accuracy[seed]) * 100 for seed in SEEDS]
    if any(not 0 <= value <= 100 for value in values):
        raise ValueError("ACC must be a finite fraction in [0,1]")
    return {"status": "complete", "seeds": list(SEEDS),
            "per_seed_percent": dict(zip(map(str, SEEDS), values)),
            "mean_percent": statistics.mean(values),
            "sample_std_percent": statistics.stdev(values), "ddof": 1}


def summarize_run(run_root, *, expected_datasets=None):
    run_root = Path(run_root)
    plan_path = run_root / "planned_jobs.json"
    if expected_datasets is None:
        if not plan_path.exists():
            raise ValueError("Supply expected_datasets or a frozen planned_jobs.json; scope is never inferred from successes")
        expected_datasets = json.loads(plan_path.read_text())["datasets"]
    expected_datasets = tuple(expected_datasets)
    if (not expected_datasets or len(set(expected_datasets)) != len(expected_datasets)
            or set(expected_datasets) - set(DATASETS)):
        raise ValueError("Expected scope must contain distinct registered datasets")
    groups = {dataset: {} for dataset in expected_datasets}
    methods = set()
    for file in sorted(run_root.glob("*/seed*/result.json")):
        raw = json.loads(file.read_text())
        if raw["phase"] != "formal":
            continue
        if raw["dataset"] not in groups or raw["seed"] not in SEEDS:
            raise ValueError("Unexpected formal dataset/seed in run scope")
        if raw["seed"] in groups[raw["dataset"]]:
            raise ValueError("Duplicate formal seed; retries cannot be averaged")
        checked = recount_result(file.parent)
        manifest = checked["manifest"]
        if not manifest.get("paper_eligible") or manifest["run_id"] != run_root.name:
            raise ValueError("Non-formal run cannot enter a paper summary")
        if file.parent.name != f"seed{raw['seed']}" or file.parent.parent.name != raw["dataset"]:
            raise ValueError("Run directory/identity mismatch")
        methods.add(manifest["method"])
        replay_file = file.parent / "replay.json"
        replay_ready = False
        if checked["result"].get("model_replay_verified") and replay_file.exists():
            replay = json.loads(replay_file.read_text())
            replay_ready = replay.get("status") == "verified" and all(
                replay.get(key) == checked["result"][key]
                for key in ("checkpoint_sha256", "predictions_sha256", "manifest_sha256"))
        checked["replay_ready"] = replay_ready
        groups[raw["dataset"]][raw["seed"]] = checked
    if len(methods) > 1:
        raise ValueError("Different methods cannot share a single run summary")
    rows = []
    for dataset, records in groups.items():
        signatures = []
        for record in records.values():
            manifest = record["manifest"]
            signatures.append(json_digest({key: manifest[key] for key in
                ("protocol_id", "data_protocol_id", "method", "dataset", "data",
                 "config_sha256", "sources", "adaptation", "parameters",
                 "selection_rule", "environment")}))
        if len(set(signatures)) > 1:
            raise ValueError(f"{dataset}: data/config/source/adaptation/environment differ between seeds")
        values = {seed: record["acc"] for seed, record in records.items()}
        row = {"dataset": dataset, "observed_seeds": sorted(records),
               **five_seed_statistics(values)}
        unverified = sorted(seed for seed, record in records.items() if not record["replay_ready"])
        row["unverified_replay_seeds"] = unverified
        if unverified:
            row.update(status="incomplete", mean_percent=None, sample_std_percent=None)
        rows.append(row)
    return {"protocol_id": PROTOCOL_ID, "method": next(iter(methods)) if methods else None,
            "status": "complete" if all(row["status"] == "complete" for row in rows) else "incomplete",
            "expected_datasets": list(expected_datasets), "rows": rows,
            "scope": "Fixed outer split; five independent training seeds; sample std ddof=1."}


def write_summary(run_root, summary):
    """LaTeX rows contain only complete/replayed results, never partial estimates."""
    run_root = Path(run_root)
    write_json(run_root / "summary.json", summary)
    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "status", "mean_percent",
                                                    "sample_std_percent", "missing_seeds",
                                                    "unverified_replay_seeds"])
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    lines = ["% Fixed split, seeds 1024-1028, sample std ddof=1.",
             "% Incomplete or unverified datasets are intentionally omitted."]
    for row in summary["rows"]:
        if row["status"] == "complete":
            lines.append(f"{row['dataset']} & {row['mean_percent']:.2f}"
                         + r"\,$\pm$\," + f"{row['sample_std_percent']:.2f}" + r" \\")
    (run_root / "table_rows.tex").write_text("\n".join(lines) + "\n")

