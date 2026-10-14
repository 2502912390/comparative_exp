"""CPU audit, matrix planning, artifact recount and complete-five-seed summaries."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.io import loadmat

from . import (DATASETS, PROTOCOL_ID, IMPLEMENTATION_VERSION, load_dataset,
               adaptation_metadata, check_reference_sources, read_registry,
               plan_method, save_plan, summarize_run, write_summary, recount_result)
from .io import write_json, validate_run_id
from .protocol import REPO_ROOT


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def write(self, text):
        self.original.write(text)
        self.stream.write(text)
        self.stream.flush()

    def flush(self):
        self.original.flush()
        self.stream.flush()


def _wikipedia_membership(data):
    """Independent multiset comparison using raw full/test observations."""
    full = loadmat(str(data.roots.multiview / "Wikipedia.mat"))
    test = loadmat(str(data.roots.multiview / "Wikipedia-test.mat"))
    def entries(mat):
        labels = np.asarray(mat["y"] if "y" in mat else mat["Y"]).ravel()
        views = list(np.asarray(mat["X"]).ravel())
        arrays = []
        for view in views:
            view = view.toarray() if hasattr(view, "toarray") else np.asarray(view)
            view = np.asarray(view, dtype=np.float32)
            if view.shape[0] != len(labels):
                view = view.T
            arrays.append(view)
        return [(float(labels[i]), *(np.ascontiguousarray(view[i]).tobytes() for view in arrays))
                for i in range(len(labels))]
    full_entries, test_entries = entries(full), entries(test)
    selected = Counter(full_entries[i] for i in data.indices("test").tolist())
    if selected != Counter(test_entries):
        raise ValueError("Wikipedia frozen test mask does not match the independent test observations")
    return {"status": "matched", "test_nodes": len(test_entries), "matching": "raw-view/label multiset"}


def audit(args):
    shared = REPO_ROOT / "shared"
    run_id = "audit_" + validate_run_id(args.run_id)
    exp, log = shared / "exp" / run_id, shared / "log" / run_id
    if exp.exists() or log.exists():
        raise FileExistsError("Audit run ID already exists; choose a new ID")
    exp.mkdir(parents=True)
    log.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    started = time.time()
    report = {"protocol_id": PROTOCOL_ID, "implementation_version": IMPLEMENTATION_VERSION,
              "status": "running", "executor": "current Codex session; no subagents",
              "scope": "CPU data/identity/adaptation and functional checks; no baseline training",
              "datasets": [], "reference_model_graphs_used": False}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (log / "console.log").open("w", encoding="utf-8", buffering=1) as stream:
        sys.stdout, sys.stderr = Tee(original_stdout, stream), Tee(original_stderr, stream)
        try:
            registry = read_registry()
            report["reference_source_checks"] = check_reference_sources(registry)
            print("Pinned reference sources/configuration MATCH", flush=True)
            for dataset in args.datasets:
                data = load_dataset(dataset)
                row = data.identity()
                row["status"] = "matched"
                row["sparse_graph_edges"] = [graph._nnz() for graph in data.provided_adjs]
                row["reference_model_graph_hash_scope"] = "recorded in registry; deliberately not used or compared"
                if data.table == 1:
                    row["relation_union"] = adaptation_metadata(data, "relation_union")
                if data.table == 3:
                    row["graph_to_feature_context_v1"] = adaptation_metadata(data, "graph_to_feature_context_v1")
                    expected = ({"mv_acm": [1870, 3740, 3740, 3740, 3740],
                                 "mv_citeseer": [7406, 3703]}.get(dataset, data.input_dims))
                    if row["graph_to_feature_context_v1"]["input_dims"] != expected:
                        raise ValueError("Table3 relation-view adaptation dimensions differ")
                if dataset == "mv_wikipedia":
                    row["independent_test_membership"] = _wikipedia_membership(data)
                report["datasets"].append(row)
                write_json(exp / "audit.json", report)
                print(dataset, "MATCH", data.nodes, "nodes;", data.input_dims, flush=True)
                del data
            if args.self_tests:
                tests = shared / "tests"
                output = exp / "unit_tests"
                output.mkdir()
                import os
                env = dict(os.environ)
                env["COE_BENCHMARK_TEST_OUTPUT"] = str(output)
                env["CUDA_VISIBLE_DEVICES"] = ""  # CPU fixtures must not create contexts on busy GPUs.
                completed = subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "-s", str(tests),
                     "-p", "test_*.py", "-v"], cwd=REPO_ROOT, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                (log / "unit_tests.log").write_text(completed.stdout)
                print(completed.stdout, flush=True)
                report["functional_tests"] = {"returncode": completed.returncode,
                                             "log": str(log / "unit_tests.log"),
                                             "scope": "Unit fixtures and a two-step CPU checkpoint/export/replay fixture; not a paper baseline."}
                if completed.returncode:
                    raise RuntimeError("Functional checks failed; inspect unit_tests.log")
            report["status"] = "passed"
        except Exception as exc:
            report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            report["elapsed_seconds"] = time.time() - started
            write_json(exp / "audit.json", report)
            sys.stdout, sys.stderr = original_stdout, original_stderr
    print(json.dumps({"status": report["status"], "checked_datasets": len(report["datasets"]),
                      "report": str(exp / "audit.json"), "log": str(log / "console.log")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--run-id", required=True)
    check.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    check.add_argument("--cpu-threads", type=int, default=2)
    check.add_argument("--self-tests", action="store_true")
    plan = commands.add_parser("plan")
    plan.add_argument("--method", required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--devices", nargs="+", default=["cpu"])
    plan.add_argument("--phase", choices=("smoke", "dev", "formal"), default="formal")
    plan.add_argument("--save", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--datasets", nargs="+", choices=DATASETS)
    recount = commands.add_parser("recount")
    recount.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        if args.cpu_threads < 1:
            parser.error("cpu-threads must be positive")
        audit(args)
    elif args.command == "plan":
        result = plan_method(args.method, args.run_id, devices=tuple(args.devices), phase=args.phase)
        if args.save:
            result["plan_path"] = str(save_plan(result))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "summarize":
        result = summarize_run(args.run_root, expected_datasets=args.datasets)
        write_summary(args.run_root, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "recount":
        result = recount_result(args.job_dir)
        print(json.dumps({key: result[key] for key in ("acc", "correct", "total")}, indent=2))


if __name__ == "__main__":
    main()

