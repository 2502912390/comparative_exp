"""One Mowst-GCN job: fixed author settings, canonical union graph, all state saved."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("COE_DATA_ROOT", "/home/chenzhichao/Research/CoE2/data/CoE_dataset")
os.environ.setdefault("COE_MULTIVIEW_DATA_ROOT", "/home/chenzhichao/Research/CoE2/data/Multi-view datasets")

import torch
from shared.coe_benchmark import (RunPaths, load_dataset, seed_everything, build_manifest,
                                  adaptation_metadata, write_json)
from shared.coe_benchmark.protocol import sha256_file
from table1.Mowst_2024.reproduce.native import METHOD_DIR, DATASETS, Mowst, graph_data
from table1.Mowst_2024.reproduce.inference import monte_carlo
from table1.Mowst_2024.reproduce.training import Trainer
from table1.Mowst_2024.reproduce.artifacts import export_result


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def core_sources():
    return (list((METHOD_DIR / "reproduce").glob("*.py")) + list((METHOD_DIR / "configs").glob("*.json"))
        + [METHOD_DIR / name for name in json.loads((METHOD_DIR / "configs/author_sources.json").read_text())]
        + [METHOD_DIR / "ADAPTATION.md", METHOD_DIR / "REPRODUCE.md", METHOD_DIR / "sources.json",
           METHOD_DIR / "environment/env_spec.json"])


def execute(args, paths, config):
    if args.phase == "formal" and config != json.loads((METHOD_DIR / "configs/default.json").read_text()):
        raise ValueError("Formal Mowst uses only the fixed author configuration")
    torch.set_num_threads(2)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no CPU fallback")
    torch.empty(1, device=args.device)
    print(f"MOWST_START dataset={args.dataset} seed={args.seed} device={args.device} phase={args.phase}", flush=True)
    print("FIXED_CONFIGURATION", json.dumps(config), flush=True)
    started = time.monotonic()
    canonical = load_dataset(args.dataset)
    graph = graph_data(canonical, args.device)
    model = Mowst(canonical.input_dims[0], canonical.classes, config).to(args.device)
    # Match author MowstEngine: construct, seed, then reset all three modules.
    seed_everything(args.seed)
    model.reset_parameters()
    adaptation = {**adaptation_metadata(canonical, "relation_union"),
        "variant": "Mowst-GCN", "parameter_policy": "author_default_no_hpo",
        "parameter_sources_sha256": sha256_file(METHOD_DIR / "configs/PARAMETER_SOURCES.json"),
        "evaluation": "native_hard_Bernoulli_routing_50_draw_mean_ACC",
        "pyg_interface": "remove_union_diagonal_then_native_COO_GCNConv_adds_unit_diagonal_once",
        "result_schema": "mowst_bernoulli_mc_v1", "checkpoint": "MLP_GCN_gate_and_BN_buffers_together"}
    author = json.loads((METHOD_DIR / "sources.json").read_text())
    parameters = sum(p.numel() for p in model.parameters())
    manifest = build_manifest(canonical, paths, method="Mowst", config=config,
        source_files=core_sources(), adaptation=adaptation, parameter_count=parameters,
        device=args.device, author_commit=author["code_commit"])
    write_json(paths.exp / "resolved_config.json", config)
    write_json(paths.exp / "parameter_sources.json", json.loads((METHOD_DIR / "configs/PARAMETER_SOURCES.json").read_text()))
    write_json(paths.exp / "resource.json", {"device": args.device, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU"})
    print(f"DATA_READY nodes={canonical.nodes} union_edges={graph.union_edges} input_offdiagonal_edges={graph.adj_t._nnz()} features={canonical.input_dims[0]} parameters={parameters}", flush=True)
    with (paths.exp / "training.jsonl").open("x", buffering=1) as trace:
        trainer = Trainer(model, graph, canonical, paths, manifest, trace)
        timing, best = trainer.train()
    output = monte_carlo(model, graph, canonical.indices("test"), best["payload"]["inference_rng_state"],
                         keep_predictions=True)
    timing.update(elapsed_seconds=time.monotonic() - started,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0)
    result = export_result(paths, canonical, output, timing)
    print(f"MOWST_COMPLETE ACC={100*result['test']['acc']:.6f}% samples={result['test']['mc_samples']} unique_test_nodes={result['test']['unique_test_nodes']} correct_observations={result['test']['correct']}/{result['test']['total']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--phase", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=METHOD_DIR / "exp")
    parser.add_argument("--config", type=Path, default=METHOD_DIR / "configs/default.json")
    args = parser.parse_args()
    if args.output_root.resolve() != (METHOD_DIR / "exp").resolve():
        parser.error("output-root must be the Mowst method's exp directory")
    paths = RunPaths.create(METHOD_DIR, args.run_id, args.dataset, args.seed, args.phase)
    with (paths.log / "console.log").open("x", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                execute(args, paths, json.loads(args.config.read_text()))
            except BaseException as error:
                import traceback
                traceback.print_exc()
                write_json(paths.exp / "failure.json", {"status": "failed", "error": repr(error)})
                raise


if __name__ == "__main__":
    main()
