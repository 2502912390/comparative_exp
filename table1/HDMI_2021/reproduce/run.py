"""One unsupervised HDMI backbone and one validation-selected linear probe."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # Repository namespace only; never author code/.
os.environ.setdefault("COE_DATA_ROOT", "/home/chenzhichao/Research/CoE2/data/CoE_dataset")
os.environ.setdefault("COE_MULTIVIEW_DATA_ROOT", "/home/chenzhichao/Research/CoE2/data/Multi-view datasets")

import numpy as np
import torch
from shared.coe_benchmark import (RunPaths, load_dataset, seed_everything,
    BestCheckpoint, load_best, build_manifest, export_result, write_json)
from shared.coe_benchmark.protocol import json_digest, sha256_file
from table1.HDMI_2021.reproduce.model import (METHOD_DIR, make_model, LogReg,
    mutual_information_loss)

DATASETS = ("acm", "dblp", "yelp", "amazon", "mag")


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


def validate_config(config, phase):
    expected = json.loads((METHOD_DIR / "configs/search_space.json").read_text())["candidates"][0]["config"]
    if set(config) != set(expected):
        raise ValueError("Configuration keys differ from the registered HDMI schema")
    if any(config[key] < 1 for key in ("width", "pretrain_epochs", "probe_epochs", "discriminator_chunk_size", "cpu_threads")):
        raise ValueError("Invalid dimension/budget/chunk size")
    if config["pretrain_label_access"] or config["pretrain_selection"] != "fixed_final_unsupervised_epoch":
        raise ValueError("Unsupervised backbone must not use labels")
    if config["probe_selection"] != expected["probe_selection"] or config["probe_seed_rule"] != expected["probe_seed_rule"]:
        raise ValueError("Selection and probe seed rules are frozen")
    if len(config["coef_layers"]) != 3 or len(config["coef_fusion"]) != 3:
        raise ValueError("All six high-order mutual-information terms are required")
    if any(weight <= 0 for weight in config["coef_layers"] + config["coef_fusion"]):
        raise ValueError("Do not remove HDMI mutual-information terms")
    if phase != "smoke" and (config["pretrain_epochs"] < 400 or config["probe_epochs"] < 400):
        raise ValueError("Short runs are smoke fixtures, not development/formal experiments")


def core_sources():
    files = list((METHOD_DIR / "reproduce").glob("*.py"))
    files += list((METHOD_DIR / "configs").rglob("*.json"))
    files += [METHOD_DIR / name for name in json.loads((METHOD_DIR / "configs/author_sources.json").read_text())]
    files += [METHOD_DIR / "sources.json", METHOD_DIR / "ADAPTATION.md",
              METHOD_DIR / "REPRODUCE.md", METHOD_DIR / "environment/env_spec.json"]
    return files


def check_device(device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; no CPU fallback")
    torch.empty(1, device=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def execute(args, paths, config):
    validate_config(config, args.phase)
    torch.set_num_threads(config["cpu_threads"])
    check_device(args.device)
    seed_everything(args.seed)
    started = time.monotonic()
    print(f"HDMI_START dataset={args.dataset} seed={args.seed} phase={args.phase} device={args.device}", flush=True)
    data = load_dataset(args.dataset)
    if any(not torch.equal(view, data.features[0]) for view in data.features[1:]):
        raise ValueError("HDMI Table1 requires the common node attributes for every relation")
    model = make_model(data.input_dims[0], len(data.provided_adjs), config).to(args.device)
    head = LogReg(config["width"], data.classes).to(args.device)
    parameters = sum(parameter.numel() for parameter in model.parameters()) + sum(parameter.numel() for parameter in head.parameters())
    origin = json.loads((METHOD_DIR / "sources.json").read_text())
    adaptation = {"id": "hdmi_canonical_sparse_relations_v1", "input_dims": data.input_dims,
                  "provided_graphs_sparse_sha256": data.fingerprints["provided_graphs_sparse"],
                  "normalization": "use_canonical_provided_graphs_once_no_extra_feature_transform",
                  "relations": len(data.provided_adjs), "nodes": data.nodes,
                  "discriminator": "native_six_terms_equivalent_GEMM_node_chunks_global_readout",
                  "pretrain_selection": config["pretrain_selection"],
                  "probe_runs_per_seed": 1, "executor_model": "current session GPT-6",
                  "independent_reviewer": "not_run"}
    manifest = build_manifest(data, paths, method="HDMI", config=config,
        source_files=core_sources(), adaptation=adaptation, parameter_count=parameters,
        device=args.device, author_commit=origin["code_commit"])
    write_json(paths.exp / "resolved_config.json", config)
    write_json(paths.exp / "resource.json", {"device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else "CPU",
        "cpu_threads": config["cpu_threads"]})
    print(f"DATA_READY nodes={data.nodes} views={len(data.features)} dims={data.input_dims} parameters={parameters}", flush=True)
    features = data.features[0].to(args.device)
    graphs = tuple(graph.to(args.device) for graph in data.provided_adjs)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    pretrain_start = time.monotonic()
    model.train()
    with (paths.exp / "training.jsonl").open("x", buffering=1) as trace:
        for epoch in range(1, config["pretrain_epochs"] + 1):
            epoch_start = time.monotonic()
            permutation = torch.from_numpy(np.random.permutation(data.nodes)).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = mutual_information_loss(model(features, features[permutation], graphs, True), config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite HDMI loss at epoch {epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite HDMI gradient at epoch {epoch}")
            optimizer.step()
            row = {"stage": "unsupervised", "epoch": epoch, "loss": loss.item(),
                   "mi_terms": [term.item() for term in terms], "label_access": False,
                   "elapsed_seconds": time.monotonic() - epoch_start}
            trace.write(json.dumps(row, allow_nan=False) + "\n")
            if epoch == 1 or epoch % 20 == 0 or epoch == config["pretrain_epochs"]:
                print(f"UNSUP epoch={epoch}/{config['pretrain_epochs']} loss={loss.item():.7f} seconds={row['elapsed_seconds']:.3f}", flush=True)
        pretrain_seconds = time.monotonic() - pretrain_start
        torch.save({"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "config_sha256": manifest["config_sha256"], "epoch": config["pretrain_epochs"],
                    "selection": "fixed_final_unsupervised_epoch", "labels_used": False}, paths.exp / "encoder_final.pt")
        model.eval()
        with torch.no_grad():
            embedding = model.embed(features, graphs, True)
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("Nonfinite final unsupervised embeddings")
        seed_everything(args.seed + 100000)
        head = LogReg(config["width"], data.classes).to(args.device)
        train_ids, val_ids = (data.indices(split).to(args.device) for split in ("train", "val"))
        # Label tensors delivered to the only supervised training loop are
        # strictly the outer training labels. Validation labels are read only
        # in no_grad selection. No test nodes/labels enter either loop.
        train_embedding, val_embedding = embedding[train_ids], embedding[val_ids]
        train_labels = data.labels[train_ids.cpu()].to(args.device)
        val_labels = data.labels[val_ids.cpu()].to(args.device)
        probe_optimizer = torch.optim.Adam(head.parameters(), lr=config["probe_learning_rate"],
                                           weight_decay=config["probe_weight_decay"])
        selector = BestCheckpoint(paths.exp / "best.pt", patience_checks=config["patience_checks"])
        probe_start = time.monotonic()
        for epoch in range(1, config["probe_epochs"] + 1):
            head.train()
            probe_optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(train_embedding), train_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite probe loss at epoch {epoch}")
            loss.backward()
            probe_optimizer.step()
            head.eval()
            with torch.no_grad():
                val_acc = (head(val_embedding).argmax(-1) == val_labels).float().mean().item()
            selector.consider(epoch, val_acc, {"model_state": model.state_dict(),
                "head_state": head.state_dict(), "optimizer_state": probe_optimizer.state_dict(),
                "config_sha256": manifest["config_sha256"],
                "pretrain_epoch": config["pretrain_epochs"], "probe_seed": args.seed + 100000})
            trace.write(json.dumps({"stage": "linear_probe", "epoch": epoch, "train_ce": loss.item(),
                                   "validation_acc": val_acc, "best_epoch": selector.best_epoch}) + "\n")
            if epoch == 1 or epoch % 20 == 0 or selector.should_stop:
                print(f"PROBE epoch={epoch} val_acc={val_acc:.6f} best={selector.best_acc:.6f} best_epoch={selector.best_epoch}", flush=True)
            if selector.should_stop:
                break
        probe_seconds = time.monotonic() - probe_start
    timing = {"unsupervised_seconds": pretrain_seconds, "linear_probe_seconds": probe_seconds,
              "unsupervised_updates": config["pretrain_epochs"], "linear_probe_updates": selector.last_epoch,
              "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else 0}
    selected = load_best(paths.exp / "best.pt")
    if selected["payload"]["config_sha256"] != json_digest(config):
        raise ValueError("Selected inference state configuration mismatch")
    model.load_state_dict(selected["payload"]["model_state"], strict=True)
    head.load_state_dict(selected["payload"]["head_state"], strict=True)
    write_json(paths.exp / "validation_result.json", {"status": "complete_validation_only",
        "dataset": data.dataset, "seed": args.seed, "phase": args.phase,
        "best_validation_acc": selector.best_acc, "best_epoch": selector.best_epoch,
        "config_sha256": json_digest(config), "checkpoint_sha256": sha256_file(paths.exp / "best.pt"),
        "timing": timing, "test_evaluated": False})
    if args.phase == "dev":
        print(f"HDMI_DEV_COMPLETE val_acc={selector.best_acc:.6f}; no test evaluation", flush=True)
        return
    model.eval()
    head.eval()
    with torch.no_grad():
        ids = data.indices("test")
        scores = head(model.embed(features, graphs, True)[ids.to(args.device)])
    result = export_result(paths, data, checkpoint=paths.exp / "best.pt",
        best_epoch=selector.best_epoch, best_validation_acc=selector.best_acc,
        node_ids=ids, predictions=scores.argmax(-1), scores=scores, score_kind="logits", timing=timing)
    print(f"HDMI_TEST_COMPLETE acc={result['test']['acc']:.8f} correct={result['test']['correct']}/{result['test']['total']}; replay pending", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--phase", choices=("smoke", "dev", "formal"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=METHOD_DIR / "exp")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.resolve() != (METHOD_DIR / "exp").resolve():
        parser.error("output-root must be the HDMI method-local exp directory")
    config = json.loads(args.config.read_text())
    paths = RunPaths.create(METHOD_DIR, args.run_id, args.dataset, args.seed, args.phase)
    with (paths.log / "console.log").open("x", buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            try:
                execute(args, paths, config)
            except BaseException as error:
                import traceback
                traceback.print_exc()
                write_json(paths.exp / "failure.json", {"status": "failed", "error": repr(error),
                    "dataset": args.dataset, "seed": args.seed, "phase": args.phase})
                raise


if __name__ == "__main__":
    main()
