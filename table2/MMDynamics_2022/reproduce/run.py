"""One fresh native MMDynamics fit; validation selection then a single frozen test."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from shared.coe_benchmark import (RunPaths, seed_everything, BestCheckpoint,
    build_manifest, export_result, write_json, adaptation_metadata)
from table2.MMDynamics_2022.reproduce.common import (
    METHOD_DIR, DATASETS, EXECUTOR_MODEL, read_config, canonical_data,
    make_model, split_features, infer, restore_model, source_files, check_author_sources)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def fit(paths, data, config, device):
    start = time.monotonic()
    seed_everything(paths.seed)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no silent CPU fallback")
    model = make_model(data, config, device)
    author = check_author_sources()
    adaptation = adaptation_metadata(data, "canonical")
    adaptation.update(view_order=["text", "image"], additional_normalization=False,
                      model_input="canonical_text_image_features")
    manifest = build_manifest(data, paths, method="MMDynamics", config=config,
        source_files=source_files(), adaptation=adaptation,
        parameter_count=sum(p.numel() for p in model.parameters()), device=device,
        author_commit=author["author_commit"])
    write_json(paths.exp / "environment.json", manifest["environment"])
    write_json(paths.exp / "execution.json", {"executor_model": EXECUTOR_MODEL,
        "independent_reviewer": "not_run", "pid": os.getpid(),
        "parameter_sources": "source_snapshot/table2/MMDynamics_2022/configs/PARAMETER_SOURCES.json"})
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"], betas=tuple(config["adam_betas"]),
        eps=config["adam_eps"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=config["step_size"], gamma=config["gamma"])
    # Full-batch native training: only outer-train features and labels reach forward.
    train_x = split_features(data, "train", device)
    train_y = data.labels[data.indices("train")].to(device)
    val_x = split_features(data, "val", device)
    val_y = data.labels[data.indices("val")].to(device)
    selector = BestCheckpoint(paths.exp / "best.pt", patience_checks=0)
    print("MMDYNAMICS_START", json.dumps({"dataset": data.dataset, "seed": paths.seed,
        "updates": config["training_updates"], "parameters": manifest["parameters"],
        "device": str(device), "parameter_policy": config["parameter_policy"]}), flush=True)
    with (paths.exp / "training.jsonl").open("w") as trace:
        for update in range(1, config["training_updates"] + 1):
            model.train()
            optimizer.zero_grad()
            used_lr = optimizer.param_groups[0]["lr"]
            loss, _ = model(train_x, train_y)
            loss = torch.mean(loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite native gradients")
            optimizer.step()
            scheduler.step()
            val_scores = infer(model, val_x)
            val_acc = int((val_scores.argmax(-1) == val_y).sum()) / len(val_y)
            improved = selector.consider(update, val_acc, {
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "config_sha256": manifest["config_sha256"],
                "input_dims": data.input_dims, "classes": data.classes,
                "author_epoch_index": update - 1})
            record = {"update": update, "author_epoch_index": update - 1,
                "loss": loss.item(), "validation_acc": val_acc,
                "best_validation_acc": selector.best_acc, "selected": improved,
                "learning_rate": used_lr, "elapsed_seconds": time.monotonic() - start}
            trace.write(json.dumps(record, allow_nan=False) + "\n")
            trace.flush()
            if update == 1 or update % 50 == 0 or update == config["training_updates"]:
                print("TRAIN", json.dumps(record), flush=True)
    training_seconds = time.monotonic() - start
    # Restore the selected complete state before accessing test features or labels.
    model, selected = restore_model(paths.exp, data, device)
    test_x = split_features(data, "test", device)
    scores = infer(model, test_x).cpu()
    result = export_result(paths, data, checkpoint=paths.exp / "best.pt",
        best_epoch=selected["epoch"], best_validation_acc=selected["validation_acc"],
        node_ids=data.indices("test"), predictions=scores.argmax(-1),
        scores=scores, score_kind="logits", timing={"training_seconds": training_seconds,
            "total_seconds": time.monotonic() - start, "updates": config["training_updates"],
            "stop_reason": "author_fixed_budget_no_early_stopping",
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0})
    print("MMDYNAMICS_COMPLETE", json.dumps(result["test"]), flush=True)
    return result


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("smoke", "dev", "formal"), default="formal")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=METHOD_DIR / "exp")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke-updates", type=int)
    args = parser.parse_args()
    if args.output_root.resolve() != (METHOD_DIR / "exp").resolve():
        parser.error("output-root must remain in this method's exp")
    config = read_config(args.config, phase=args.phase, smoke_updates=args.smoke_updates)
    data = canonical_data(args.dataset)
    paths = RunPaths.create(METHOD_DIR, args.run_id, args.dataset, args.seed, args.phase)
    with (paths.log / "console.log").open("w") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                fit(paths, data, config, args.device)
            except Exception:
                traceback.print_exc()
                write_json(paths.exp / "failure.json", {"status": "failed", "traceback": traceback.format_exc()})
                raise


if __name__ == "__main__":
    main()
