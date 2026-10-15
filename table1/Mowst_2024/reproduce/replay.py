"""New inference from all three saved modules and the saved evaluation RNG."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("COE_DATA_ROOT", "/home/chenzhichao/Research/CoE2/data/CoE_dataset")
os.environ.setdefault("COE_MULTIVIEW_DATA_ROOT", "/home/chenzhichao/Research/CoE2/data/Multi-view datasets")

import torch
from shared.coe_benchmark import load_dataset, load_best
from shared.coe_benchmark.protocol import json_digest
from table1.Mowst_2024.reproduce.native import Mowst, graph_data
from table1.Mowst_2024.reproduce.inference import monte_carlo
from table1.Mowst_2024.reproduce.artifacts import recount, verify_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested replay CUDA unavailable")
    checked = recount(args.job_dir)
    manifest = checked["manifest"]
    config = manifest["config"]
    canonical = load_dataset(manifest["dataset"])
    if canonical.identity() != manifest["data"]:
        raise ValueError("Replay canonical data identity mismatch")
    graph = graph_data(canonical, args.device)
    model = Mowst(canonical.input_dims[0], canonical.classes, config).to(args.device)
    saved = load_best(args.job_dir / "best.pt")
    if saved["payload"]["config_sha256"] != json_digest(config):
        raise ValueError("Replay configuration mismatch")
    model.load_state_dict(saved["payload"]["model_state"], strict=True)
    model.eval()
    fresh = monte_carlo(model, graph, canonical.indices("test"),
                        saved["payload"]["inference_rng_state"], keep_predictions=True)
    result = verify_replay(args.job_dir, fresh)
    print("MOWST_REPLAY_VERIFIED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
