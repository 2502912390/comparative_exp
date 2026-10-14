"""Rebuild HDMI and its selected probe; fresh inference without any fitting."""
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
from shared.coe_benchmark import load_dataset, load_best, recount_result, verify_model_replay
from shared.coe_benchmark.protocol import json_digest, sha256_file, REPO_ROOT
from table1.HDMI_2021.reproduce.model import make_model, LogReg
from table1.HDMI_2021.reproduce.run import check_device


def replay(job_dir, device):
    checked = recount_result(job_dir)
    manifest = checked["manifest"]
    for name, expected in manifest["sources"]["files"].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise ValueError(f"Current replay source differs from checkpoint snapshot: {name}")
    config = manifest["config"]
    torch.set_num_threads(config["cpu_threads"])
    check_device(device)
    data = load_dataset(manifest["dataset"])
    if data.identity() != manifest["data"]:
        raise ValueError("Replay canonical data differs from training")
    saved = load_best(Path(job_dir) / checked["result"]["checkpoint"])
    if saved["payload"]["config_sha256"] != json_digest(config):
        raise ValueError("Replay configuration mismatch")
    model = make_model(data.input_dims[0], len(data.provided_adjs), config).to(device)
    head = LogReg(config["width"], data.classes).to(device)
    model.load_state_dict(saved["payload"]["model_state"], strict=True)
    head.load_state_dict(saved["payload"]["head_state"], strict=True)
    model.eval()
    head.eval()
    with torch.no_grad():
        ids = data.indices("test")
        embedding = model.embed(data.features[0].to(device),
            tuple(graph.to(device) for graph in data.provided_adjs), True)
        scores = head(embedding[ids.to(device)])
    result = verify_model_replay(job_dir, node_ids=ids, predictions=scores.argmax(-1), scores=scores)
    print("HDMI_REPLAY_VERIFIED", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    replay(args.job_dir, args.device)


if __name__ == "__main__":
    main()
