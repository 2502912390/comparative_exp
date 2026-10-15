"""Reconstruct a fresh native model from selected weights; never refit."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from shared.coe_benchmark import recount_result, verify_model_replay, write_json
from table2.MMDynamics_2022.reproduce.common import canonical_data, restore_model, split_features, infer


def replay(job_dir, device):
    job_dir = Path(job_dir)
    checked = recount_result(job_dir)
    data = canonical_data(checked["result"]["dataset"])
    model, _ = restore_model(job_dir, data, device)
    scores = infer(model, split_features(data, "test", device)).cpu().numpy()
    with np.load(job_dir / "predictions.npz", allow_pickle=False) as saved:
        difference = float(np.max(np.abs(saved["scores"] - scores)))
    verified = verify_model_replay(job_dir, node_ids=data.indices("test").numpy(),
                                  predictions=scores.argmax(-1))
    verified.update(kind="fresh_native_MMDynamic_infer_from_selected_state_dict",
                    maximum_logit_difference=difference, scores_finite=bool(np.isfinite(scores).all()),
                    precision="float32", refitting=False)
    write_json(job_dir / "replay.json", verified)
    print("MMDYNAMICS_REPLAY_VERIFIED", json.dumps(verified), flush=True)
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    replay(args.job_dir, args.device)
