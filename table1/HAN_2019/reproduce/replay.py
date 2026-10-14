"""Reconstruct and infer from best checkpoint; does not retrain."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from shared.coe_benchmark import load_best, verify_model_replay, seed_everything
from reproduce.common import METHOD_DIR,reconstruct


def main():
    p=argparse.ArgumentParser();p.add_argument("--job-dir",required=True,type=Path)
    p.add_argument("--device",default="cuda:0");p.add_argument("--threads",type=int,default=2)
    args=p.parse_args();job=args.job_dir.resolve()
    if not job.is_relative_to(METHOD_DIR/"exp"):
        raise ValueError("Replay must identify this method's exp job")
    torch.set_num_threads(args.threads)
    manifest=json.loads((job/"manifest.json").read_text());seed_everything(manifest["seed"])
    data,features,graphs,model=reconstruct(manifest,args.device)
    best=load_best(job/"best.pt")
    if best["payload"]["config_sha256"] != manifest["config_sha256"]:
        raise ValueError("Checkpoint configuration mismatch")
    model.load_state_dict(best["payload"]["model_state"],strict=True);model.eval()
    with torch.no_grad(): logits=model(features,graphs).cpu()
    replay=verify_model_replay(job,node_ids=data.node_ids,predictions=logits.argmax(-1),scores=logits)
    print(json.dumps({"event":"checkpoint_replay",**replay}),flush=True)


if __name__=="__main__":main()
