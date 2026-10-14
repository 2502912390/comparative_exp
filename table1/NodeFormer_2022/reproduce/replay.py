import argparse
import json
from pathlib import Path
import torch
from shared.coe_benchmark import load_best, restore_rng, verify_model_replay, recount_result
from shared.coe_benchmark.protocol import sha256_file, REPO_ROOT
from .common import setup, inputs
from .model import build_model, freeze_projections


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    setup()
    checked=recount_result(args.job_dir)
    manifest=checked['manifest']
    for name,digest in manifest['sources']['files'].items():
        if sha256_file(REPO_ROOT/name)!=digest:
            raise ValueError(f'Live source changed: {name}; run recorded snapshot instead')
    cfg=manifest['config']
    data,x,adjs=inputs(manifest['dataset'],args.device)
    if data.identity()!=manifest['data']:
        raise ValueError('Replay data identity mismatch')
    model=build_model(x.shape[1],data.classes,cfg).to(args.device)
    saved=load_best(args.job_dir/'best.pt')
    if saved['payload']['config_sha256']!=manifest['config_sha256']:
        raise ValueError('Checkpoint config differs')
    model.load_state_dict(saved['payload']['model_state'],strict=True)
    freeze_projections(model)
    restore_rng(saved['payload']['inference_rng'])
    model.eval()
    with torch.no_grad():
        scores,_=model(x,adjs,cfg['tau'])
    result=verify_model_replay(args.job_dir,node_ids=data.node_ids,predictions=scores.argmax(-1),scores=scores)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
