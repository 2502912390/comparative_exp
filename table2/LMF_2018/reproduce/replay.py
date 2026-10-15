"""Fresh FP32 inference from best checkpoint; no optimization or History update."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from shared.coe_benchmark import load_best,recount_result,verify_model_replay,write_json
from shared.coe_benchmark.artifacts import source_manifest
from .common import METHOD_DIR,source_files,prepare,infer
from .model import make_model,restore_aux,auxiliary_state
def replay(path,device):
    path=Path(path);checked=recount_result(path);manifest=checked['manifest']
    if source_manifest(source_files())!=manifest['sources']:raise ValueError('Current source differs from checkpoint')
    selected=load_best(path/'best.pt');state=selected['payload'];cfg=state['config']
    if state['config_sha256']!=manifest['config_sha256']:raise ValueError('Checkpoint config mismatch')
    data,features,meta=prepare(manifest['dataset'],device)
    for key in meta:
        if state['adaptation'][key]!=meta[key]:raise ValueError(f'Inference input adaptation changed: {key}')
    model=make_model(state['input_dims'],state['classes'],cfg).to(device);model.load_state_dict(state['model_state'],strict=True)
    aux=restore_aux(state['auxiliary_state'],data.indices('train'))
    scores=infer(model,features,data.node_ids.to(device),state['inference_batch_size']).cpu()
    # Eval does not receive histories or labels and cannot update them.
    with np.load(path/'predictions.npz',allow_pickle=False) as saved:
        old=saved['scores'];fresh=scores.numpy()
        if old.shape!=fresh.shape or not np.isfinite(fresh).all():raise ValueError('Invalid fresh scores')
        delta=float(np.max(np.abs(old.astype(np.float64)-fresh.astype(np.float64))))
    verify_model_replay(path,node_ids=data.node_ids,predictions=scores.argmax(1))
    record=dict(status='verified',max_abs_logit_difference=delta,dtype=str(scores.dtype),score_difference_is_gate=False,
                rule='exact fresh node predictions and ACC; finite FP32 logits drift diagnostic only')
    write_json(path/'replay_numerics.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-dir',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();replay(a.job_dir,a.device)
