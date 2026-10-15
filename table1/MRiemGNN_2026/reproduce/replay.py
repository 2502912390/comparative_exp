"""Fresh FP32 inference; exact node predictions/ACC, score noise is diagnostic."""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]));__package__='reproduce'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from shared.coe_benchmark import load_dataset,load_best,seed_everything,recount_result,verify_model_replay
from shared.coe_benchmark.io import write_json
from .run import inference
def main():
    p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--device',default='cpu');o=p.parse_args()
    torch.set_num_threads(2);checked=recount_result(o.job_dir);m=checked['manifest'];seed_everything(m['seed'])
    data=load_dataset(m['dataset']);best=load_best(o.job_dir/'best.pt')
    if best['payload']['config_sha256']!=m['config_sha256']:raise ValueError('Checkpoint config mismatch')
    scores=inference(data,best['payload'],o.device).detach().cpu().numpy()
    if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite fresh replay scores')
    with np.load(o.job_dir/'predictions.npz',allow_pickle=False) as saved:
        prior=saved['scores']
        if scores.shape!=prior.shape:raise ValueError('Replay score shape mismatch')
        diag={'max_abs_logit_difference':float(abs(scores-prior).max()),'score_difference_is_gate':False,
              'dtype':str(scores.dtype),'rule':'exact fresh node predictions and Acc; ordinary FP32 score noise recorded only'}
    record=verify_model_replay(o.job_dir,node_ids=data.node_ids,predictions=scores.argmax(1))
    write_json(o.job_dir/'replay_numerics.json',diag)
    print(json.dumps({'replay':record,'numerics':diag}),flush=True)
if __name__=='__main__':main()
