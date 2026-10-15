if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "reproduce"

import argparse,json
from pathlib import Path
import torch
from shared.coe_benchmark import load_dataset,load_best,verify_model_replay,seed_everything
from shared.coe_benchmark.protocol import sha256_file,REPO_ROOT
from .run import reconstruct

def main():
    p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--device',default='cpu');o=p.parse_args();torch.set_num_threads(2)
    manifest=json.loads((o.job_dir/'manifest.json').read_text())
    for name,expected in manifest['sources']['files'].items():
        if sha256_file(REPO_ROOT/name)!=expected:raise ValueError(f'Current core changed since formal run: {name}')
    seed_everything(manifest['seed']);data=load_dataset(manifest['dataset']);best=load_best(o.job_dir/'best.pt')
    if best['payload']['config_sha256']!=manifest['config_sha256']:raise ValueError('Checkpoint config mismatch')
    scores=reconstruct(data,best['payload'],o.device)
    print(json.dumps(verify_model_replay(o.job_dir,node_ids=data.node_ids,predictions=scores.argmax(1),scores=scores)),flush=True)
if __name__=='__main__':main()
