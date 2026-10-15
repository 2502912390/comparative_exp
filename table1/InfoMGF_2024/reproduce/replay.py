"""Fresh checkpoint inference without refitting graph learners or GCN."""
import argparse
import json
from pathlib import Path
import torch
from shared.coe_benchmark import load_dataset,load_best,verify_model_replay,recount_result
from .mechanism import InfoMGF,Classifier,unpack_graph


def replay(path,device):
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text())
    recount_result(path)
    selected=load_best(path/'best.pt');s=selected['payload'];cfg=s['config']
    data=load_dataset(manifest['dataset'])
    x=data.features[0] if data.table==1 else torch.cat(data.features,dim=1)
    x=x.to(device)
    # Restore the unsupervised chain as evidence, but the inference graph is
    # already frozen in checkpoint; deterministic replay never refits it.
    encoder=InfoMGF(s['input_dim'],s['view_dims'],cfg).to(device)
    encoder.load_state_dict(s['model_state'],strict=True);encoder.eval()
    head=Classifier(s['input_dim'],s['classes'],cfg).to(device)
    head.load_state_dict(s['head_state'],strict=True);head.eval()
    with torch.no_grad():scores=head(x,unpack_graph(s['graph_state'],device))
    verify_model_replay(path,node_ids=data.node_ids,predictions=scores.argmax(1).cpu(),scores=scores.cpu())
    print(json.dumps({'replay':'passed','job_dir':str(path)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-dir',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();replay(a.job_dir,a.device)
