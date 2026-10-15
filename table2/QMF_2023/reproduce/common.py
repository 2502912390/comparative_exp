from pathlib import Path
import json
from shared.coe_benchmark import load_dataset,graph_to_feature_context_v1,adaptation_metadata
METHOD_DIR=Path(__file__).resolve().parents[1]
DATASETS=('esp','flickr','iapr','nus','mv_acm','mv_citeseer','mv_prokaryotic','mv_reuters1200','mv_wikipedia')
METHOD=json.loads((METHOD_DIR/'configs/author_defaults.json').read_text())['method']
def config_for(dataset):
    if dataset not in DATASETS:raise ValueError(dataset)
    return {**json.loads((METHOD_DIR/'configs/author_defaults.json').read_text()),'dataset':dataset}
def source_files():
    return sorted([*METHOD_DIR.joinpath('reproduce').rglob('*.py'),*METHOD_DIR.joinpath('configs').glob('*.json'),
        *METHOD_DIR.joinpath('environment').glob('*.json'),*METHOD_DIR.joinpath('environment').glob('*.md'),
        METHOD_DIR/'ADAPTATION.md',METHOD_DIR/'REPRODUCE.md',METHOD_DIR/'sources.json',*METHOD_DIR.joinpath('code').rglob('*.py'),
        *METHOD_DIR.joinpath('code').glob('README*')])
def prepare(dataset,device):
    data=load_dataset(dataset)
    kind='graph_to_feature_context_v1' if data.table==3 else 'canonical'
    meta=adaptation_metadata(data,kind)
    features=graph_to_feature_context_v1(data) if data.table==3 else data.features
    meta.update(encoder_adaptation='static_vectors',input_dims=[x.shape[1] for x in features],views=len(features))
    expected={'mv_acm':[1870,3740,3740,3740,3740],'mv_citeseer':[7406,3703],'mv_prokaryotic':[393,3,438]}
    if dataset in expected and meta['input_dims']!=expected[dataset]:raise ValueError('View dimensions changed')
    return data,tuple(x.to(device) for x in features),meta
def batches(ids,batch_size,shuffle=False,merge_singleton=False):
    order=ids[__import__('torch').randperm(len(ids),device=ids.device)] if shuffle else ids
    chunks=list(order.split(batch_size))
    if merge_singleton and len(chunks)>1 and len(chunks[-1])==1:
        chunks[-2]=__import__('torch').cat((chunks[-2],chunks[-1]));chunks.pop()
    if merge_singleton and any(len(c)<2 for c in chunks):raise ValueError('BatchNorm needs >=2 train samples')
    return chunks
def infer(model,features,ids,batch_size):
    import torch
    model.eval()
    with torch.no_grad():
        output=torch.cat([model([x[ix] for x in features]) for ix in batches(ids,batch_size)],dim=0)
    if output.shape[0]!=len(ids) or not torch.isfinite(output).all():raise ValueError('Invalid model inference')
    return output
