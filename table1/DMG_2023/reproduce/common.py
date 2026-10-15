from pathlib import Path
from types import SimpleNamespace
import json
import torch
from shared.coe_benchmark.fingerprints import graph_fingerprint

METHOD_DIR=Path(__file__).resolve().parents[1]
DATASETS=('acm','dblp','yelp','amazon','mag')

def resolve_config(dataset):
    if dataset not in DATASETS: raise ValueError(dataset)
    document=json.loads((METHOD_DIR/'configs/author_defaults.json').read_text())
    args={**document['generic_defaults'],**document['dataset_overrides'].get(dataset,{})}
    return {**args,'policy':document['policy'],'parameter_source':document['parameter_sources'][dataset],'fixed_adaptations':document['fixed_adaptations']}

def args_for(config,data):
    return SimpleNamespace(**{**config,'ft_size':data.features[0].shape[1],'num_view':len(data.features)})

def source_files():
    return sorted([* (METHOD_DIR/'reproduce').glob('*.py'),* (METHOD_DIR/'configs').glob('*.json'),METHOD_DIR/'ADAPTATION.md',METHOD_DIR/'sources.json',* (METHOD_DIR/'environment').glob('*.json'),* (METHOD_DIR/'environment').glob('*.md'),* (METHOD_DIR/'code/models').glob('*.py'),METHOD_DIR/'code/embedder.py',METHOD_DIR/'code/utils/process.py',METHOD_DIR/'code/main.py',METHOD_DIR/'code/evaluate.py',METHOD_DIR/'code/args.yaml'])

class CyclicNeighbors:
    """Exact author 1-based cyclic neighbor cache represented as CSR offsets."""
    def __init__(self,indices,n):
        degree=torch.bincount(indices[0],minlength=n)
        if not bool((degree>0).all()): raise ValueError('Native neighbor cycling requires positive degree')
        self.degree=degree
        self.offset=torch.cat([degree.new_zeros(1),degree.cumsum(0)])[:-1]
        self.columns=indices[1]
    def at(self,zero_based_cache_index):
        return self.columns[self.offset+((zero_based_cache_index+1)%self.degree)]
    def to(self,device):
        self.degree=self.degree.to(device);self.offset=self.offset.to(device);self.columns=self.columns.to(device);return self


def prepare_graphs(data,sc,device):
    adjs=[];neighbors=[];hashes=[]
    for adj in data.provided_adjs:
        adj=adj.coalesce();ix=adj.indices();mask=(adj.values()>0)&(ix[0]!=ix[1]);ix=ix[:,mask]
        diag=torch.arange(data.nodes);ix=torch.cat([ix,torch.stack([diag,diag])],1)
        values=torch.cat([torch.ones(mask.sum()),torch.full((data.nodes,),float(sc))])
        raw=torch.sparse_coo_tensor(ix,values,(data.nodes,data.nodes)).coalesce()
        neighbor=CyclicNeighbors(raw.indices(),data.nodes)
        degree=torch.zeros(data.nodes).scatter_add_(0,raw.indices()[0],raw.values())
        inv=(degree.clamp(min=0)+2.2204e-16).pow(-.5)
        norm=raw.values()*inv[raw.indices()[0]]*inv[raw.indices()[1]]
        graph=torch.sparse_coo_tensor(raw.indices(),norm,raw.shape).coalesce()
        hashes.append(graph_fingerprint([graph]))
        adjs.append(graph.to(device));neighbors.append(neighbor.to(device))
    return adjs,neighbors,{'id':'dmg_native_multirelation_v1','rule':'canonical positive support + sc diagonal + native symmetric degree normalization','graphs_sha256':hashes,'views':len(adjs),'input_dims':[x.shape[1] for x in data.features],'public_features_unchanged':True}
