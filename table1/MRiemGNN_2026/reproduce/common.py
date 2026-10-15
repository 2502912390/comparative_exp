from pathlib import Path
import json,hashlib
import torch
from .native import MRiemGNN
METHOD_DIR=Path(__file__).resolve().parents[1]
DATASETS=('acm','dblp','yelp','amazon','mag')
MODEL_KEYS=('hid_dim','num_layers','num_kernels','dropout','kernel_learnable','shared_space_kernel','learnable_beta',
 'hyperbolic_proj','input_skip','input_skip_weight','layer_norm','relation_dropout','node_relation_attention',
 'node_relation_attention_mix','classifier_hidden','feature_logit_weight','behavior_projector','node_embedding_dim')
def config_for(dataset):
    if dataset not in DATASETS:raise ValueError(dataset)
    return {**json.loads((METHOD_DIR/'configs/author_defaults.json').read_text())['effective'],'dataset':dataset}
def create_model(data,config,relations):
    return MRiemGNN(in_dim=data.features[0].shape[1],out_dim=data.classes,
        relation_types=list(range(relations)),spaces=config['spaces'],
        manifold_params=config['manifold_params'],beta_init=config['beta'],num_nodes=data.nodes,
        **{k:config[k] for k in MODEL_KEYS})
def source_files():
    return sorted([*METHOD_DIR.joinpath('reproduce').rglob('*.py'),*METHOD_DIR.joinpath('configs').glob('*.json'),
        *METHOD_DIR.joinpath('environment').glob('*.json'),*METHOD_DIR.joinpath('environment').glob('*.md'),
        METHOD_DIR/'ADAPTATION.md',METHOD_DIR/'REPRODUCE.md',METHOD_DIR/'sources.json',
        *METHOD_DIR.joinpath('code').glob('*.py'),*METHOD_DIR.joinpath('code/manifolds').glob('*.py'),METHOD_DIR/'code/config.yaml'])
def tensor_hash(t):
    t=t.detach().cpu().contiguous();h=hashlib.sha256(str(tuple(t.shape)).encode()+str(t.dtype).encode());h.update(t.numpy().tobytes());return h.hexdigest()
def graph_hash(graph):
    a=graph.coalesce();return hashlib.sha256((str(tuple(a.shape))+tensor_hash(a.indices())+tensor_hash(a.values())).encode()).hexdigest()
