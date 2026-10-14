from pathlib import Path
import json
import os
import torch
from shared.coe_benchmark import load_dataset, relation_union, adaptation_metadata
from .model import build_model

METHOD = Path(__file__).resolve().parents[1]
REPO = METHOD.parents[1]
DATASETS = ('acm', 'dblp', 'yelp', 'amazon', 'mag')

def read_config(path=None):
    config = json.loads((METHOD / 'configs/default.json').read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        if supplied != config:
            raise ValueError('Only the precommitted author-default configuration is accepted')
    return config

def setup():
    os.environ['COE_DATA_ROOT'] = '/home/chenzhichao/Research/CoE2/data/CoE_dataset'
    os.environ['COE_MULTIVIEW_DATA_ROOT'] = '/home/chenzhichao/Research/CoE2/data/Multi-view datasets'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def inputs(dataset, device):
    data = load_dataset(dataset)
    union = relation_union(data.provided_adjs)
    # Public adjacency row=receiver, col=sender; official edge_index=(sender,receiver).
    edge = union.indices().flip(0).to(device)
    return data, data.features[0].to(device), [edge]

def sources():
    files = list((METHOD/'reproduce').glob('*.py'))
    files += list((METHOD/'reproduce').glob('*.sh'))
    files += list((METHOD/'configs').glob('*.json')) + list((METHOD/'environment').glob('*.json'))
    files += [METHOD/'code'/n for n in ('nodeformer.py','main.py','main-batch.py','parse.py','data_utils.py','run.sh')]
    files += [METHOD/'sources.json', METHOD/'ADAPTATION.md']
    return files
