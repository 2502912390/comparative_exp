from pathlib import Path
import json
import os
import sys
import torch
from shared.coe_benchmark import load_dataset, relation_union, adaptation_metadata
from shared.coe_benchmark.fingerprints import graph_fingerprint
from shared.coe_benchmark.protocol import sha256_file, json_digest
from .model import GCN, official_normalize

METHOD_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = METHOD_DIR.parents[1]
DATASETS = ('acm', 'dblp', 'yelp', 'amazon', 'mag')
CONFIG_KEYS = {'hidden_dim', 'dropout', 'learning_rate', 'weight_decay', 'epochs',
               'early_stopping', 'early_stop_rule', 'optimizer', 'adaptation', 'graph_operator', 'feature_processing', 'configuration_policy', 'training_precision', 'evaluation_precision'}


def read_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != CONFIG_KEYS:
        raise ValueError(f'Invalid config keys: {set(config) ^ CONFIG_KEYS}')
    if (config['adaptation'] != 'relation_union' or config['optimizer'] != 'TF1Adam'
            or config['graph_operator'] != 'tkipf_utils_normalize_adj_transpose_v1'
            or config['feature_processing'] != 'canonical_unchanged'
            or config['hidden_dim'] != 16 or not 0 <= config['dropout'] < 1
            or config['learning_rate'] <= 0 or config['weight_decay'] < 0
            or config['configuration_policy'] != 'author_defaults_no_hpo_v1'
            or config['training_precision'] != 'float32' or config['evaluation_precision'] != 'float64'
            or config['learning_rate'] != 0.01 or config['weight_decay'] != 0.0005
            or config['dropout'] != 0.5 or config['epochs'] <= 0 or config['epochs'] > 200
            or config['early_stopping'] != 10
            or config['early_stop_rule'] != 'official_zero_based_validation_loss_moving_mean'):
        raise ValueError('Config violates the frozen GCN adaptation')
    return config


def core_files():
    files = list((METHOD_DIR / 'reproduce').rglob('*.py'))
    files += list((METHOD_DIR / 'configs').glob('*.json'))
    files += list((METHOD_DIR / 'environment').glob('*.json'))
    files += [METHOD_DIR / 'sources.json', METHOD_DIR / 'ADAPTATION.md']
    files += [METHOD_DIR / 'code/gcn' / name for name in (
        'models.py', 'layers.py', 'utils.py', 'train.py', 'inits.py', 'metrics.py')]
    return sorted(files)


def prepare(dataset, device):
    if dataset not in DATASETS:
        raise ValueError('GCN supports all and only its five registered Table1 datasets')
    data = load_dataset(dataset)
    metadata = adaptation_metadata(data, 'relation_union')
    # metadata checks the features really are the same in every relation view.
    union = relation_union(data.provided_adjs)
    support = official_normalize(union)
    metadata.update(graph_operator='tkipf_utils_normalize_adj_transpose_v1',
                    normalized_graph_sha256=graph_fingerprint([support]),
                    graph_layout='sparse_csr', feature_processing='canonical_unchanged',
                    training_precision='float32', evaluation_precision='float64',
                    evaluation_reason='CUDA CSR FP32 accumulation exceeded strict 1e-6 checkpoint replay tolerance')
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; no CPU fallback')
    features = data.features[0].to(target)
    # CSR avoids dense NxN construction and reduces full-MAG SpMM overhead.
    support = support.to(target).to_sparse_csr()
    labels = data.labels.to(target)
    masks = {key: mask.to(target) for key, mask in data.masks.items()}
    return data, features, support, labels, masks, metadata


def make_model(config, data, device):
    return GCN(data.input_dims[0], data.classes, config['hidden_dim'], config['dropout']).to(device)


def configure_runtime(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    os.environ.setdefault('COE_DATA_ROOT', '/home/chenzhichao/Research/CoE2/data/CoE_dataset')
    os.environ.setdefault('COE_MULTIVIEW_DATA_ROOT', '/home/chenzhichao/Research/CoE2/data/Multi-view datasets')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def assert_source_snapshot(job_dir):
    manifest = json.loads((job_dir / 'manifest.json').read_text())
    if json_digest(manifest['config']) != manifest['config_sha256']:
        raise ValueError('Config hash mismatch')
    for name, expected in manifest['sources']['files'].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise ValueError(f'Current source differs from the frozen run: {name}')
        if sha256_file(job_dir / 'source_snapshot' / name) != expected:
            raise ValueError(f'Snapshot corruption: {name}')
    return manifest
