"""Canonical shared inputs and frozen own-view feature graphs."""
from pathlib import Path
import hashlib
import os
import torch
import torch.nn.functional as F
from shared.coe_benchmark import adaptation_metadata
from shared.coe_benchmark.protocol import json_digest
from .mechanism import learned_graph, graph_mm, pack_graph, unpack_graph


def fixed_feature_graphs(data, cfg):
    """Build the untrained Table 2 input once on CPU, then verify/reuse it.

    CPU avoids CUDA scatter rounding changing a fixed input's byte fingerprint.
    The original exact top-k, symmetrization, ReLU and normalization stay intact.
    Learned graph training still uses its original device-specific formulas.
    """
    recipe = {'rule': 'infomgf_feature_views_cosine_knn_v1',
              'identity': data.identity(), 'k': cfg['k'],
              'graph_block': cfg['graph_block'], 'torch': str(torch.__version__),
              'mechanism_sha256': hashlib.sha256(Path(__file__).with_name('mechanism.py').read_bytes()).hexdigest(),
              'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key = json_digest(recipe)
    cache = Path(__file__).resolve().parents[1] / 'exp/_cache/initial_graphs'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (data.dataset + '_' + key + '.pt')
    if path.exists():
        saved = torch.load(path, map_location='cpu')
        if saved['recipe'] != recipe or saved['key'] != key:
            raise ValueError('Fixed input graph cache identity changed')
    else:
        with torch.no_grad():
            states = [pack_graph(learned_graph(f.cpu(), cfg['k'], False, 0,
                                               False, cfg['graph_block']))
                      for f in data.features]
        saved = {'key': key, 'recipe': recipe, 'graphs': states}
        temp = path.with_suffix('.tmp.' + str(os.getpid()))
        torch.save(saved, temp)
        os.replace(temp, path)
    if len(saved['graphs']) != len(data.features):
        raise ValueError('Fixed graph cache dropped a view')
    graphs = [unpack_graph(state, 'cpu') for state in saved['graphs']]
    if any(g.shape != (data.nodes, data.nodes) for g in graphs):
        raise ValueError('Fixed graph cache node identity changed')
    return graphs, key


def prepare(data, cfg, device):
    features = [f.to(device) for f in data.features]
    if data.table == 1:
        x = features[0]
        if not all(torch.equal(x, f) for f in features):
            raise ValueError('Native multiplex input expects identical features per relation')
        adjs = [a.to(device) for a in data.provided_adjs]
        view_features = features
        meta = adaptation_metadata(data, 'canonical')
    elif data.table == 2:
        x = torch.cat(features, dim=1)
        fixed, key = fixed_feature_graphs(data, cfg)
        adjs = [a.to(device) for a in fixed]
        view_features = features
        meta = adaptation_metadata(data, 'canonical')
        meta.update(adaptation_id='infomgf_feature_views_cosine_knn_v1',
                    shared_features='concat_text_image', provided_empty_graphs_used=False,
                    initial_graphs=[{k: v for k, v in pack_graph(a).items()
                                     if k in ('type', 'n', 'sha256')} for a in fixed],
                    initial_graph_k=cfg['k'], initial_graph_cache_key=key,
                    initial_graph_build_device='cpu')
    else:
        raise ValueError('InfoMGF registered scope is Tables 1 and 2')
    contexts = []
    with torch.no_grad():
        for f, a in zip(view_features, adjs):
            z = f
            for _ in range(cfg['r']):
                z = graph_mm(a, z)
            contexts.append(F.normalize(z, dim=1, p=2))
    return x, adjs, contexts, meta
