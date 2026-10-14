"""Protocol input adaptations. These are not CCoE's learned graph mechanisms."""
import torch
from torch.nn import functional as F
from .fingerprints import graph_fingerprint, tensor_digest
import hashlib


def relation_union(graphs):
    """Unweighted union of positive off-diagonal relations, then one unit self loop.

    Returns raw sparse support. The method must apply its own graph operator;
    no CCoE normalization, learned weights, parameters or labels enter this rule.
    """
    if not graphs:
        raise ValueError("At least one relation graph is required")
    size, device = graphs[0].shape, graphs[0].device
    if len(size) != 2 or size[0] != size[1]:
        raise ValueError("Relation graph must be square")
    endpoints = []
    for graph in graphs:
        graph = graph.to_sparse_coo().coalesce()
        if graph.shape != size or graph.device != device:
            raise ValueError("Relations must align on shape and device")
        if not torch.isfinite(graph.values()).all() or (graph.values() < 0).any():
            raise ValueError("Relations must have finite nonnegative weights")
        row, column = graph.indices()
        keep = row.ne(column) & graph.values().gt(0)
        endpoints.append(graph.indices()[:, keep])
    diagonal = torch.arange(size[0], device=device)
    endpoints.append(torch.stack((diagonal, diagonal)))
    indices = torch.cat(endpoints, dim=1)
    raw = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], device=device), size).coalesce()
    return torch.sparse_coo_tensor(raw.indices(), torch.ones_like(raw.values()), size).coalesce()


@torch.no_grad()
def graph_to_feature_context_v1(data):
    """For MV relation views only: concat(X, row_l2(A^2 X)); no view is dropped."""
    relation_views = {"mv_acm": (1, 2, 3, 4), "mv_citeseer": (0,)}.get(data.dataset, ())
    features = []
    for index, (view, graph) in enumerate(zip(data.features, data.provided_adjs)):
        if index in relation_views:
            context = view.detach()
            for _ in range(2):
                context = torch.sparse.mm(graph, context)
            features.append(torch.cat((view, F.normalize(context, p=2, dim=-1)), dim=-1))
        else:
            features.append(view)
    return tuple(features)


def adaptation_metadata(data, adaptation):
    if adaptation == "canonical":
        return {"id": adaptation, "input_dims": data.input_dims,
                "provided_graphs_sparse_sha256": graph_fingerprint(data.provided_adjs),
                "role": "canonical observations; no model graph"}
    if adaptation == "relation_union":
        if any(view.shape != data.features[0].shape or not torch.equal(view, data.features[0])
               for view in data.features[1:]):
            raise ValueError("Single-graph relation_union requires common node features")
        union = relation_union(data.provided_adjs)
        return {"id": adaptation, "input_dims": [data.input_dims[0]],
                "graph_sha256": graph_fingerprint([union]), "edges": union._nnz(),
                "role": "untrained input adaptation, not CCoE model mechanism"}
    if adaptation == "graph_to_feature_context_v1":
        effective = graph_to_feature_context_v1(data)
        digest = hashlib.sha256()
        for view in effective:
            tensor_digest(digest, view)
        return {"id": adaptation, "input_dims": [view.shape[1] for view in effective],
                "effective_features_sha256": digest.hexdigest(),
                "role": "untrained input adaptation, not CCoE model mechanism"}
    raise ValueError(f"Unknown adaptation: {adaptation}")

