"""Canonical benchmark data: all views, identities, masks and sparse provided graphs."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np
import torch
from scipy.io import loadmat

from ._paths import DataRoots, _ACTIVE_ROOTS
from . import _reference_loader
from .fingerprints import (source_fingerprint, split_fingerprint, label_identity,
                           graph_fingerprint)
from .protocol import (read_registry, dataset_record, check_reference_sources,
                       IdentityMismatch, DATA_PROTOCOL_ID, PROTOCOL_ID)


@dataclass(frozen=True)
class BenchmarkData:
    dataset: str
    table: int
    family: str
    features: tuple
    provided_adjs: tuple
    labels: torch.Tensor
    masks: dict
    class_mapping: tuple
    historical_graph_layouts: tuple
    fingerprints: dict
    roots: DataRoots

    @property
    def nodes(self):
        return len(self.labels)

    @property
    def classes(self):
        return len(self.class_mapping)

    @property
    def input_dims(self):
        return [view.shape[1] for view in self.features]

    @property
    def node_ids(self):
        return torch.arange(self.nodes, dtype=torch.long, device=self.labels.device)

    def indices(self, split):
        if split not in ("train", "val", "test"):
            raise ValueError("Only train/val/test splits are registered")
        return self.masks[split].nonzero(as_tuple=False).flatten()

    def to(self, device):
        return replace(self, features=tuple(view.to(device) for view in self.features),
                       provided_adjs=tuple(graph.to(device) for graph in self.provided_adjs),
                       labels=self.labels.to(device),
                       masks={key: mask.to(device) for key, mask in self.masks.items()})

    def identity(self):
        return {"protocol_id": PROTOCOL_ID, "data_protocol_id": DATA_PROTOCOL_ID,
                "dataset": self.dataset, "table": self.table, "family": self.family,
                "nodes": self.nodes, "classes": self.classes, "views": len(self.features),
                "input_dims": self.input_dims,
                "split_sizes": {key: int(mask.sum()) for key, mask in self.masks.items()},
                "fingerprints": dict(self.fingerprints),
                "diagnostic_identity": label_identity(self.labels, self.class_mapping),
                "historical_graph_layouts": list(self.historical_graph_layouts)}


def verify_identity(data, registry=None):
    """Recompute all canonical identities; mutations cannot reuse stale hashes."""
    registry = read_registry() if registry is None else registry
    record = dataset_record(data.dataset, registry)
    if len(data.features) != record["views"] or len(data.provided_adjs) != record["views"]:
        raise IdentityMismatch(data.dataset, "views", record["views"], len(data.features))
    for field, actual, expected in (
        ("nodes", data.nodes, record["nodes"]),
        ("classes", data.classes, record["classes"]),
        ("input_dims", data.input_dims, record["input_dims"]),
        ("split_sizes", {key: int(mask.sum()) for key, mask in data.masks.items()}, record["split_sizes"]),
    ):
        if actual != expected:
            raise IdentityMismatch(data.dataset, field, expected, actual)
    if data.labels.dtype != torch.long or data.labels.ndim != 1:
        raise ValueError("Labels must be a 1D int64 vector")
    if not torch.equal(torch.unique(data.labels).cpu(), torch.arange(data.classes)):
        raise ValueError("Class IDs must be contiguous")
    if set(data.masks) != {"train", "val", "test"}:
        raise ValueError("Exactly three outer masks required")
    train, val, test = (data.masks[key] for key in ("train", "val", "test"))
    if any(mask.shape != data.labels.shape or mask.dtype != torch.bool for mask in (train, val, test)):
        raise ValueError("Masks must be boolean and aligned")
    if ((train & val) | (train & test) | (val & test)).any() or not (train | val | test).all():
        raise ValueError("Outer masks must be disjoint and exhaustive")
    for mask in (train, val, test):
        if len(torch.unique(data.labels[mask])) != data.classes:
            raise ValueError("Every split must contain every class")
    for view, graph in zip(data.features, data.provided_adjs):
        if view.ndim != 2 or view.shape[0] != data.nodes or not torch.isfinite(view).all():
            raise ValueError("Non-finite or misaligned feature matrix")
        if graph.layout == torch.strided or graph.shape != (data.nodes, data.nodes):
            raise ValueError("Public graph must be square, aligned and sparse")
        weights = graph.to_sparse_coo().coalesce().values()
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Invalid graph weights")
    actual = {
        "source": source_fingerprint(data.dataset, data.classes, data.features,
                                     data.provided_adjs, data.historical_graph_layouts),
        "splits": split_fingerprint(data.masks),
        **label_identity(data.labels, data.class_mapping),
    }
    graph_hash = graph_fingerprint(data.provided_adjs)
    if data.fingerprints.get("provided_graphs_sparse") != graph_hash:
        raise IdentityMismatch(data.dataset, "provided_graphs_sparse", graph_hash,
                              data.fingerprints.get("provided_graphs_sparse"))
    expected = {"source": record["canonical_source_fingerprint"],
                "splits": record["split_fingerprint"], **record["diagnostic_identity"]}
    for key, value in expected.items():
        if actual[key] != value:
            raise IdentityMismatch(data.dataset, key, value, actual[key])
        if data.fingerprints.get(key) != actual[key]:
            raise IdentityMismatch(data.dataset, "declared_"+key, actual[key], data.fingerprints.get(key))
    return actual


def load_dataset(dataset, roots=None, *, registry=None, relation_mapping=None):
    """Load and verify frozen data. No trained graph, encoder, or weights are used."""
    registry = read_registry() if registry is None else registry
    record = dataset_record(dataset, registry)
    check_reference_sources(registry)
    roots = DataRoots.from_environment() if roots is None else roots
    roots = DataRoots(roots.coe.resolve(), roots.multiview.resolve())
    token = _ACTIVE_ROOTS.set(roots)
    try:
        raw = _reference_loader.load_data(SimpleNamespace(
            dataset=dataset, split_seed=42, feature_normalization="row_l2",
            relation_mapping=relation_mapping))
    finally:
        _ACTIVE_ROOTS.reset(token)
    original, _, labels, classes, train, val, test, adjacency = raw
    features = (tuple(original.float() for _ in adjacency) if torch.is_tensor(original)
                else tuple(view.float() for view in original))
    layouts = tuple("dense" if graph.layout == torch.strided else "sparse" for graph in adjacency)
    # Preserve the original dense/sparse serialization before public sparsification.
    source = source_fingerprint(dataset, classes, features, adjacency, layouts)
    graphs = tuple(graph.to_sparse_coo().coalesce() for graph in adjacency)
    masks = {"train": train.bool(), "val": val.bool(), "test": test.bool()}
    if dataset.startswith("mv_"):
        raw_labels = loadmat(str(roots.multiview / _reference_loader.MULTIVIEW_NODE_DATASETS[dataset]),
                             variable_names=["y", "Y"])
        mapping = np.unique(raw_labels["y"] if "y" in raw_labels else raw_labels["Y"]).tolist()
    elif record["table"] == 2:
        mapping = list(range(1, classes + 1))
    else:
        mapping = list(range(classes))
    fingerprints = {"source": source, "splits": split_fingerprint(masks),
                    **label_identity(labels, mapping),
                    "provided_graphs_sparse": graph_fingerprint(graphs)}
    data = BenchmarkData(dataset, record["table"], record["family"], features, graphs,
                         labels, masks, tuple(mapping), layouts, fingerprints, roots)
    verify_identity(data, registry)
    return data

