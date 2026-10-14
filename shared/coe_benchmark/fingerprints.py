"""Historical identity encoding, independent of the public sparse storage."""
import hashlib
import json
import torch


def tensor_digest(hasher, tensor):
    """Exact identity serialization from 1005 joint_data.tensor_digest."""
    tensor = tensor.detach().cpu()
    hasher.update(str(tuple(tensor.shape)).encode())
    hasher.update(str(tensor.dtype).encode())
    if tensor.layout != torch.strided:
        tensor = tensor.to_sparse_coo().coalesce()
        tensor_digest(hasher, tensor.indices())
        tensor_digest(hasher, tensor.values())
    else:
        hasher.update(tensor.contiguous().numpy().tobytes())


def dense_graph_digest(hasher, adjacency, chunk_rows=128):
    """Hash a historical dense graph without recreating a full N x N tensor."""
    graph = adjacency.detach().cpu().to_sparse_coo().coalesce()
    hasher.update(str(tuple(graph.shape)).encode())
    hasher.update(str(graph.dtype).encode())
    row, column = graph.indices()
    for start in range(0, graph.shape[0], chunk_rows):
        end = min(start + chunk_rows, graph.shape[0])
        left = int(torch.searchsorted(row, start))
        right = int(torch.searchsorted(row, end))
        indices = torch.stack((row[left:right] - start, column[left:right]))
        block = torch.sparse_coo_tensor(indices, graph.values()[left:right],
                                       (end - start, graph.shape[1])).to_dense()
        hasher.update(block.contiguous().numpy().tobytes())


def source_fingerprint(dataset, classes, features, graphs, historical_layouts):
    digest = hashlib.sha256()
    digest.update(f"{dataset}:{classes}:row_l2".encode())
    if len(features) != len(graphs) or len(graphs) != len(historical_layouts):
        raise ValueError("Feature/graph/layout counts differ")
    for feature, graph, layout in zip(features, graphs, historical_layouts):
        tensor_digest(digest, feature)
        if layout == "dense":
            if graph.layout == torch.strided:
                tensor_digest(digest, graph)
            else:
                dense_graph_digest(digest, graph)
        elif layout == "sparse":
            tensor_digest(digest, graph.to_sparse_coo().coalesce())
        else:
            raise ValueError(f"Unknown historical graph layout: {layout}")
    return digest.hexdigest()


def split_fingerprint(masks):
    digest = hashlib.sha256()
    for name in ("train", "val", "test"):
        tensor_digest(digest, masks[name])
    return digest.hexdigest()


def label_identity(labels, mapping):
    digest = hashlib.sha256()
    tensor_digest(digest, labels)
    encoded = json.dumps(mapping, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {"labels_sha256": digest.hexdigest(), "class_mapping": list(mapping),
            "class_mapping_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def graph_fingerprint(graphs):
    digest = hashlib.sha256()
    for graph in graphs:
        tensor_digest(digest, graph.to_sparse_coo().coalesce())
    return digest.hexdigest()

