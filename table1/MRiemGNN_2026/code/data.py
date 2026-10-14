import itertools
from collections import defaultdict

import numpy as np
import scipy.io
import scipy.sparse
import torch


STRUCTURAL_FIELDS = {
    "__header__",
    "__version__",
    "__globals__",
    "feature",
    "features",
    "full_feature",
    "attr",
    "attribute",
    "x",
    "feat",
    "label",
    "labels",
    "y",
    "target",
    "gt",
    "train_idx",
    "valid_idx",
    "val_idx",
    "test_idx",
    "train",
    "valid",
    "val",
    "test",
    "train_mask",
    "valid_mask",
    "val_mask",
    "test_mask",
    "train_edges",
    "train_edges_neg",
    "valid_edges",
    "valid_edges_neg",
    "test_edges",
    "test_edges_neg",
    "item_num",
    "user_num",
    "node_num",
    "n_nodes",
    "class",
}


def _to_numpy(field):
    if scipy.sparse.issparse(field):
        return field.toarray()
    return np.asarray(field)


def _flatten_index(field):
    arr = _to_numpy(field)
    return arr.reshape(-1).astype(np.int64)


def _normalize_features(field):
    arr = _to_numpy(field).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Feature field must be 2D, got shape={arr.shape}")
    return arr


def _normalize_labels(field, num_nodes):
    arr = _to_numpy(field)
    if arr.ndim == 2 and arr.shape[1] > 1 and arr.shape[0] <= num_nodes:
        labels = arr.argmax(axis=1).astype(np.int64)
        if labels.size == num_nodes:
            return labels
        padded = np.full(num_nodes, -1, dtype=np.int64)
        padded[: labels.size] = labels
        return padded
    if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] <= num_nodes:
        labels = arr.argmax(axis=0).astype(np.int64)
        if labels.size == num_nodes:
            return labels
        padded = np.full(num_nodes, -1, dtype=np.int64)
        padded[: labels.size] = labels
        return padded
    arr = arr.reshape(-1)
    labels = arr.astype(np.int64)
    if labels.size == num_nodes:
        return labels
    if labels.size < num_nodes:
        padded = np.full(num_nodes, -1, dtype=np.int64)
        padded[: labels.size] = labels
        return padded
    raise ValueError(f"Label length {labels.size} exceeds node count {num_nodes}")


def _normalize_edges(arr):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2:
        raise ValueError(f"Edge array must be 2D, got shape={arr.shape}")
    if arr.shape[1] == 2:
        return arr
    if arr.shape[0] == 2:
        return arr.T
    raise ValueError(f"Edge array shape not recognized: {arr.shape}")


def load_link_pred_edges(mat_path):
    mat = scipy.io.loadmat(mat_path)
    splits = {}
    for split in ("train", "valid", "test"):
        pos = _normalize_edges(mat.get(f"{split}_edges"))
        neg = _normalize_edges(mat.get(f"{split}_edges_neg"))
        splits[split] = (pos, neg)
    return splits


def build_adj_dict(edge_index, edge_type, num_relations, composite=False, relation_names=None, add_reverse_edges=False):
    if add_reverse_edges and edge_index.numel():
        rev_edge_index = edge_index.flip(0)
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        edge_type = torch.cat([edge_type, edge_type], dim=0)

    if not composite:
        adj_dict = {}
        rel_names = relation_names or [f"relation_{i}" for i in range(num_relations)]
        for r in range(num_relations):
            idx = (edge_type == r).nonzero(as_tuple=False).view(-1)
            adj_dict[r] = edge_index[:, idx] if idx.numel() else torch.zeros(2, 0, dtype=torch.long)
        return adj_dict, rel_names

    pairs = defaultdict(int)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    rel = edge_type.cpu().numpy()
    for u, v, r in zip(src, dst, rel):
        pairs[(int(u), int(v))] |= 1 << int(r)

    grouped = defaultdict(list)
    for pair, mask in pairs.items():
        grouped[mask].append(pair)

    masks = sorted(grouped)
    adj_dict = {}
    rel_names = []
    base_names = relation_names or [f"relation_{i}" for i in range(num_relations)]
    for new_id, mask in enumerate(masks):
        edges = np.asarray(grouped[mask], dtype=np.int64)
        adj_dict[new_id] = torch.from_numpy(edges.T).long()
        parts = [base_names[i] for i in range(num_relations) if mask & (1 << i)]
        rel_names.append("+".join(parts))
    return adj_dict, rel_names


def load_data(
    prefix="data/",
    x_file="node_features.pt",
    y_file="labels.pt",
    edge_index_file="edge_index.pt",
    edge_type_file="edge_type.pt",
    train_file="train_idx.pt",
    val_file="val_idx.pt",
    test_file="test_idx.pt",
):
    x = torch.load(prefix + x_file)
    y = torch.load(prefix + y_file)
    edge_index = torch.load(prefix + edge_index_file)
    edge_type = torch.load(prefix + edge_type_file)
    train_idx = torch.load(prefix + train_file)
    val_idx = torch.load(prefix + val_file)
    test_idx = torch.load(prefix + test_file)
    num_relations = int(edge_type.max().item()) + 1
    return x, y, edge_index, edge_type, train_idx, val_idx, test_idx, num_relations


def _is_square_matrix(value, n):
    try:
        shape = value.shape if scipy.sparse.issparse(value) else np.asarray(value).shape
        return len(shape) == 2 and shape[0] == shape[1] == n
    except Exception:
        return False


def _auto_detect_relation_fields(mat, node_num):
    relation_fields = []
    for key, value in mat.items():
        if key in STRUCTURAL_FIELDS:
            continue
        if isinstance(value, np.ndarray) and value.dtype == np.object_:
            for item in value.flatten():
                if _is_square_matrix(item, node_num):
                    relation_fields.append((key, item))
        elif _is_square_matrix(value, node_num):
            relation_fields.append((key, value))
    if not relation_fields:
        raise ValueError("No square adjacency relation fields were found in the .mat file.")
    return relation_fields


def _detect_split_relation_fields(mat, node_num, split_mode="all"):
    split_keys = ("train", "valid", "test") if split_mode == "all" else (split_mode,)
    split_arrays = []
    for key in split_keys:
        value = mat.get(key)
        if not isinstance(value, np.ndarray) or value.dtype != np.object_:
            continue
        matrices = list(value.flatten())
        if matrices and all(_is_square_matrix(matrix, node_num) for matrix in matrices):
            split_arrays.append((key, matrices))

    if not split_arrays:
        return []

    num_relations = len(split_arrays[0][1])
    relation_fields = []
    for rel_id in range(num_relations):
        merged = None
        used_splits = []
        for split_name, matrices in split_arrays:
            matrix = matrices[rel_id]
            matrix = matrix.tocsr() if scipy.sparse.issparse(matrix) else scipy.sparse.csr_matrix(matrix)
            merged = matrix.copy() if merged is None else merged + matrix
            used_splits.append(split_name)
        merged.data[:] = 1
        merged.setdiag(0)
        merged.eliminate_zeros()
        relation_fields.append((f"{'+'.join(used_splits)}_relation_{rel_id}", merged.tocoo()))
    return relation_fields


def _auto_generate_metapaths(adj_fields, max_length=2):
    if max_length < 2:
        return []
    meta_adjs = []
    for length in range(2, max_length + 1):
        for path in itertools.product(adj_fields, repeat=length):
            name = "_".join(item[0] for item in path)
            adj = path[0][1]
            if not scipy.sparse.issparse(adj):
                adj = scipy.sparse.coo_matrix(adj)
            for _, next_adj in path[1:]:
                if not scipy.sparse.issparse(next_adj):
                    next_adj = scipy.sparse.coo_matrix(next_adj)
                adj = adj @ next_adj
            adj = adj.tocoo()
            adj.setdiag(0)
            adj.eliminate_zeros()
            meta_adjs.append((name, adj))
    return meta_adjs


def _auto_detect_fields(mat):
    feature_candidates = ("feature", "features", "x", "feat", "attr", "attribute", "full_feature")
    label_candidates = ("label", "labels", "y", "target", "gt")
    train_candidates = ("train_idx", "train", "train_mask")
    valid_candidates = ("valid_idx", "val_idx", "valid", "val", "valid_mask", "val_mask")
    test_candidates = ("test_idx", "test", "test_mask")

    def first_present(candidates):
        return next((field for field in candidates if field in mat), None)

    fields = (
        first_present(feature_candidates),
        first_present(label_candidates),
        first_present(train_candidates),
        first_present(valid_candidates),
        first_present(test_candidates),
    )
    if not all(fields):
        raise ValueError(f"Could not auto-detect feature/label/split fields. Found: {list(mat.keys())}")
    return fields


def load_mat_multiplex_auto(
    mat_path,
    enable_metapath=False,
    max_metapath_length=2,
    use_full_graph=False,
    graph_relation_split="all",
    split_mat_path=None,
    feature_field=None,
    label_field=None,
    train_idx_field=None,
    val_idx_field=None,
    test_idx_field=None,
    node_num_field="item_num",
):
    mat = scipy.io.loadmat(mat_path)
    split_mat = scipy.io.loadmat(split_mat_path) if split_mat_path else mat
    if use_full_graph and feature_field is None and "full_feature" in mat:
        feature_field = "full_feature"

    if not feature_field or not label_field:
        feature_candidates = ("feature", "features", "x", "feat", "attr", "attribute", "full_feature")
        label_candidates = ("label", "labels", "y", "target", "gt")
        feature_field = feature_field or next((field for field in feature_candidates if field in mat), None)
        label_field = label_field or next((field for field in label_candidates if field in mat), None)
        if feature_field is None or label_field is None:
            raise ValueError(f"Could not auto-detect feature/label fields. Found: {list(mat.keys())}")

    if not all([train_idx_field, val_idx_field, test_idx_field]):
        if all(field in split_mat for field in ("train_idx", "valid_idx", "test_idx")):
            train_idx_field = train_idx_field or "train_idx"
            val_idx_field = val_idx_field or "valid_idx"
            test_idx_field = test_idx_field or "test_idx"
        else:
            _, _, detected_train, detected_val, detected_test = _auto_detect_fields(mat)
            train_idx_field = train_idx_field or detected_train
            val_idx_field = val_idx_field or detected_val
            test_idx_field = test_idx_field or detected_test

    x_np = _normalize_features(mat[feature_field])
    node_num = int(mat[node_num_field][0][0]) if node_num_field in mat else x_np.shape[0]
    if use_full_graph:
        node_num = x_np.shape[0]
    if x_np.shape[0] != node_num:
        raise ValueError(f"Feature rows {x_np.shape[0]} do not match node count {node_num}")

    adj_fields = _detect_split_relation_fields(mat, node_num, graph_relation_split) if use_full_graph else []
    if not adj_fields:
        adj_fields = _auto_detect_relation_fields(mat, node_num)
    if enable_metapath:
        adj_fields = adj_fields + _auto_generate_metapaths(adj_fields, max_metapath_length)

    edge_indices = []
    edge_types = []
    relation_names = []
    for rel_id, (rel_name, adj) in enumerate(adj_fields):
        coo = adj.tocoo() if scipy.sparse.issparse(adj) else scipy.sparse.coo_matrix(adj)
        mask = coo.row != coo.col
        edges = np.vstack((coo.row[mask], coo.col[mask]))
        if edges.shape[1] == 0:
            continue
        edge_indices.append(edges)
        edge_types.append(np.full(edges.shape[1], rel_id, dtype=np.int64))
        relation_names.append(rel_name)

    if not edge_indices:
        raise ValueError("No non-self-loop edges were found in relation fields.")

    y_np = _normalize_labels(mat[label_field], node_num)
    train_idx = _flatten_index(split_mat[train_idx_field])
    val_idx = _flatten_index(split_mat[val_idx_field])
    test_idx = _flatten_index(split_mat[test_idx_field])

    edge_index = np.hstack(edge_indices)
    edge_type = np.concatenate(edge_types)
    return (
        torch.from_numpy(x_np).float(),
        torch.from_numpy(y_np).long(),
        torch.from_numpy(edge_index).long(),
        torch.from_numpy(edge_type).long(),
        torch.from_numpy(train_idx).long(),
        torch.from_numpy(val_idx).long(),
        torch.from_numpy(test_idx).long(),
        len(relation_names),
        relation_names,
    )
