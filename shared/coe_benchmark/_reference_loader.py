"""Extracted pure 1005 data functions, with namespaced imports and per-call roots.

Numerical/split logic is preserved. No legacy path fallback or import-time runner.
See provenance.json for pinned source identities and extraction locations.
"""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from ._paths import get_roots
from ._reference_ops import (normalize, symmetrize, remove_self_loop,
                            sparse_tensor_add_self_loop, adj_values_one,
                            sparse_mx_to_torch_sparse_tensor)


def _data_path(dataset, subdir=""):
    path = get_roots().coe / dataset
    if subdir:
        path = path / subdir
    if not path.is_dir():
        raise FileNotFoundError(path)
    return str(path) + os.sep


MULTIVIEW_NODE_DATASETS = {
    "mv_3sources": "3Sources.mat",
    "mv_acm": "ACM.mat",
    "mv_bbcsport": "BBCSport.mat",
    "mv_citeseer": "CiteSeer.mat",
    "mv_cora": "Cora.mat",
    "mv_movies": "Movies.mat",
    "mv_prokaryotic": "Prokaryotic.mat",
    "mv_reuters1200": "Reuters-1200.mat",
    "mv_webkb": "WebKB.mat",
    "mv_wikipedia": "Wikipedia.mat",
    "mv_handwritten": "hw-6v/handwritten.mat",
    "mv_uci_digit": "uci-digit/uci-digit.mat",
}

MULTIVIEW_GRAPH_SPECS = {
    "mv_acm": {"content_view": 0, "relation_views": (1, 2, 3, 4)},
    "mv_cora": {"content_view": 1, "relation_views": (0, 2, 3)},
    "mv_citeseer": {"content_view": 1, "relation_views": (0,)},
}

def load_mag():

    path = _data_path("mag-4")
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)

    feat = torch.load(path+'feat.pt').float()
    nnodes = len(feat)
    adj_1 = torch.load(path+'pap.pt').coalesce()
    adj_2 = torch.load(path+'pp.pt').coalesce()
    adjs = [adj_1, adj_2]
    adjs = [sparse_tensor_add_self_loop(adj).coalesce() for adj in adjs]
    adjs = [adj_values_one(adj) for adj in adjs]
    adjs = [normalize(adj, mode='sym', sparse=True) for adj in adjs]

    train_index = torch.load(path+'train_index.pt')
    val_index = torch.load(path+'val_index.pt')
    test_index = torch.load(path+'test_index.pt')

    train_mask = torch.zeros(nnodes, dtype=torch.int64)
    train_mask[train_index] = 1
    val_mask = torch.zeros(nnodes, dtype=torch.int64)
    val_mask[val_index] = 1
    test_mask = torch.zeros(nnodes, dtype=torch.int64)
    test_mask[test_index] = 1

    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()
    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs

def load_dblp():
    path = _data_path("dblp", "processed")
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    adjs = torch.load(path+'adj.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]

    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')


    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs

def load_acm():
    path = _data_path("acm", "processed")
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    adjs = torch.load(path+'adj.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]
    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs

def load_yelp():
    path = _data_path("yelp", "processed")
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    feat = F.normalize(feat, p=2, dim=1)
    adjs = torch.load(path+'adj.pt')

    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs

def load_amazon():
    path = _data_path("amazon")
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    feat = feat.type(torch.float32)
    feat = F.normalize(feat, p=2, dim=1)
    adj1 = torch.load(path+'upu.pt')
    adj2 = torch.load(path+'usu.pt')
    adj3 = torch.load(path+'uvu.pt')
    adjs = [adj1,adj2,adj3]
    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')
    adjs = [normalize(adj, mode='sym',sparse=True) for adj in adjs]
    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs

def load_esp(split_seed=42, normalization="row_l2"):
    folder_path = _data_path("esp-game")
    # load data
    L = scipy.io.loadmat(folder_path + 'L.mat')
    txt = scipy.io.loadmat(folder_path + 'txt.mat')
    img = scipy.io.loadmat(folder_path + 'img.mat')
    # print("L:",L.keys())
    # print("txt:",txt.keys())
    # print("img",img.keys())
    L_data = L['L']-1
    txt_data = txt['txt']
    img_data = img['img']
    # shape
    # print("L shape:", np.shape(L_data))
    # print("txt shape:", np.shape(txt_data))
    # print("img shape:", np.shape(img_data))
    # label
    unique_labels = np.unique(L_data)
    num_classes = len(unique_labels)
    # print("Unique labels:", unique_labels)
    # print("Number of classes:", num_classes)
    # create mask
    train_mask, val_mask, test_mask = create_masks(L_data, random_seed=split_seed)
    # print("Train mask count:", np.sum(train_mask))
    # print("Validation mask count:", np.sum(val_mask))
    # print("Test mask count:", np.sum(test_mask))
    # feature
    txt_data = _feature_tensor(txt_data, normalization)
    img_data = _feature_tensor(img_data, normalization)
    view_feature = [txt_data,img_data]
    # adj
    adjs = [create_zero_adjacency_matrix(txt_data.shape) for _ in range(2)]
    # print("adjs:",adjs)
    # to tensor
    label = torch.tensor(L_data.ravel(), dtype=torch.int64)
    # print("Label shape:", label.shape)  #
    # print("Label:", label)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    concat_feature = torch.cat(view_feature, dim=1)
    # print("concat feature shape:",concat_feature.shape)
    # print("label:",label)
    # print("View feature",view_feature)
    return view_feature,txt_data.shape[1],label,num_classes,train_mask, val_mask, test_mask, adjs

def load_flick(split_seed=42, normalization="row_l2"):
    folder_path = _data_path("flickr30k")
    # load data
    L = scipy.io.loadmat(folder_path + 'L.mat')
    txt = scipy.io.loadmat(folder_path + 'txt.mat')
    img = scipy.io.loadmat(folder_path + 'img.mat')
    L_data = L['L']-1
    txt_data = txt['txt']
    img_data = img['img']
    # shape
    # print("L shape:", np.shape(L_data))
    # print("txt shape:", np.shape(txt_data))
    # print("img shape:", np.shape(img_data))
    # label
    unique_labels = np.unique(L_data)
    num_classes = len(unique_labels)
    # print("Unique labels:", unique_labels)
    # print("Number of classes:", num_classes)
    # create mask
    train_mask, val_mask, test_mask = create_masks(L_data, random_seed=split_seed)
    # print("Train mask count:", np.sum(train_mask))
    # print("Validation mask count:", np.sum(val_mask))
    # print("Test mask count:", np.sum(test_mask))
    # feature
    txt_data = _feature_tensor(txt_data, normalization)
    img_data = _feature_tensor(img_data, normalization)
    view_feature = [txt_data,img_data]
    # adj
    adjs = [create_zero_adjacency_matrix(txt_data.shape) for _ in range(2)]
    # print("adjs:",adjs)
    # to tensor
    # label = torch.tensor(L_data, dtype=torch.int64)
    label = torch.tensor(L_data.ravel(), dtype=torch.int64)
    # print("Label shape:", label.shape)
    # print("Label:", label)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    concat_feature = torch.cat(view_feature, dim=1)
    # print("concat feature shape:",concat_feature.shape)
    # print("label:",label)
    # print("View feature",view_feature)
    return view_feature,txt_data.shape[1],label,num_classes,train_mask, val_mask, test_mask, adjs

def load_iap(split_seed=42, normalization="row_l2"):
    folder_path = _data_path("iaprtc12")
    # load data
    L = scipy.io.loadmat(folder_path + 'L.mat')
    txt = scipy.io.loadmat(folder_path + 'txt.mat')
    img = scipy.io.loadmat(folder_path + 'img.mat')
    L_data = L['L']-1
    txt_data = txt['txt']
    img_data = img['img']
    # shape
    # print("L shape:", np.shape(L_data))
    # print("txt shape:", np.shape(txt_data))
    # print("img shape:", np.shape(img_data))
    # label
    unique_labels = np.unique(L_data)
    num_classes = len(unique_labels)
    # print("Unique labels:", unique_labels)
    # print("Number of classes:", num_classes)
    # create mask
    train_mask, val_mask, test_mask = create_masks(L_data, random_seed=split_seed)
    # print("Train mask count:", np.sum(train_mask))
    # print("Validation mask count:", np.sum(val_mask))
    # print("Test mask count:", np.sum(test_mask))
    # feature
    txt_data = _feature_tensor(txt_data, normalization)
    img_data = _feature_tensor(img_data, normalization)
    view_feature = [txt_data,img_data]
    # adj
    adjs = [create_zero_adjacency_matrix(txt_data.shape) for _ in range(2)]
    # print("adjs:",adjs)
    # to tensor
    # label = torch.tensor(L_data, dtype=torch.int64)
    label = torch.tensor(L_data.ravel(), dtype=torch.int64)
    # print("Label shape:", label.shape)
    # print("Label:", label)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    concat_feature = torch.cat(view_feature, dim=1)
    # print("concat feature shape:",concat_feature.shape)
    # print("label:",label)
    return view_feature,txt_data.shape[1],label,num_classes,train_mask, val_mask, test_mask, adjs

def load_nus(split_seed=42, normalization="row_l2"):
    folder_path = _data_path("nus-wide")
    # load data
    L = scipy.io.loadmat(folder_path + 'L.mat')
    txt = scipy.io.loadmat(folder_path + 'txt.mat')
    img = scipy.io.loadmat(folder_path + 'img.mat')
    L_data = L['L']-1
    txt_data = txt['txt']
    img_data = img['img']
    # shape
    # print("L shape:", np.shape(L_data))
    # print("txt shape:", np.shape(txt_data))
    # print("img shape:", np.shape(img_data))
    # label
    unique_labels = np.unique(L_data)
    num_classes = len(unique_labels)
    # print("Unique labels:", unique_labels)
    # print("Number of classes:", num_classes)
    # create mask
    train_mask, val_mask, test_mask = create_masks(L_data, random_seed=split_seed)
    # print("Train mask count:", np.sum(train_mask))
    # print("Validation mask count:", np.sum(val_mask))
    # print("Test mask count:", np.sum(test_mask))
    # feature
    txt_data = _feature_tensor(txt_data, normalization)
    img_data = _feature_tensor(img_data, normalization)
    view_feature = [txt_data,img_data]
    # adj
    adjs = [create_zero_adjacency_matrix(txt_data.shape) for _ in range(2)]
    # print("adjs:",adjs)
    # to tensor
    # label = torch.tensor(L_data, dtype=torch.int64)
    label = torch.tensor(L_data.ravel(), dtype=torch.int64)
    # print("Label shape:", label.shape)
    # print("Label:", label)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    concat_feature = torch.cat(view_feature, dim=1)
    # print("concat feature shape:",concat_feature.shape)
    # print("label:",label)
    return view_feature,txt_data.shape[1],label,num_classes,train_mask, val_mask, test_mask, adjs

def create_masks(label_array, train_ratio=0.5, val_ratio=0.25, random_seed=42):
    label_array = np.asarray(label_array).reshape(-1)
    rng = np.random.RandomState(random_seed)
    data_length = len(label_array)
    unique_labels = np.unique(label_array)
    train_mask = np.zeros(data_length, dtype=bool)
    val_mask = np.zeros(data_length, dtype=bool)
    test_mask = np.zeros(data_length, dtype=bool)

    for label in unique_labels:
        label_indices = np.where(label_array == label)[0]
        n_label_samples = len(label_indices)

        train_size = int(train_ratio * n_label_samples)
        val_size = int(val_ratio * n_label_samples)
        test_size = n_label_samples - train_size - val_size
        if train_size < 2 or val_size < 1 or test_size < 1:
            raise ValueError(f"class {label} cannot support fit/calibration/validation/test")
        indices = rng.permutation(label_indices)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask

def create_zero_adjacency_matrix(feature_shape):
    num_samples = feature_shape[0]
    indices = torch.empty((2, 0), dtype=torch.int64)
    values = torch.empty(0, dtype=torch.float32)
    size = (num_samples, num_samples)
    adjacency_matrix = torch.sparse.FloatTensor(indices, values, size)
    return adjacency_matrix

def create_multiview_masks(label_array, train_ratio=0.5, val_ratio=0.25, random_seed=42):
    rng = np.random.default_rng(random_seed)
    label_array = np.asarray(label_array).reshape(-1)
    train_mask = np.zeros(len(label_array), dtype=bool)
    val_mask = np.zeros(len(label_array), dtype=bool)
    test_mask = np.zeros(len(label_array), dtype=bool)

    for label in np.unique(label_array):
        label_indices = np.where(label_array == label)[0]
        n_label_samples = len(label_indices)
        indices = rng.permutation(label_indices)
        if n_label_samples < 4:
            raise ValueError(f"class {label} needs at least four samples for fit/calibration/validation/test")
        train_size = max(2, int(train_ratio * n_label_samples))
        val_size = max(1, int(val_ratio * n_label_samples))
        if train_size + val_size >= n_label_samples:
            val_size = max(1, n_label_samples - train_size - 1)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

    overlap = (train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)
    if overlap.any() or not np.all(train_mask | val_mask | test_mask):
        raise RuntimeError("multi-view split masks must be disjoint and exhaustive")
    if not val_mask.any() or not test_mask.any():
        raise ValueError("dataset is too small for disjoint train/validation/test masks")
    return train_mask, val_mask, test_mask

def _row_bytes(view, index):
    if sp.issparse(view):
        row = view.getrow(index).toarray()
    else:
        row = np.asarray(view[index])
    return np.ascontiguousarray(row).tobytes()

def _joint_view_key(views, label, index):
    return (int(label[index]),) + tuple(_row_bytes(view, index) for view in views)

def create_wikipedia_masks(views, label_raw, random_seed=42, train_ratio=0.8):
    test_mat = scipy.io.loadmat(
        str(get_roots().multiview / "Wikipedia-test.mat"),
        squeeze_me=False,
        struct_as_record=False,
    )
    test_label_key = "y" if "y" in test_mat else "Y"
    test_labels = np.asarray(test_mat[test_label_key]).reshape(-1)
    test_views = [_coerce_view(view, len(test_labels)) for view in _mat_cell_values(test_mat["X"])]
    if len(test_views) != len(views):
        raise ValueError("Wikipedia train/test files have different view counts")

    label_values = np.unique(label_raw)
    label_to_index = {value: index for index, value in enumerate(label_values)}
    encoded_labels = np.asarray([label_to_index[value] for value in label_raw], dtype=np.int64)
    try:
        encoded_test_labels = np.asarray([label_to_index[value] for value in test_labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Wikipedia test contains an unknown label: {exc.args[0]}") from exc

    candidates = defaultdict(list)
    for index in range(len(encoded_labels)):
        candidates[_joint_view_key(views, encoded_labels, index)].append(index)

    test_indices = []
    for index in range(len(encoded_test_labels)):
        key = _joint_view_key(test_views, encoded_test_labels, index)
        matches = candidates.get(key)
        if not matches:
            raise ValueError("Wikipedia-test contains a row not found in Wikipedia.mat")
        test_indices.append(matches.pop())

    num_nodes = len(encoded_labels)
    test_mask = np.zeros(num_nodes, dtype=bool)
    test_mask[test_indices] = True
    if int(test_mask.sum()) != len(encoded_test_labels):
        raise RuntimeError("Wikipedia test rows did not map one-to-one")

    rng = np.random.default_rng(random_seed)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    for label in np.unique(encoded_labels):
        indices = np.where((encoded_labels == label) & ~test_mask)[0]
        indices = rng.permutation(indices)
        if len(indices) < 2:
            raise ValueError(f"Wikipedia class {label} has too few development samples")
        train_size = min(len(indices) - 1, max(1, int(train_ratio * len(indices))))
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:]] = True

    overlap = (train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)
    if overlap.any() or not np.all(train_mask | val_mask | test_mask):
        raise RuntimeError("Wikipedia masks must be disjoint and exhaustive")
    return train_mask, val_mask, test_mask

def _mat_cell_values(cell_array):
    array = np.asarray(cell_array)
    if array.dtype == object:
        return list(array.ravel())
    return [array]

def _coerce_view(view, num_nodes):
    if sp.issparse(view):
        coerced = view.astype(np.float32)
    else:
        coerced = np.asarray(view, dtype=np.float32)
    if len(coerced.shape) != 2:
        raise ValueError(f"Expected a 2D view matrix, got shape={coerced.shape}")
    if coerced.shape[0] != num_nodes and coerced.shape[1] == num_nodes:
        coerced = coerced.T
    if coerced.shape[0] != num_nodes:
        raise ValueError(f"View row count {coerced.shape[0]} does not match labels {num_nodes}")
    return coerced

def _feature_tensor(view, normalization="row_l2"):
    values = view.toarray() if sp.issparse(view) else np.asarray(view)
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("feature view contains NaN/Inf; repair the source data explicitly")
    tensor = torch.tensor(values, dtype=torch.float32)
    if normalization == "none":
        return tensor
    if normalization != "row_l2":
        raise ValueError(f"unsupported feature normalization: {normalization}")
    return F.normalize(tensor, p=2, dim=1)

def _relation_adjacency(view, num_nodes, column_node_ids=None):
    adjacency = sp.csr_matrix(view, dtype=np.float32)
    if adjacency.shape[0] != num_nodes:
        raise ValueError(f"relation row count {adjacency.shape[0]} does not match {num_nodes} nodes")
    if column_node_ids is not None:
        ids = np.asarray(column_node_ids)
        if ids.dtype.kind not in "iu" or ids.shape != (adjacency.shape[1],):
            raise ValueError("column_node_ids must give one integer node ID per relation column")
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= num_nodes):
            raise ValueError("relation column node IDs must be unique and in range")
        coo = adjacency.tocoo()
        adjacency = sp.csr_matrix((coo.data, (coo.row, ids[coo.col])), shape=(num_nodes, num_nodes))
    elif adjacency.shape[1] < num_nodes:
        raise ValueError("rectangular relation requires verified column_node_ids; trailing zero padding is forbidden")
    elif adjacency.shape[1] > num_nodes:
        raise ValueError(f"relation column count {adjacency.shape[1]} exceeds {num_nodes} nodes")

    if not np.isfinite(adjacency.data).all() or np.any(adjacency.data < 0):
        raise ValueError("relation weights must be finite and nonnegative")
    adjacency.setdiag(1.0)
    row_sum = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_degree = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum != 0
    inv_degree[nonzero] = 1.0 / row_sum[nonzero]
    adjacency = sp.diags(inv_degree) @ adjacency
    return sparse_mx_to_torch_sparse_tensor(adjacency.tocoo()).coalesce()

def load_multiview_node(dataset_name, split_seed=42, normalization="row_l2", relation_mapping=None):
    if dataset_name not in MULTIVIEW_NODE_DATASETS:
        raise ValueError(f"Unsupported multi-view node dataset: {dataset_name}")
    mat_path = get_roots().multiview / MULTIVIEW_NODE_DATASETS[dataset_name]
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    label_key = "y" if "y" in mat else "Y"
    label_raw = np.asarray(mat[label_key]).reshape(-1)
    _, label_encoded = np.unique(label_raw, return_inverse=True)
    num_nodes = len(label_encoded)
    nclasses = int(label_encoded.max() + 1)

    raw_views = [_coerce_view(raw_view, num_nodes) for raw_view in _mat_cell_values(mat["X"])]
    if dataset_name == "mv_wikipedia":
        train_mask, val_mask, test_mask = create_wikipedia_masks(raw_views, label_raw, random_seed=split_seed)
    else:
        train_mask, val_mask, test_mask = create_multiview_masks(label_encoded, random_seed=split_seed)

    mappings = {}
    if relation_mapping is not None:
        manifest = json.loads(Path(relation_mapping).read_text(encoding="utf-8"))
        hasher = hashlib.sha256()
        with mat_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if manifest.get("dataset_sha256") != digest or not manifest.get("source") or manifest.get("index_base") != 0:
            raise ValueError("relation mapping must include matching dataset_sha256, source, and index_base=0")
        mappings = manifest["column_node_ids"]

    graph_spec = MULTIVIEW_GRAPH_SPECS.get(dataset_name)
    view_features = []
    adjs = []
    if graph_spec is None:
        for view in raw_views:
            feature = _feature_tensor(view, normalization)
            view_features.append(feature)
            adjs.append(create_zero_adjacency_matrix(feature.shape))
    else:
        content_feature = _feature_tensor(raw_views[graph_spec["content_view"]], normalization)
        relation_views = set(graph_spec["relation_views"])
        for view_index, view in enumerate(raw_views):
            if view_index in relation_views:
                view_features.append(content_feature.clone())
                try:
                    adjs.append(_relation_adjacency(view, num_nodes, mappings.get(str(view_index))))
                except ValueError as exc:
                    raise ValueError(f"{dataset_name} view {view_index}: {exc}; provide -relation_mapping with verified IDs") from exc
            else:
                feature = _feature_tensor(view, normalization)
                view_features.append(feature)
                adjs.append(create_zero_adjacency_matrix(feature.shape))

    label = torch.tensor(label_encoded, dtype=torch.int64)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    view_dims = [int(feature.shape[1]) for feature in view_features]
    return view_features, view_dims, label, nclasses, train_mask, val_mask, test_mask, adjs

def _load_data(args):
    if args.dataset == 'dblp':
        return load_dblp()
    elif args.dataset == 'mag':
        return load_mag()
    elif args.dataset == 'acm':
        return load_acm()
    elif args.dataset == 'yelp':
        return load_yelp()
    elif args.dataset == 'amazon':
        return load_amazon()
    elif args.dataset == 'esp':
        return load_esp(getattr(args, "split_seed", 42), getattr(args, "feature_normalization", "row_l2"))
    elif args.dataset == 'flickr':
        return load_flick(getattr(args, "split_seed", 42), getattr(args, "feature_normalization", "row_l2"))
    elif args.dataset == 'iapr':
        return load_iap(getattr(args, "split_seed", 42), getattr(args, "feature_normalization", "row_l2"))
    elif args.dataset == 'nus':
        return load_nus(getattr(args, "split_seed", 42), getattr(args, "feature_normalization", "row_l2"))
    elif args.dataset in MULTIVIEW_NODE_DATASETS:
        return load_multiview_node(args.dataset, getattr(args, "split_seed", 42),
                                   getattr(args, "feature_normalization", "row_l2"),
                                   getattr(args, "relation_mapping", None))
    raise ValueError(f"unsupported dataset: {args.dataset}")

def validate_dataset(data):
    features, _, labels, nclasses, train, val, test, adjs = data
    if labels.ndim != 1 or labels.dtype != torch.long or not torch.equal(torch.unique(labels), torch.arange(nclasses)):
        raise ValueError("dataset labels must be contiguous int64 class indices")
    if any(mask.dtype != torch.bool or mask.shape != labels.shape or not mask.any() for mask in (train, val, test)):
        raise ValueError("split masks must be nonempty boolean vectors aligned with labels")
    if ((train & val) | (train & test) | (val & test)).any():
        raise ValueError("train/validation/test masks overlap")
    for mask, name in ((train, "train"), (val, "validation"), (test, "test")):
        if len(torch.unique(labels[mask])) != nclasses:
            raise ValueError(f"{name} split is missing classes")
    views = features if isinstance(features, list) else [features]
    if len(adjs) < 2:
        raise ValueError("CoE requires at least two graph views")
    if any(view.ndim != 2 or view.shape[0] != len(labels) or not torch.isfinite(view).all() for view in views):
        raise ValueError("invalid feature shape or NaN/Inf in dataset")
    for adj in adjs:
        weights = adj.coalesce().values() if adj.is_sparse else adj
        if adj.shape != (len(labels), len(labels)) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("invalid adjacency shape or weights")
    return data

def load_data(args):
    return validate_dataset(_load_data(args))
