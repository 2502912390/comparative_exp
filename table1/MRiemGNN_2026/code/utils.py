import os

import torch
import numpy as np

def accuracy(pred, labels):
    return (pred == labels).sum().item() / labels.size(0)

def set_seed(seed, deterministic=False):
    import random
    import numpy as np
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

def get_link_pred_split(adj, val_ratio=0.1, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    pos_edges = np.array(np.transpose(np.triu(adj, 1).nonzero()))
    neg_edges = np.array(np.transpose(np.triu(1 - adj, 1).nonzero()))
    np.random.shuffle(pos_edges)
    np.random.shuffle(neg_edges)
    n_pos = len(pos_edges)
    n_val = int(n_pos * val_ratio)
    n_test = int(n_pos * test_ratio)
    n_train = n_pos - n_val - n_test
    train_edges = pos_edges[:n_train]
    val_edges = pos_edges[n_train:n_train+n_val]
    test_edges = pos_edges[n_train+n_val:]
    train_edges_false = neg_edges[:n_train]
    val_edges_false = neg_edges[n_train:n_train+n_val]
    test_edges_false = neg_edges[n_train+n_val:n_train+n_val+n_test]
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def mask_edges_adjs(adj, train_edges, train_edges_false):
    adj_train = np.zeros_like(adj)
    for e in train_edges:
        adj_train[e[0], e[1]] = 1
        adj_train[e[1], e[0]] = 1
    return adj_train
