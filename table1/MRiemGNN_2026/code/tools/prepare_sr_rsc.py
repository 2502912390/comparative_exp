import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SR-RSC/BPHGNN-source datasets to MRiemGNN pt format.")
    parser.add_argument("--input_dir", type=str, default="external/SR-RSC/dataset/imdb")
    parser.add_argument("--output", type=str, default="data/imdb_sr_rsc_pt")
    parser.add_argument("--binarize_features", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def feature_tensor(features, binarize=True):
    if sp.issparse(features):
        features = features.toarray()
    features = np.asarray(features, dtype=np.float32)
    if binarize:
        features = (features > 0).astype(np.float32)
    return torch.from_numpy(features).float()


def edge_tensors(edges):
    edge_indices = []
    edge_types = []
    relation_names = []
    for rel_id, relation in enumerate(sorted(edges.keys())):
        matrix = edges[relation]
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        coo = matrix.tocoo()
        edge_index = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)).long()
        edge_indices.append(edge_index)
        edge_types.append(torch.full((edge_index.size(1),), rel_id, dtype=torch.long))
        relation_names.append(relation)
    return torch.cat(edge_indices, dim=1), torch.cat(edge_types, dim=0), relation_names


def labels_and_splits(labels, num_nodes):
    split_arrays = []
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for split in labels:
        arr = np.asarray(split, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected label split as [node_id, label], got shape={arr.shape}")
        idx = torch.from_numpy(arr[:, 0]).long()
        target = torch.from_numpy(arr[:, 1]).long()
        y[idx] = target
        split_arrays.append(idx)
    if len(split_arrays) != 3:
        raise ValueError(f"Expected train/valid/test labels, got {len(split_arrays)} splits")
    return y, split_arrays[0], split_arrays[1], split_arrays[2]


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    features = feature_tensor(
        load_pickle(input_dir / "node_features.pkl"),
        binarize=args.binarize_features,
    )
    edges = load_pickle(input_dir / "edges.pkl")
    labels = load_pickle(input_dir / "labels.pkl")

    edge_index, edge_type, relation_names = edge_tensors(edges)
    y, train_idx, val_idx, test_idx = labels_and_splits(labels, features.size(0))

    torch.save(features, output / "node_features.pt")
    torch.save(y, output / "labels.pt")
    torch.save(edge_index, output / "edge_index.pt")
    torch.save(edge_type, output / "edge_type.pt")
    torch.save(train_idx, output / "train_idx.pt")
    torch.save(val_idx, output / "val_idx.pt")
    torch.save(test_idx, output / "test_idx.pt")
    (output / "relations.txt").write_text("\n".join(relation_names), encoding="utf-8")

    print(f"Saved SR-RSC dataset to {output}")
    print(f"nodes={features.size(0)} features={features.size(1)} edges={edge_index.size(1)} relations={len(relation_names)}")
    print(f"train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()} labels={(y >= 0).sum().item()}")
    print(f"relations={', '.join(relation_names)}")


if __name__ == "__main__":
    main()
