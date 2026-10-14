import argparse
from pathlib import Path

import torch
import torch.optim as optim

from data import build_adj_dict, load_data, load_link_pred_edges, load_mat_multiplex_auto
from model_mriemgnn import MRiemGNN
from train import train
from utils import set_seed


DEFAULT_DATA = "data/small_alibaba_1_10/small_alibaba_1_10.mat"
DEFAULT_LP_SPLIT = "data/small_alibaba_1_10/small_alibaba_1_10_linkpred_split.mat"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MRiemGNN.")
    parser.add_argument("--data_type", type=str, default="mat", choices=["pt", "mat"])
    parser.add_argument("--pt_dir", type=str, default="data/")
    parser.add_argument("--mat_path", type=str, default=DEFAULT_DATA)
    parser.add_argument("--split_mat_path", type=str, default="")
    parser.add_argument("--feature_path", type=str, default="")
    parser.add_argument("--feature_encoding", type=str, default="raw", choices=["raw", "onehot"])
    parser.add_argument("--feature_columns", type=str, default="")
    parser.add_argument("--resplit_labeled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--split_strategy", type=str, default="stratified", choices=["stratified", "random"])
    parser.add_argument("--link_split_path", type=str, default=DEFAULT_LP_SPLIT)
    parser.add_argument("--task", type=str, default="node_cls", choices=["node_cls", "link_pred"])
    parser.add_argument("--enable_metapath", action="store_true")
    parser.add_argument("--max_metapath_length", type=int, default=2)
    parser.add_argument("--composite_relations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_full_graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_reverse_edges", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "plateau"])
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=80)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_kernels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernel_learnable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shared_space_kernel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learnable_beta", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hyperbolic_proj", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input_skip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input_skip_weight", type=float, default=0.5)
    parser.add_argument("--layer_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--relation_dropout", type=float, default=0.2)
    parser.add_argument("--node_relation_attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--node_relation_attention_mix", type=float, default=1.0)
    parser.add_argument("--lambda_mutual", type=float, default=0.01)
    parser.add_argument("--mutual_target", type=str, default="embeddings", choices=["embeddings", "logits"])
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--classifier_hidden", type=int, default=0)
    parser.add_argument("--feature_logit_weight", type=float, default=0.0)
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"])
    parser.add_argument("--class_weight_strength", type=float, default=1.0)
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument("--feature_norm", type=str, default="standard", choices=["none", "standard", "row"])
    parser.add_argument("--augment_degree_features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--node_embedding_dim", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lambda_behavior", type=float, default=0.0)
    parser.add_argument("--behavior_tau", type=float, default=0.5)
    parser.add_argument("--behavior_projector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval_mode", type=str, default="classifier", choices=["classifier", "logreg"])
    parser.add_argument("--logreg_epochs", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--logit_smoothing_steps", type=int, default=0)
    parser.add_argument("--logit_smoothing_alpha", type=float, default=0.5)
    parser.add_argument("--logit_smoothing_use_train_labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logit_prior_alpha", type=float, default=0.0)
    parser.add_argument("--train_logit_prior_alpha", type=float, default=0.0)
    parser.add_argument("--logit_prior_search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logit_prior_grid_min", type=float, default=-1.0)
    parser.add_argument("--logit_prior_grid_max", type=float, default=2.0)
    parser.add_argument("--logit_prior_grid_steps", type=int, default=31)
    parser.add_argument("--prototype_logit_weight", type=float, default=0.0)
    parser.add_argument("--prototype_temperature", type=float, default=0.2)
    parser.add_argument("--prototype_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--save_logits_path", type=str, default="")
    parser.add_argument("--merge_val_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor_split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--monitor_metric", type=str, default="macro_f1", choices=["macro_f1", "micro_f1", "acc"])
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_graph(args):
    graph_relation_split = "train" if args.task == "link_pred" else "all"
    if args.data_type == "pt":
        pt_dir = args.pt_dir if args.pt_dir.endswith(("/", "\\")) else args.pt_dir + "/"
        x, y, edge_index, edge_type, train_idx, val_idx, test_idx, num_relations = load_data(prefix=pt_dir)
        relation_names = [f"relation_{i}" for i in range(num_relations)]
    else:
        x, y, edge_index, edge_type, train_idx, val_idx, test_idx, num_relations, relation_names = (
            load_mat_multiplex_auto(
                args.mat_path,
                enable_metapath=args.enable_metapath,
                max_metapath_length=args.max_metapath_length,
                use_full_graph=args.use_full_graph,
                graph_relation_split=graph_relation_split,
                split_mat_path=args.split_mat_path or None,
            )
        )
        if args.feature_path:
            x = load_feature_override(
                args.feature_path,
                x.size(0),
                encoding=args.feature_encoding,
                columns=parse_feature_columns(args.feature_columns),
            )

    adj_dict, relation_names = build_adj_dict(
        edge_index,
        edge_type,
        num_relations,
        composite=args.composite_relations,
        relation_names=relation_names,
        add_reverse_edges=args.add_reverse_edges,
    )
    data = {
        "x": x,
        "y": y,
        "adj_dict": adj_dict,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    if args.resplit_labeled:
        split_seed = args.seed if args.split_seed is None else args.split_seed
        data["train_idx"], data["val_idx"], data["test_idx"] = make_train_test_split(
            data["y"], train_ratio=args.train_ratio, seed=split_seed, strategy=args.split_strategy
        )
    if args.augment_degree_features:
        data["x"] = torch.cat([data["x"], degree_features(data["adj_dict"], data["x"].size(0))], dim=1)
    data["behavior_x"] = data["x"].clone()
    data["x"] = normalize_features(data["x"], args.feature_norm)
    data["behavior_x"] = normalize_features(data["behavior_x"], args.feature_norm)
    return data, list(adj_dict.keys()), relation_names


def normalize_features(x, mode):
    if mode == "none":
        return x
    if mode == "standard":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std
    if mode == "row":
        denom = x.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
        return x / denom
    raise ValueError(f"Unknown feature_norm: {mode}")


def parse_feature_columns(columns):
    if not columns:
        return None
    return [int(item.strip()) for item in columns.split(",") if item.strip()]


def load_feature_override(path, num_nodes, encoding="raw", columns=None):
    suffix = Path(path).suffix.lower()
    if suffix == ".pt":
        x = torch.load(path, map_location="cpu")
    elif suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        import numpy as np

        x = torch.from_numpy(np.loadtxt(path, delimiter=delimiter)).float()
    else:
        raise ValueError(f"Unsupported feature_path suffix: {suffix}")
    if columns is not None:
        x = x[:, columns]
    if encoding == "onehot":
        encoded = []
        for col in range(x.size(1)):
            values = x[:, col].long()
            _, inverse = torch.unique(values, sorted=True, return_inverse=True)
            encoded.append(torch.nn.functional.one_hot(inverse).float())
        x = torch.cat(encoded, dim=1)
    elif encoding != "raw":
        raise ValueError(f"Unsupported feature_encoding: {encoding}")
    x = x.float()
    if x.size(0) != num_nodes:
        raise ValueError(f"Feature override rows {x.size(0)} do not match node count {num_nodes}")
    return x


def degree_features(adj_dict, num_nodes):
    feats = []
    for relation in sorted(adj_dict.keys()):
        edge_index = adj_dict[relation]
        out_degree = torch.zeros(num_nodes, dtype=torch.float32)
        in_degree = torch.zeros(num_nodes, dtype=torch.float32)
        if edge_index.numel():
            row, col = edge_index
            out_degree.index_add_(0, row.cpu(), torch.ones(row.numel()))
            in_degree.index_add_(0, col.cpu(), torch.ones(col.numel()))
        feats.extend([torch.log1p(out_degree), torch.log1p(in_degree)])
    return torch.stack(feats, dim=1)


def make_train_test_split(y, train_ratio=0.7, seed=42, strategy="stratified"):
    labeled = (y >= 0).nonzero(as_tuple=False).view(-1)
    generator = torch.Generator().manual_seed(seed)
    if strategy == "random":
        labeled = labeled[torch.randperm(labeled.numel(), generator=generator)]
        n_train = max(1, int(round(labeled.numel() * train_ratio)))
        if labeled.numel() > 1:
            n_train = min(n_train, labeled.numel() - 1)
        train_idx = labeled[:n_train].long()
        test_idx = labeled[n_train:].long()
        val_idx = torch.empty(0, dtype=torch.long)
        return train_idx, val_idx, test_idx
    if strategy != "stratified":
        raise ValueError("split strategy must be 'stratified' or 'random'")
    train_parts = []
    test_parts = []
    for label in torch.unique(y[labeled]).tolist():
        idx = labeled[y[labeled] == label]
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        n_train = max(1, int(round(idx.numel() * train_ratio)))
        if idx.numel() > 1:
            n_train = min(n_train, idx.numel() - 1)
        train_parts.append(idx[:n_train])
        test_parts.append(idx[n_train:])
    train_idx = torch.cat(train_parts).long()
    test_idx = torch.cat(test_parts).long()
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=generator)]
    val_idx = torch.empty(0, dtype=torch.long)
    return train_idx, val_idx, test_idx


def load_link_splits(args):
    split_path = Path(args.link_split_path)
    if not split_path.exists():
        split_path = Path(args.mat_path)
    splits = load_link_pred_edges(str(split_path))
    return (
        splits["train"][0],
        splits["train"][1],
        splits["valid"][0],
        splits["valid"][1],
        splits["test"][0],
        splits["test"][1],
    )


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)

    data, relation_types, relation_names = load_graph(args)
    spaces = ["Euclidean", "Hyperbolic"]
    manifold_params = {
        "Euclidean": {"init_k": 1.0, "sigma": 1.0},
        "Hyperbolic": {"init_k": 1.0, "sigma": 1.0},
    }
    num_features = data["x"].size(1)
    num_classes = int(data["y"].max().item() + 1)

    print(f"Device: {device}")
    print(f"Task: {args.task}")
    print(f"Full graph mode: {args.use_full_graph}")
    print(f"Nodes: {data['x'].size(0)} | Features: {num_features} | Classes: {num_classes}")
    print(f"Relations: {len(relation_types)} ({', '.join(relation_names[:8])}{'...' if len(relation_names) > 8 else ''})")

    model = MRiemGNN(
        in_dim=num_features,
        hid_dim=args.hid_dim,
        out_dim=num_classes,
        relation_types=relation_types,
        spaces=spaces,
        num_layers=args.num_layers,
        manifold_params=manifold_params,
        num_kernels=args.num_kernels,
        dropout=args.dropout,
        kernel_learnable=args.kernel_learnable,
        learnable_beta=args.learnable_beta,
        beta_init=args.beta,
        hyperbolic_proj=args.hyperbolic_proj,
        input_skip=args.input_skip,
        input_skip_weight=args.input_skip_weight,
        layer_norm=args.layer_norm,
        relation_dropout=args.relation_dropout,
        node_relation_attention=args.node_relation_attention,
        node_relation_attention_mix=args.node_relation_attention_mix,
        behavior_dim=data.get("behavior_x").size(1) if args.behavior_projector else None,
        behavior_projector=args.behavior_projector,
        num_nodes=data["x"].size(0),
        node_embedding_dim=args.node_embedding_dim,
        shared_space_kernel=args.shared_space_kernel,
        classifier_hidden=args.classifier_hidden,
        feature_logit_weight=args.feature_logit_weight,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    link_args = (None, None, None, None, None, None)
    if args.task == "link_pred":
        link_args = load_link_splits(args)

    train(
        model,
        optimizer,
        data,
        relation_types,
        spaces,
        *link_args,
        epochs=args.epochs,
        device=device,
        task=args.task,
        patience=args.patience,
        lr_scheduler=args.lr_scheduler,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_patience=args.lr_decay_patience,
        min_lr=args.min_lr,
        lambda_mutual=args.lambda_mutual,
        mutual_target=args.mutual_target,
        tau=args.tau,
        beta=args.beta,
        class_weight=args.class_weight,
        class_weight_strength=args.class_weight_strength,
        log_every=args.log_every,
        merge_val_train=args.merge_val_train,
        monitor_split=args.monitor_split,
        monitor_metric=args.monitor_metric,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        lambda_behavior=args.lambda_behavior,
        behavior_tau=args.behavior_tau,
        eval_mode=args.eval_mode,
        logreg_epochs=args.logreg_epochs,
        eval_every=args.eval_every,
        logit_smoothing_steps=args.logit_smoothing_steps,
        logit_smoothing_alpha=args.logit_smoothing_alpha,
        logit_smoothing_use_train_labels=args.logit_smoothing_use_train_labels,
        logit_prior_alpha=args.logit_prior_alpha,
        train_logit_prior_alpha=args.train_logit_prior_alpha,
        logit_prior_search=args.logit_prior_search,
        logit_prior_grid_min=args.logit_prior_grid_min,
        logit_prior_grid_max=args.logit_prior_grid_max,
        logit_prior_grid_steps=args.logit_prior_grid_steps,
        prototype_logit_weight=args.prototype_logit_weight,
        prototype_temperature=args.prototype_temperature,
        prototype_metric=args.prototype_metric,
        save_logits_path=args.save_logits_path,
    )


if __name__ == "__main__":
    main()
