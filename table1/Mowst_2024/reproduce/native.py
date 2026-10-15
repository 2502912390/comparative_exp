"""Load the author's classes and update functions without generic imports.

Copyright (c) Meta Platforms, Inc. and affiliates. Author code is MIT licensed
in ../code/LICENSE. The selected AST definitions are executed unmodified.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv

METHOD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_DIR.parents[1]
DATASETS = ("acm", "dblp", "yelp", "amazon", "mag")


def author_nodes(relative, names, namespace):
    path = METHOD_DIR / relative
    pinned = json.loads((METHOD_DIR / "configs/author_sources.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != pinned[relative]:
        raise ValueError(f"Author source changed: {relative}")
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError("Author definition set mismatch")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


_namespace = {"__name__": __name__, "torch": torch, "nn": nn, "F": F,
              "np": np, "Categorical": Categorical, "GCNConv": GCNConv}
author_nodes("code/model.py", ["GCN", "MLP", "GateMLP"], _namespace)
author_nodes("code/train_test_engine.py", ["compute_confidence", "train_vanilla",
    "train_mowst1_simple_gate", "train_mowst2_simple_gate"], _namespace)
GCN, MLP, GateMLP = (_namespace[name] for name in ("GCN", "MLP", "GateMLP"))
compute_confidence = _namespace["compute_confidence"]
train_vanilla = _namespace["train_vanilla"]
train_weak = _namespace["train_mowst1_simple_gate"]
train_strong = _namespace["train_mowst2_simple_gate"]


class Mowst(nn.Module):
    def __init__(self, input_dim, classes, config):
        super().__init__()
        self.args = SimpleNamespace(**config)
        self.weak = MLP(input_dim, config["model1_hidden_dim"], classes,
                        config["model1_num_layers"], config["dropout"], self.args)
        self.strong = GCN(input_dim, config["model2_hidden_dim"], classes,
                          config["model2_num_layers"], config["dropout"], self.args)
        self.gate = GateMLP(2, config["model1_hidden_dim"], 1,
                           config["model1_num_layers"], config["dropout"], self.args)

    def reset_parameters(self):
        self.weak.reset_parameters()
        self.strong.reset_parameters()
        self.gate.reset_parameters()

    def set_mixture_dropout(self):
        for model in (self.weak, self.strong, self.gate):
            model.dropout = self.args.dropout_gate

    @torch.no_grad()
    def inference_parts(self, data):
        self.eval()
        weak = self.weak(data)
        strong = self.strong(data)
        dispersion = torch.cat((compute_confidence(weak, "variance"),
                                compute_confidence(weak, "entropy")), dim=1)
        confidence = torch.sigmoid(self.gate(dispersion))
        return F.softmax(weak, dim=1), F.softmax(strong, dim=1), confidence


def graph_data(data, device):
    from shared.coe_benchmark import relation_union
    graph = relation_union(data.provided_adjs)
    # Author ToSparseTensor/SparseTensor uses fill_diag(1) in GCNConv. PyG's
    # built-in torch COO branch instead ADDS a loop even if one exists. Remove
    # the union's diagonal immediately before that native operator adds it back,
    # keeping exactly the same unit-loop graph and avoiding optional extensions.
    row, column = graph.indices()
    off_diagonal = row != column
    input_graph = torch.sparse_coo_tensor(graph.indices()[:, off_diagonal],
        graph.values()[off_diagonal], graph.shape).coalesce()
    return SimpleNamespace(x=data.features[0].to(device), mlp_x=data.features[0].to(device),
        adj_t=input_graph.to(device), union_edges=graph._nnz(), y=data.labels.to(device).view(-1, 1),
        train_mask=data.masks["train"].to(device), val_mask=data.masks["val"].to(device),
        test_mask=data.masks["test"].to(device), num_nodes=data.nodes,
        num_features=data.input_dims[0], num_classes=data.classes)
