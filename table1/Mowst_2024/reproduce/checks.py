"""Check defaults, directed graph orientation, native sampling and label scope."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from shared.coe_benchmark import seed_everything
from table1.Mowst_2024.reproduce.native import (Mowst, METHOD_DIR, author_nodes, train_weak, train_strong)
from table1.Mowst_2024.reproduce.inference import monte_carlo, initial_rng


def author_defaults(config):
    source = ast.parse((METHOD_DIR / "code/main.py").read_text())
    main = next(node for node in source.body if isinstance(node, ast.If))
    statements = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets):
            break
        statements.append(node)
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=statements, type_ignores=[]), "author_main_defaults", "exec"), namespace)
    defaults = vars(namespace["parser"].parse_args([]))
    explicit = {"variant", "parameter_policy", "method", "submethod", "original_data", "infer_method", "run",
                "validation_rng_seed_offset", "test_rng_seed_offset"}
    for key, value in config.items():
        if key not in explicit:
            assert value == defaults[key], (key, value, defaults[key])
    print("AUTHOR_NUMERIC_DEFAULTS_MATCH", flush=True)


def fixture(device):
    n = 17
    x = torch.randn(n, 7, device=device)
    rows = torch.arange(n, device=device)
    edges = torch.cat((torch.stack((rows, rows)), torch.stack((rows, (rows + 1) % n))), dim=1)
    adjacency = torch.sparse_coo_tensor(edges, torch.ones(2*n, device=device), (n,n)).coalesce()
    return SimpleNamespace(x=x, mlp_x=x, adj_t=adjacency,
        y=(rows % 3).view(-1, 1), train_mask=rows < 8,
        val_mask=(rows >= 8) & (rows < 12), test_mask=rows >= 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    config = json.loads((METHOD_DIR / "configs/default.json").read_text())
    author_defaults(config)
    seed_everything(18)
    graph = fixture(args.device)
    # Sparse adjacency row=receiver/column=sender matches edge_index's reverse.
    sparse_layer = GCNConv(7, 5, cached=False).to(args.device)
    edge_layer = GCNConv(7, 5, cached=False).to(args.device)
    edge_layer.load_state_dict(sparse_layer.state_dict())
    raw_row, raw_column = graph.adj_t.indices()
    off = raw_row != raw_column
    sparse_input = torch.sparse_coo_tensor(graph.adj_t.indices()[:, off], graph.adj_t.values()[off], graph.adj_t.shape).coalesce()
    a = sparse_layer(graph.x, sparse_input)
    b = edge_layer(graph.x, graph.adj_t.indices().flip(0))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    print("DIRECTED_UNION_PYG_ORIENTATION_MATCH", flush=True)
    graph.adj_t = sparse_input
    model = Mowst(7, 3, config).to(args.device)
    model.reset_parameters()
    ids = graph.val_mask.nonzero().view(-1)
    state = initial_rng(4781)
    output = monte_carlo(model, graph, ids, state, keep_predictions=True)
    weak, strong, confidence = model.inference_parts(graph)
    namespace = {"torch": torch, "np": np}
    author_nodes("code/train_test_engine.py", ["eval_for_multi"], namespace)
    class Evaluator:
        def eval(self, d):
            return (d["y_true"] == d["y_pred"]).double().mean().item()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(state)
        native = namespace["eval_for_multi"](model.args, confidence, args.device, weak, strong,
            Evaluator(), graph, {"train": ids, "valid": ids, "test": ids})
        native_after = torch.get_rng_state()
    assert abs(native[1] - output["acc"]) < 1e-14
    assert torch.equal(native_after, output["rng_after"])
    again = monte_carlo(model, graph, ids, state, keep_predictions=True)
    assert np.array_equal(again["predictions"], output["predictions"])
    assert np.array_equal(again["routes"], output["routes"])
    print("NATIVE_MC50_METRIC_AND_RNG_MATCH", flush=True)
    other = Mowst(7, 3, config).to(args.device)
    other.load_state_dict(model.state_dict())
    changed = copy.copy(graph)
    changed.y = graph.y.clone()
    changed.y[~graph.train_mask] = (changed.y[~graph.train_mask] + 1) % 3
    optimizer1 = torch.optim.AdamW(list(model.weak.parameters()) + list(model.gate.parameters()), lr=config["lr"])
    optimizer2 = torch.optim.AdamW(list(other.weak.parameters()) + list(other.gate.parameters()), lr=config["lr"])
    old_strong = {name: parameter.detach().clone() for name, parameter in model.strong.named_parameters()}
    old_gate = {name: parameter.detach().clone() for name, parameter in model.gate.named_parameters()}
    seed_everything(72)
    loss1, _, _ = train_weak(model.weak, model.strong, model.gate, graph, nn.NLLLoss(reduction="none"), optimizer1, model.args)
    seed_everything(72)
    loss2, _, _ = train_weak(other.weak, other.strong, other.gate, changed, nn.NLLLoss(reduction="none"), optimizer2, other.args)
    assert abs(loss1 - loss2) < 1e-7
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[name])
    assert all(torch.equal(parameter, old_strong[name]) for name, parameter in model.strong.named_parameters())
    assert any(not torch.equal(parameter, old_gate[name]) for name, parameter in model.gate.named_parameters())
    old_weak_gate = {name: p.detach().clone() for name, p in model.named_parameters() if not name.startswith("strong.")}
    optimizer = torch.optim.AdamW(model.strong.parameters(), lr=config["lr"])
    loss, _, _ = train_strong(model.weak, model.strong, model.gate, graph, nn.NLLLoss(reduction="none"), optimizer, model.args)
    assert np.isfinite(loss)
    assert all(torch.equal(p, old_weak_gate[name]) for name, p in model.named_parameters() if not name.startswith("strong."))
    print("NATIVE_ALTERNATING_UPDATES_AND_NONTRAIN_LABEL_ISOLATION_PASS", flush=True)
    print("MOWST_CHECKS_PASSED", json.dumps({"device": args.device,
        "torch": torch.__version__, "native_GCNConv": True, "no_hpo": True}), flush=True)


if __name__ == "__main__":
    main()
