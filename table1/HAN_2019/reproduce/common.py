"""Method-local paths, fixed configurations and source provenance."""
from pathlib import Path
import json
import torch
from shared.coe_benchmark import load_dataset, check_reference_sources, read_registry
from shared.coe_benchmark.fingerprints import graph_fingerprint
from shared.coe_benchmark.protocol import sha256_file, json_digest
from reproduce.han_model import HAN, graph_from_adjacency

METHOD_DIR = Path(__file__).resolve().parents[1]
DATASETS = ("acm","dblp","yelp","amazon","mag")
SEARCH = METHOD_DIR / "configs/search_space.json"


def core_files():
    files = list((METHOD_DIR / "reproduce").glob("*.py"))
    files += list((METHOD_DIR / "configs").glob("*.json"))
    files += [METHOD_DIR / "environment/env_spec.json"]
    files += [METHOD_DIR / "ADAPTATION.md", METHOD_DIR / "REPRODUCE.md", METHOD_DIR / "sources.json",
              METHOD_DIR / "environment/README.md"]
    files += [METHOD_DIR / "code" / name for name in (
        "ex_acm3025.py","models/gat.py","models/base_gattn.py","utils/layers.py","utils/process.py")]
    return sorted(files)


def registered_config(path):
    value = json.loads(Path(path).read_text())
    config = value["config"] if "config" in value else value
    candidates = json.loads(SEARCH.read_text())["candidates"]
    if config not in candidates:
        raise ValueError("Config is not one of the pre-registered 12 complete candidates")
    return config


def data_and_graphs(dataset, device):
    if dataset not in DATASETS:
        raise ValueError("HAN primary matrix is the five registered Table1 datasets")
    data = load_dataset(dataset)
    graphs, supports = zip(*(graph_from_adjacency(adj, device) for adj in data.provided_adjs))
    features = tuple(view.to(device) for view in data.features)
    return data, features, tuple(graphs), tuple(supports)


def adaptation(data, supports):
    return {"id": "han_official_multirelation_sparse_v1", "kind": "input_interface_adaptation",
        "not_ccoe_model_mechanism": True,
        "all_relations_preserved": True, "view_count": len(supports),
        "relation_support_nnz": [support._nnz() for support in supports],
        "relation_support_fingerprint": graph_fingerprint(supports),
        "provided_graph_fingerprint": data.fingerprints["provided_graphs_sparse"],
        "support_rule": "positive provided edges, one self-loop; no symmetrization, edge weights unused as official bias support",
        "direction": "adjacency row=receiver,col=sender -> DGL src=col,dst=row",
        "semantic_attention": "official_HeteGAT_multi_per_node_softmax_over_all_relations",
        "width": 256, "heads": 8, "head_width": 32,
        "sparse_kernel": "DGL u_add_v / edge_softmax(dst) / u_mul_e_sum; no E*H*D message tensor",
        "determinism_limit": "DGL CUDA sparse reductions may use nondeterministic atomic reductions; fixed seeds do not imply bitwise training identity"}


def verify_current_sources(manifest):
    for name, expected in manifest["sources"]["files"].items():
        path = METHOD_DIR.parents[1] / name
        if sha256_file(path) != expected:
            raise ValueError(f"Current replay source differs from frozen snapshot: {name}")
    check_reference_sources(read_registry())


def reconstruct(manifest, device):
    verify_current_sources(manifest)
    data, features, graphs, supports = data_and_graphs(manifest["dataset"], device)
    if adaptation(data, supports) != manifest["adaptation"]:
        raise ValueError("Replay graph/feature adapter differs from training")
    model = HAN(data.input_dims, data.classes, manifest["config"]).to(device)
    return data, features, graphs, model
