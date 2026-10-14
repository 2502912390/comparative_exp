"""Numerical and mechanism checks against the actual pinned author classes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from shared.coe_benchmark import seed_everything
from table1.HDMI_2021.reproduce.model import make_model, mutual_information_loss, METHOD_DIR
from table1.HDMI_2021.reproduce.run import check_device


def flattened(outputs):
    return list(outputs[0]) + list(outputs[1]) + list(outputs[2]) + list(outputs[3:])


def compare(device, views, shared_discriminator):
    seed_everything(9281)
    config = json.loads((METHOD_DIR / "configs/smoke.json").read_text())
    config.update(width=7, discriminator_chunk_size=5, same_discriminator=shared_discriminator)
    native = make_model(6, views, config, optimized=False).to(device).double()
    adapted = make_model(6, views, config, optimized=True).to(device).double()
    adapted.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(11, 6, device=device, dtype=torch.float64)
    negative = x[torch.randperm(11, device=device)]
    adjacency = []
    for relation in range(views):
        dense = torch.eye(11, device=device, dtype=torch.float64)
        dense[torch.arange(11), (torch.arange(11) + relation + 1) % 11] = 0.4
        adjacency.append(dense.to_sparse_coo().coalesce())
    original = native(x, negative, adjacency, True)
    changed = adapted(x, negative, adjacency, True)
    for a, b in zip(flattened(original), flattened(changed)):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    loss_original, _ = mutual_information_loss(original, config)
    loss_changed, terms = mutual_information_loss(changed, config)
    loss_original.backward()
    loss_changed.backward()
    worst = 0.0
    for (name_a, a), (name_b, b) in zip(native.named_parameters(), adapted.named_parameters()):
        assert name_a == name_b
        assert a.grad is not None and b.grad is not None, name_a
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-9, msg=name_a)
        worst = max(worst, (a.grad - b.grad).abs().max().item())
    with torch.no_grad():
        torch.testing.assert_close(native.embed(x, adjacency, True), adapted.embed(x, adjacency, True))
        # Dense/sparse GCN direction/operator must coincide on directed input.
        for encoder, graph in zip(adapted.gcn_list, adjacency):
            dense = encoder(x.unsqueeze(0), graph.to_dense().unsqueeze(0), False)
            sparse = encoder(x, graph, True)
            torch.testing.assert_close(dense, sparse)
    print(f"NATIVE_EQUIVALENCE views={views} shared={shared_discriminator} six_terms={len(terms)} max_gradient_error={worst:.3e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    check_device(args.device)
    for views in (2, 3):
        for shared in (False, True):
            compare(args.device, views, shared)
    seed_everything(372)
    x = torch.randn(9, 6, device=args.device, requires_grad=True)
    (x.square().sum()).backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0
    print("HDMI_CHECKS_PASSED", json.dumps({"device": args.device,
        "name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU"}), flush=True)


if __name__ == "__main__":
    main()
