"""Small independent dense equations and full-data checks; no HPO/formal training."""
import argparse
from pathlib import Path
import json
import numpy as np
import scipy.sparse as sp
import torch
from shared.coe_benchmark import relation_union, seed_everything
from shared.coe_benchmark.io import write_json
from gcn_baseline.common import METHOD_DIR, configure_runtime
from gcn_baseline.model import GCN, TF1Adam, official_normalize


def equation_check(device):
    seed_everything(1024)
    # Directed case is essential: official normalization transposes A.
    raw = torch.tensor([[1.,1.,0.,0.], [0.,1.,1.,1.], [1.,0.,1.,0.], [0.,0.,1.,1.]])
    union = relation_union([raw.to_sparse()])
    actual = official_normalize(union).to_dense().numpy()
    adjacency = sp.coo_matrix(raw.numpy())
    diagonal = sp.diags(np.power(np.asarray(adjacency.sum(1)).ravel(), -0.5))
    expected = adjacency.dot(diagonal).transpose().dot(diagonal).toarray()
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
    support = official_normalize(union).to(device).to_sparse_csr()
    dense = torch.tensor(expected, device=device)
    features = torch.randn(4, 3, device=device)
    model = GCN(3, 2, 5, dropout=0).to(device)
    model.train()
    output = model(features, support)
    reference = dense @ (torch.relu(dense @ (features @ model.weight0)) @ model.weight1)
    torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)
    gradient = torch.autograd.grad(output.square().sum(), tuple(model.parameters()), retain_graph=True)
    expected_gradient = torch.autograd.grad(reference.square().sum(), tuple(model.parameters()))
    for left, right in zip(gradient, expected_gradient):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    labels = torch.tensor([0,1,0,1], device=device)
    mask = torch.tensor([True,False,True,False], device=device)
    loss = model.loss(output, labels, mask, 0.0005)
    ce = torch.nn.functional.cross_entropy(output, labels, reduction='none')
    official = (ce * mask / mask.float().mean()).mean() + 0.0005 * 0.5 * model.weight0.square().sum()
    torch.testing.assert_close(loss, official)
    # Explicit TF1 Adam reference including near-zero gradients/epsilon.
    parameter = torch.nn.Parameter(torch.tensor([1.,2.], dtype=torch.float64))
    optimizer = TF1Adam([parameter], lr=0.01)
    values = parameter.detach().numpy().copy()
    m, v = np.zeros(2), np.zeros(2)
    for step in range(1, 4):
        grad = np.array([0.2 * step, 1e-10])
        parameter.grad = torch.tensor(grad, dtype=torch.float64)
        optimizer.step()
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        rate = 0.01 * np.sqrt(1-0.999**step)/(1-0.9**step)
        values -= rate * m/(np.sqrt(v)+1e-8)
    np.testing.assert_allclose(parameter.detach().numpy(), values, atol=1e-14, rtol=0)
    return dict(status='passed', device=str(device), normalization='asymmetric scipy official expression',
                sparse_dense_forward=True, sparse_dense_gradients=True,
                masked_ce_first_layer_half_l2=True, tf1_adam_numpy_reference=True,
                gpu_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    configure_runtime(2)
    destination = METHOD_DIR / 'exp' / ('smoke_' + args.run_id) / 'equations.json'
    if destination.exists():
        raise FileExistsError(destination)
    report = equation_check(args.device)
    write_json(destination, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
