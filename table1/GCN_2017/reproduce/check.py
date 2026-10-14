"""Small independent dense equations and full-data checks; no HPO/formal training."""
import argparse
from pathlib import Path
import json
import numpy as np
import scipy.sparse as sp
import torch
from shared.coe_benchmark import relation_union, seed_everything
from shared.coe_benchmark.io import write_json
from gcn_baseline.common import METHOD_DIR, configure_runtime, prepare, read_config
from gcn_baseline.model import GCN, TF1Adam, official_normalize, author_early_stop


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
    model = GCN(3, 2, 16, dropout=0).to(device)
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
    # Exact official boundary: zero-based epoch=10 cannot stop; epoch=11 can.
    histories = [np.asarray([1.0] * 10 + [2.0, 3.0], dtype=np.float32),
                 np.asarray([2.0] * 10 + [1.0, 1.0], dtype=np.float32)]
    for history in histories:
        for index in range(len(history)):
            prefix = history[:index+1]
            literal = index > 10 and prefix[-1] > np.mean(prefix[-11:-1])
            assert author_early_stop(prefix, index, 10) == literal
    assert not author_early_stop(histories[0][:11], 10, 10)
    assert author_early_stop(histories[0], 11, 10)
    assert not author_early_stop([1.0]*12, 11, 10)  # strict > retains equality
    return dict(status='passed', device=str(device), normalization='asymmetric scipy official expression',
                hidden_dim=16, sparse_dense_forward=True, sparse_dense_gradients=True,
                masked_ce_first_layer_half_l2=True, tf1_adam_numpy_reference=True,
                official_early_stop_boundary_and_mean=True,
                python_executable=__import__('sys').executable, torch_version=torch.__version__,
                gpu_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--data-check', action='store_true')
    args = parser.parse_args()
    configure_runtime(2)
    destination = METHOD_DIR / 'exp' / ('smoke_' + args.run_id) / 'equations.json'
    if destination.exists():
        raise FileExistsError(destination)
    report = equation_check(args.device)
    if args.data_check:
        data, features, support, labels, masks, metadata = prepare('acm', args.device)
        config = read_config(METHOD_DIR / 'configs/default.json')
        model = GCN(data.input_dims[0], data.classes, 16, .5).to(args.device)
        optimizer = TF1Adam(model.parameters(), lr=.01)
        model.train()
        loss = model.loss(model(features, support), labels, masks['train'], .0005)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(features, support)
            validation_loss = float(model.loss(logits, labels, masks['val'], .0005))
        report['canonical_acm_one_update'] = dict(nodes=data.nodes, edges=support._nnz(),
            identity=data.identity(), validation_loss=validation_loss, train_loss=float(loss.detach()),
            test_evaluated=False, config=config)
    write_json(destination, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
