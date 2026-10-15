"""Mechanism, gradient and canonical identity witnesses; no parameter selection."""
import copy
import torch
import torch.nn.functional as F
from shared.coe_benchmark import seed_everything
from table2.MMDynamics_2022.code.model import MMDynamic
from table2.MMDynamics_2022.reproduce.common import canonical_data, DATASETS


def check_mechanism(device):
    seed_everything(1024)
    # Unequal view widths test the native V=2 path rather than assuming omics V=3.
    native = MMDynamic([5, 7], [11], 3, dropout=0).to(device)
    formula = copy.deepcopy(native)
    x = [torch.randn(9, 5, device=device), torch.randn(9, 7, device=device)]
    labels = torch.arange(9, device=device) % 3
    actual_loss, actual_logits = native(x, labels)
    weighted, info_penalty, confidence_penalty = [], 0, 0
    for v in range(2):
        info = torch.sigmoid(formula.FeatureInforEncoder[v](x[v]))
        z = F.relu(formula.FeatureEncoder[v](x[v] * info))
        logits = formula.TCPClassifierLayer[v](z)
        confidence = formula.TCPConfidenceLayer[v](z)
        weighted.append(z * confidence)
        target = logits.softmax(-1).gather(1, labels[:, None]).flatten()
        # No detach or sigmoid is added to the author's raw confidence predictor.
        info_penalty = info_penalty + info.mean()
        confidence_penalty = confidence_penalty + F.mse_loss(confidence.flatten(), target) + F.cross_entropy(logits, labels)
    expected_logits = formula.MMClasifier(torch.cat(weighted, dim=1))
    expected_loss = F.cross_entropy(expected_logits, labels) + info_penalty + confidence_penalty
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    actual_loss.backward()
    expected_loss.backward()
    for (name, p), (other, q) in zip(native.named_parameters(), formula.named_parameters()):
        assert name == other and p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=1e-6)
    with torch.no_grad():
        native.eval()
        torch.testing.assert_close(native.infer(x), expected_logits)
        # Test labels never enter infer, including auxiliary confidence supervision.
        torch.testing.assert_close(native(x, label=None, infer=True), native.infer(x))
    return {"native_loss_and_logits": "passed", "all_parameter_gradients": "passed",
            "two_unequal_view_dimensions": "passed", "label_free_infer": "passed"}


def check_data():
    records = {}
    for dataset in DATASETS:
        data = canonical_data(dataset)
        assert data.table == 2 and len(data.features) == 2
        assert all(view.dtype == torch.float32 for view in data.features)
        assert all(graph._nnz() == 0 for graph in data.provided_adjs)
        records[dataset] = data.identity()
    return records
