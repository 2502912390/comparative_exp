"""Two graph convolutions; matches tkipf/gcn's actual TF1 operators."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def author_early_stop(validation_losses, epoch_index, early_stopping=10):
    """Official train.py's zero-based moving-validation-loss stop condition.

    Current loss is already appended. TF's scalar loss is float32, so keep
    that dtype for the literal preceding-window mean. This is independent
    of the benchmark's best-checkpoint choice on validation accuracy.
    """
    if len(validation_losses) != epoch_index + 1 or epoch_index < 0:
        raise ValueError('Validation history must contain exactly the completed epochs')
    losses = np.asarray(validation_losses, dtype=np.float32)
    if not np.isfinite(losses).all() or early_stopping != 10:
        raise ValueError('Use finite validation losses and official early_stopping=10')
    return bool(epoch_index > early_stopping and
                losses[-1] > np.mean(losses[-(early_stopping + 1):-1]))


def official_normalize(union):
    """D^-1/2 A^T D^-1/2, exactly utils.normalize_adj, with existing unit loops.

    Public relation_union has already added each loop once. No new self-loop,
    symmetrization or feature normalization is performed here.
    """
    graph = union.to_sparse_coo().coalesce()
    row, column = graph.indices()
    degree = torch.zeros(graph.shape[0], dtype=graph.dtype, device=graph.device)
    degree.scatter_add_(0, row, graph.values())
    inverse = degree.pow(-0.5)
    inverse[~torch.isfinite(inverse)] = 0
    return torch.sparse_coo_tensor(
        torch.stack((column, row)), graph.values() * inverse[row] * inverse[column],
        graph.shape, device=graph.device).coalesce()


class GCN(nn.Module):
    def __init__(self, input_dim, classes, hidden_dim=16, dropout=0.5):
        super().__init__()
        self.weight0 = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.weight1 = nn.Parameter(torch.empty(hidden_dim, classes))
        self.dropout = float(dropout)
        # gcn/inits.py glorot: uniform +/-sqrt(6/(fan_in+fan_out)).
        nn.init.xavier_uniform_(self.weight0, gain=1.0)
        nn.init.xavier_uniform_(self.weight1, gain=1.0)

    def forward(self, features, support):
        dropped = F.dropout(features, p=self.dropout, training=self.training)
        hidden = F.relu(torch.sparse.mm(support, dropped @ self.weight0))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return torch.sparse.mm(support, hidden @ self.weight1)

    def loss(self, logits, labels, train_mask, weight_decay):
        # masked mean CE equals official mask/mean(mask) then global mean.
        # TF nn.l2_loss includes 1/2; only first convolution is regularized.
        return F.cross_entropy(logits[train_mask], labels[train_mask]) + (
            0.5 * float(weight_decay) * self.weight0.square().sum())


class TF1Adam(torch.optim.Optimizer):
    """TF1 AdamOptimizer update, including epsilon before bias correction.

    PyTorch Adam's default epsilon placement is slightly different; use the
    official update instead of quietly replacing it with AdamW/global decay.
    """
    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise ValueError('GCN parameter gradients must be dense')
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
                state['step'] += 1
                state['exp_avg'].mul_(beta1).add_(gradient, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                rate = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                parameter.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(group['eps']), value=-rate)
        return loss
