import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import SpectroRiemannianKernel
from manifolds.euclidean import EuclideanManifold
from manifolds.hyperboloid import HyperboloidManifold


class MRiemGNNLayer(nn.Module):
    """Relation-aware kernel message passing in multiple geometric spaces."""

    def __init__(
        self,
        in_dim,
        out_dim,
        relation_types,
        spaces,
        manifold_params,
        num_kernels=64,
        kernel_learnable=False,
        relation_dropout=0.0,
        shared_space_kernel=False,
        node_relation_attention=False,
        node_relation_attention_mix=1.0,
    ):
        super().__init__()
        self.relation_types = list(relation_types)
        self.spaces = list(spaces)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relation_dropout = relation_dropout
        self.shared_space_kernel = shared_space_kernel
        self.node_relation_attention = node_relation_attention
        self.node_relation_attention_mix = float(node_relation_attention_mix)

        self.manifolds = nn.ModuleDict(
            {
                "Euclidean": EuclideanManifold(),
                "Hyperbolic": HyperboloidManifold(),
            }
        )

        self.kernels = nn.ModuleDict()
        self.weight = nn.ParameterDict()
        self.curvature = nn.ParameterDict()
        for space in self.spaces:
            sigma = manifold_params.get(space, {}).get("sigma", 1.0)
            init_k = manifold_params.get(space, {}).get("init_k", 1.0)
            if self.shared_space_kernel:
                self.kernels[space] = SpectroRiemannianKernel(
                    in_dim, num_kernels=num_kernels, sigma=sigma, learnable=kernel_learnable
                )
            for relation in self.relation_types:
                key = f"{space}_{relation}"
                if not self.shared_space_kernel:
                    self.kernels[key] = SpectroRiemannianKernel(
                        in_dim, num_kernels=num_kernels, sigma=sigma, learnable=kernel_learnable
                    )
                self.weight[key] = nn.Parameter(torch.empty(num_kernels, out_dim))
                self.curvature[key] = nn.Parameter(torch.ones(1) * init_k)

        self.self_loop = nn.ParameterDict(
            {space: nn.Parameter(torch.empty(in_dim, out_dim)) for space in self.spaces}
        )
        self.bias = nn.ParameterDict(
            {space: nn.Parameter(torch.zeros(out_dim)) for space in self.spaces}
        )
        self.relation_weight = nn.ParameterDict(
            {space: nn.Parameter(torch.zeros(len(self.relation_types))) for space in self.spaces}
        )
        self.relation_scorer = nn.ModuleDict(
            {space: nn.Linear(out_dim, 1) for space in self.spaces}
        )
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.weight.values():
            nn.init.xavier_uniform_(param)
        for param in self.self_loop.values():
            nn.init.xavier_uniform_(param)
        for param in self.bias.values():
            nn.init.zeros_(param)
        for module in self.relation_scorer.values():
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, h_by_space, adj_dict):
        if torch.is_tensor(h_by_space):
            h_by_space = {space: h_by_space for space in self.spaces}

        out = {}
        for space in self.spaces:
            h = h_by_space[space]
            rel_out = []
            shared_kernel_features = self.kernels[space](h) if self.shared_space_kernel else None
            for relation in self.relation_types:
                edge_data = adj_dict[relation]
                degree = None
                edge_index = edge_data
                if isinstance(edge_index, tuple):
                    edge_index, degree = edge_index
                    edge_index = edge_index.to(h.device)
                    degree = degree.to(h.device, dtype=h.dtype)
                else:
                    edge_index = edge_index.to(h.device)
                if edge_index.numel() == 0:
                    rel_out.append(torch.zeros(h.size(0), self.out_dim, device=h.device, dtype=h.dtype))
                    continue
                row, col = edge_index
                kernel_features = shared_kernel_features
                if kernel_features is None:
                    kernel_features = self.kernels[f"{space}_{relation}"](h)
                messages = kernel_features[col] @ self.weight[f"{space}_{relation}"]
                agg = torch.zeros(h.size(0), self.out_dim, device=h.device, dtype=h.dtype)
                agg.index_add_(0, row, messages)
                if degree is None:
                    degree = torch.bincount(row, minlength=h.size(0)).clamp(min=1).to(h.dtype).unsqueeze(-1)
                rel_out.append(agg / degree)

            rel_logits = self.relation_weight[space]
            if self.training and self.relation_dropout > 0 and len(self.relation_types) > 1:
                keep = torch.rand_like(rel_logits) > self.relation_dropout
                if keep.any():
                    rel_logits = rel_logits.masked_fill(~keep, -1e9)
            rel_stack = torch.stack(rel_out, dim=0)
            global_alpha = F.softmax(rel_logits, dim=0)
            if self.node_relation_attention:
                scores = self.relation_scorer[space](rel_stack).squeeze(-1)
                scores = scores + rel_logits.view(-1, 1)
                node_alpha = F.softmax(scores, dim=0)
                mix = min(max(self.node_relation_attention_mix, 0.0), 1.0)
                alpha = (1.0 - mix) * global_alpha.view(-1, 1) + mix * node_alpha
                fused = (alpha.unsqueeze(-1) * rel_stack).sum(dim=0)
            else:
                fused = (global_alpha.view(-1, 1, 1) * rel_stack).sum(dim=0)
            out[space] = fused + h @ self.self_loop[space] + self.bias[space]

        return out
