import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_mriemgnn import MRiemGNNLayer


class MRiemGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        relation_types,
        spaces,
        num_layers=2,
        manifold_params=None,
        num_kernels=64,
        dropout=0.5,
        kernel_learnable=False,
        learnable_beta=False,
        beta_init=0.6,
        input_skip=True,
        input_skip_weight=0.5,
        layer_norm=False,
        relation_dropout=0.0,
        node_relation_attention=False,
        node_relation_attention_mix=1.0,
        behavior_dim=None,
        behavior_projector=False,
        num_nodes=None,
        node_embedding_dim=0,
        shared_space_kernel=False,
        classifier_hidden=0,
        hyperbolic_proj=False,
        feature_logit_weight=0.0,
    ):
        super().__init__()
        self.spaces = list(spaces)
        self.dropout = dropout
        self.embedding_dim = hid_dim
        self.learnable_beta = learnable_beta
        self.input_skip = input_skip
        self.input_skip_weight = input_skip_weight
        self.hyperbolic_proj = hyperbolic_proj
        self.feature_logit_weight = feature_logit_weight
        self.node_embedding = None
        effective_in_dim = in_dim
        if node_embedding_dim > 0:
            if num_nodes is None:
                raise ValueError("num_nodes is required when node_embedding_dim > 0")
            self.node_embedding = nn.Embedding(num_nodes, node_embedding_dim)
            effective_in_dim += node_embedding_dim
        beta_init = min(max(beta_init, 1e-4), 1.0 - 1e-4)
        self.beta_logit = nn.Parameter(
            torch.logit(torch.tensor(float(beta_init))), requires_grad=learnable_beta
        )
        manifold_params = manifold_params or {
            "Euclidean": {"init_k": 1.0, "sigma": 1.0},
            "Hyperbolic": {"init_k": 1.0, "sigma": 1.0},
        }

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_id in range(num_layers):
            layer_in = effective_in_dim if layer_id == 0 else hid_dim
            self.layers.append(
                MRiemGNNLayer(
                    in_dim=layer_in,
                    out_dim=hid_dim,
                    relation_types=relation_types,
                    spaces=self.spaces,
                    manifold_params=manifold_params,
                    num_kernels=num_kernels,
                    kernel_learnable=kernel_learnable,
                    relation_dropout=relation_dropout,
                    shared_space_kernel=shared_space_kernel,
                    node_relation_attention=node_relation_attention,
                    node_relation_attention_mix=node_relation_attention_mix,
                )
            )
            self.norms.append(nn.LayerNorm(hid_dim) if layer_norm else nn.Identity())

        self.input_projection = nn.Linear(effective_in_dim, hid_dim)
        self.behavior_projector = None
        if behavior_projector:
            projector_in = behavior_dim or in_dim
            self.behavior_projector = nn.Sequential(
                nn.Linear(projector_in, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
            )
        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(hid_dim, classifier_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, out_dim),
            )
        else:
            self.classifier = nn.Linear(hid_dim, out_dim)
        self.input_classifier = nn.Linear(effective_in_dim, out_dim)
        self.link_decoder = nn.Sequential(
            nn.Linear(4 * hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.node_embedding is not None:
            nn.init.normal_(self.node_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        if self.behavior_projector is not None:
            for module in self.behavior_projector:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.input_classifier.weight)
        nn.init.zeros_(self.input_classifier.bias)
        for module in self.link_decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _resolve_beta(self, beta):
        if self.learnable_beta or beta is None:
            return torch.sigmoid(self.beta_logit)
        return beta

    def _project_space(self, space, h):
        if space != "Hyperbolic" or not self.hyperbolic_proj:
            return h
        norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        max_norm = 1.0 - 1e-5
        scale = torch.clamp(max_norm / norm, max=1.0)
        return h * scale

    def _augment_input(self, x):
        if self.node_embedding is None:
            return x
        node_ids = torch.arange(x.size(0), device=x.device)
        return torch.cat([x, self.node_embedding(node_ids)], dim=-1)

    def encode(self, x, adj_dict, beta=0.6):
        beta = self._resolve_beta(beta)
        x = self._augment_input(x)
        h_by_space = {space: x for space in self.spaces}
        residual = self.input_projection(x)
        for layer_id, layer in enumerate(self.layers):
            h_by_space = layer(h_by_space, adj_dict)
            h_by_space = {
                space: self._project_space(space, self.norms[layer_id](h))
                for space, h in h_by_space.items()
            }
            is_last = layer_id == len(self.layers) - 1
            if not is_last:
                h_by_space = {
                    space: F.dropout(F.relu(h), p=self.dropout, training=self.training)
                    for space, h in h_by_space.items()
                }

        z_euc = h_by_space.get("Euclidean", next(iter(h_by_space.values())))
        z_hyp = h_by_space.get("Hyperbolic", next(iter(h_by_space.values())))
        fused = beta * z_euc + (1.0 - beta) * z_hyp
        if self.input_skip:
            fused = fused + self.input_skip_weight * residual
        return {"emb_euc": z_euc, "emb_hyp": z_hyp, "fused": fused, "input_features": x}

    def forward(self, x, adj_dict, beta=0.6):
        out = self.encode(x, adj_dict, beta=beta)
        out["Euclidean"] = self.classifier(out["emb_euc"])
        out["Hyperbolic"] = self.classifier(out["emb_hyp"])
        logits = self.classifier(out["fused"])
        if self.feature_logit_weight > 0:
            logits = logits + self.feature_logit_weight * self.input_classifier(out["input_features"])
        out["logits"] = logits
        return out

    def link_logits(self, z, edges):
        src = z[edges[:, 0]]
        dst = z[edges[:, 1]]
        pair_features = torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)
        return self.link_decoder(pair_features).squeeze(-1)

    def behavior_embedding(self, behavior_x):
        if self.behavior_projector is None:
            return None
        return self.behavior_projector(behavior_x)
