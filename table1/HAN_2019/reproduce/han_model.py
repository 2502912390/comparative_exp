"""Sparse equivalent of Jhy1993/HAN HeteGAT_multi (official ACM entrypoint).

Each head is separate, including independent input/projected-feature dropout.
DGL fused SpMM does not materialize edge-by-head-by-hidden messages.
"""
import torch
from torch import nn
from torch.nn import functional as F
import dgl
from dgl.nn.functional import edge_softmax


def graph_from_adjacency(adjacency, device):
    """Binary support, one self-loop, row receiver / col sender; no symmetrizing."""
    adjacency = adjacency.to_sparse_coo().coalesce().cpu()
    indices = adjacency.indices()[:, adjacency.values() > 0]
    indices = indices[:, indices[0] != indices[1]]
    n = adjacency.shape[0]
    loops = torch.arange(n, dtype=torch.long)
    indices = torch.cat((indices, torch.stack((loops, loops))), dim=1)
    support = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), (n,n)).coalesce()
    row, col = support.indices()
    graph = dgl.graph((col.to(device), row.to(device)), num_nodes=n, device=device)
    return graph, support


class OfficialAttentionHead(nn.Module):
    def __init__(self, input_dim, head_dim, feature_dropout, attention_dropout):
        super().__init__()
        self.projection = nn.Linear(input_dim, head_dim, bias=False)
        self.receiver = nn.Linear(head_dim, 1, bias=True)
        self.sender = nn.Linear(head_dim, 1, bias=True)
        self.output_bias = nn.Parameter(torch.zeros(head_dim))
        self.feature_dropout = feature_dropout
        self.attention_dropout = attention_dropout
        for layer in (self.projection, self.receiver, self.sender):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def project(self, features):
        return self.projection(F.dropout(features, self.feature_dropout, self.training))

    def forward(self, graph, features):
        projected = self.project(features)
        # u=sender, v=receiver. TF f1_i + f2_j has exactly this orientation.
        logits = dgl.ops.u_add_v(graph, self.sender(projected), self.receiver(projected))
        coefficients = edge_softmax(graph, F.leaky_relu(logits, negative_slope=0.2), norm_by="dst")
        coefficients = F.dropout(coefficients, self.attention_dropout, self.training)
        messages = F.dropout(projected, self.feature_dropout, self.training)
        aggregated = dgl.ops.u_mul_e_sum(graph, messages, coefficients)
        return F.elu(aggregated + self.output_bias)

    def dense_forward(self, adjacency, features):
        """Independent small-graph oracle corresponding to official dense attn_head."""
        projected = self.project(features)
        logits = self.receiver(projected) + self.sender(projected).transpose(0,1)
        logits = F.leaky_relu(logits, negative_slope=0.2).masked_fill(~adjacency, -1e9)
        coefficients = F.dropout(torch.softmax(logits, dim=-1), self.attention_dropout, self.training)
        messages = F.dropout(projected, self.feature_dropout, self.training)
        return F.elu(coefficients @ messages + self.output_bias)


class OfficialSemanticAttention(nn.Module):
    """SimpleAttLayer: per-node softmax over all relations; no node averaging."""
    def __init__(self, width, attention_size):
        super().__init__()
        self.w_omega = nn.Parameter(torch.empty(width, attention_size))
        self.b_omega = nn.Parameter(torch.empty(attention_size))
        self.u_omega = nn.Parameter(torch.empty(attention_size))
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.1)

    def forward(self, relations):
        values = torch.stack(relations, dim=1)  # [N,V,D], as official multi_embed.
        v = torch.tanh(values @ self.w_omega + self.b_omega)
        scores = v @ self.u_omega
        weights = torch.softmax(scores, dim=1)
        return (values * weights.unsqueeze(-1)).sum(dim=1), weights


class HAN(nn.Module):
    def __init__(self, input_dims, classes, config):
        super().__init__()
        heads, width = int(config["heads"]), int(config["width"])
        if width % heads or width not in (64,256) or heads != 8:
            raise ValueError("Eight heads required; current primary default is8x8=64 (256 only for preserved failed-run diagnostics)")
        self.relations = nn.ModuleList([
            nn.ModuleList([OfficialAttentionHead(dim, width//heads,
                config["feature_dropout"], config["attention_dropout"]) for _ in range(heads)])
            for dim in input_dims])
        self.semantic = OfficialSemanticAttention(width, config["semantic_attention_size"])
        self.classifier = nn.Linear(width, classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features, graphs, *, dense=False, return_attention=False):
        if len(features) != len(self.relations) or len(graphs) != len(self.relations):
            raise ValueError("All registered relations/features must be supplied")
        relations = []
        for heads, features_v, graph_v in zip(self.relations, features, graphs):
            relations.append(torch.cat([(head.dense_forward(graph_v, features_v) if dense
                else head(graph_v, features_v)) for head in heads], dim=1))
        embedding, attention = self.semantic(relations)
        logits = self.classifier(embedding)
        return (logits, attention) if return_attention else logits


def official_l2_penalty(model, coefficient):
    # Official BaseGAttN exact-name exclusions do not match scoped variable names:
    # all trainable tensors, including biases, are regularized with tf.nn.l2_loss.
    return 0.5 * coefficient * sum(parameter.square().sum() for parameter in model.parameters())
