"""Author classes, isolated from their data loader and evaluation side effects.

The sole arithmetic adaptation is an equivalent scalar-output bilinear GEMM,
with optional node chunks/checkpointing inside the original discriminator.
"""
import ast
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

METHOD_DIR = Path(__file__).resolve().parents[1]


def _author_class(relative, name):
    source = METHOD_DIR / relative
    lock = json.loads((METHOD_DIR / "configs/author_sources.json").read_text())
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != lock[relative]:
        raise ValueError(f"Author source changed: {relative}")
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected exactly one author class {name}")
    namespace = {"__name__": __name__, "torch": torch, "nn": nn}
    namespace.update({key: globals()[key] for key in ("GCN", "InterDiscriminator") if key in globals()})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


GCN = _author_class("code/layers/gcn.py", "GCN")
InterDiscriminator = _author_class("code/layers/discriminator.py", "InterDiscriminator")
AuthorModel = _author_class("code/models/hdmi.py", "modeler")
LogReg = _author_class("code/models/logreg.py", "LogReg")


class MatrixBilinear(nn.Bilinear):
    def forward(self, left, right):
        if self.out_features != 1:
            raise ValueError("HDMI discriminators have scalar outputs")
        output = (torch.matmul(left, self.weight[0]) * right).sum(dim=-1, keepdim=True)
        return output if self.bias is None else output + self.bias


class ChunkedDiscriminator(InterDiscriminator):
    def _block(self, summary, positive, negative, features, corrupted):
        return tuple(logits.reshape(2, -1) for logits in
                     InterDiscriminator.forward(self, summary, positive, negative, features, corrupted))

    def forward(self, summary, positive, negative, features, corrupted):
        blocks = []
        for start in range(0, len(positive), self.chunk_size):
            stop = min(start + self.chunk_size, len(positive))
            args = (summary, positive[start:stop], negative[start:stop],
                    features[start:stop], corrupted[start:stop])
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                blocks.append(checkpoint(self._block, *args, use_reentrant=False))
            else:
                blocks.append(self._block(*args))
        return tuple(torch.cat([block[index] for block in blocks], dim=1) for index in range(3))


def make_model(input_dim, views, config, *, optimized=True):
    model = AuthorModel(input_dim, config["width"], views, config["same_discriminator"])
    if optimized:
        # Change methods without constructing new modules, consuming RNG, or
        # changing parameter/state_dict names. All author initialization survives.
        for module in model.modules():
            if isinstance(module, nn.Bilinear):
                module.__class__ = MatrixBilinear
            elif isinstance(module, InterDiscriminator):
                module.__class__ = ChunkedDiscriminator
                module.chunk_size = config["discriminator_chunk_size"]
                module.checkpoint_chunks = config["checkpoint_discriminators"]
    return model


def mutual_information_loss(outputs, config):
    relation_e, relation_i, relation_j, fusion_e, fusion_i, fusion_j = outputs
    terms = []
    for group in (relation_e, relation_i, relation_j):
        terms.append(sum(_binary_loss(logits) for logits in group))
    terms.extend(_binary_loss(logits) for logits in (fusion_e, fusion_i, fusion_j))
    weights = config["coef_layers"] + config["coef_fusion"]
    return sum(weight * term for weight, term in zip(weights, terms)), terms


def _binary_loss(logits):
    # Native BCE mean over all positive and negative nodes, including each
    # relation separately. Chunks do not change global readout or loss weights.
    labels = torch.stack((torch.ones_like(logits[0]), torch.zeros_like(logits[1])), dim=0)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)
