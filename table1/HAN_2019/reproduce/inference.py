"""Full-graph FP64 evaluation, preserving formula and the shared1e-6 gate.

No logits rounding, no prediction-only replay, and no tolerance override.
Atomic CUDA reductions can differ at double-roundoff scale; this is numerical
stability, not a claim of bitwise deterministic CUDA inference/training.
"""
import copy
import torch


class StableInference:
    def __init__(self, model, features, graphs):
        self.model=copy.deepcopy(model).double().eval()
        self.model.requires_grad_(False)
        self.features=tuple(view.to(dtype=torch.float64) for view in features)
        self.graphs=graphs

    @torch.no_grad()
    def __call__(self, training_or_loaded_model):
        self.model.load_state_dict(training_or_loaded_model.state_dict(),strict=True)
        return self.model(self.features,self.graphs)
