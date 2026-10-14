
import torch
import torch.nn as nn

class LearnModel(nn.Module):
    def __init__(self, input_size, output_dim=256):
        super(LearnModel, self).__init__()
        self.input_sizes = list(input_size)
        self.dim = output_dim
        self.view_names = [f"view{i + 1}" for i in range(len(self.input_sizes))]
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, self.dim) for in_dim in self.input_sizes
        ])

    def forward(self, x):
        missing_views = [name for name in self.view_names if name not in x]
        if missing_views:
            raise KeyError(f"Missing input views: {missing_views}")

        return [
            projection(x[name])
            for name, projection in zip(self.view_names, self.projections)
        ]


class DifferentiatedFeatureSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(DifferentiatedFeatureSelector, self).__init__()
        self.device = torch.device('cuda')

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate_net(x)
        x_selected = gate * x
        return x_selected, gate


class MultiViewFeatureSelector(nn.Module):
    def __init__(self, input_dim=128, num_views=5):
        super(MultiViewFeatureSelector, self).__init__()
        self.device = torch.device('cuda')
        self.num_views = num_views
        self.selectors = nn.ModuleList([
            DifferentiatedFeatureSelector(input_dim=input_dim) for _ in range(num_views)
        ])
        self.fusion_layer = nn.Linear(input_dim * num_views, input_dim)  # 可根据任务调整

    def forward(self, views, strengths):
        selected_features = []
        gate_masks = []
        sorted_indices = torch.argsort(strengths)
        tmp = strengths[sorted_indices[0]]
        for i in range(self.num_views):
            if strengths[i] <= tmp:
                z, g = self.selectors[i](views[i])
                selected_features.append(z)
                gate_masks.append(g)
            else:
                selected_features.append(views[i])
                ones_mask = torch.ones_like(views[i])
                gate_masks.append(ones_mask)

        return selected_features, gate_masks

class MultiViewClass(nn.Module):
    def __init__(self, label_nums, input_dim=128):
        super(MultiViewClass, self).__init__()
        self.cls = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, label_nums)
        )

    def forward(self, x):
        out = self.cls(x)
        return out


class MultiViewClass_Droupout(nn.Module):
    def __init__(self, label_nums, input_dim=128):
        super(MultiViewClass_Droupout, self).__init__()
        self.cls = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(input_dim, label_nums),
        )

    def forward(self, x):
        out = self.cls(x)
        return out





