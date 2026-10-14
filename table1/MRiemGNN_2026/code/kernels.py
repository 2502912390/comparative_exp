import torch
import torch.nn as nn
import math

class SpectroRiemannianKernel(nn.Module):
    """
    Spectro-Riemannian Kernel Mapping Layer
    用于MRiemGNN的流形空间核特征投影
    """
    def __init__(self, in_dim, num_kernels=64, sigma=1.0, learnable=False):
        super().__init__()
        self.num_kernels = num_kernels
        self.omega = nn.Parameter(
            torch.randn(num_kernels, in_dim) * sigma, requires_grad=learnable)
        self.bias = nn.Parameter(
            torch.rand(num_kernels) * 2 * math.pi, requires_grad=learnable)

    def forward(self, h):
        # h: [N, in_dim]
        # φ(h) = sqrt(2/D) * cos(omega @ h^T + b)
        proj = torch.matmul(h, self.omega.t()) + self.bias
        return math.sqrt(2.0 / self.num_kernels) * torch.cos(proj)
