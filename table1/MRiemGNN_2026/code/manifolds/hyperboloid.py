import torch
import torch.nn as nn

class HyperboloidManifold(nn.Module):
    """
    双曲空间（Lorentz模型）：适用于MRiemGNN的流形接口
    """
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c
        self.name = 'Hyperboloid'

    def minkowski_dot(self, x, y):
        # 洛伦兹内积（首分量取反，其余分量相乘）
        res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        return res

    def expmap(self, u, base):
        # 指数映射：切空间到流形
        norm_u = torch.norm(u[..., 1:], dim=-1, keepdim=True)
        theta = torch.sqrt(torch.clamp(self.c, min=1e-5)) * norm_u
        direction = u[..., 1:] / (norm_u + 1e-15)
        exp_0 = torch.cosh(theta)
        exp_space = torch.sinh(theta) * direction
        exp_map = torch.cat([exp_0, exp_space], dim=-1)
        return self.proj(exp_map + base)

    def logmap(self, x, base):
        # 对数映射：流形到切空间
        d = self.distance(x, base)
        direction = (x[..., 1:] - base[..., 1:]) / (torch.norm(x[..., 1:] - base[..., 1:], dim=-1, keepdim=True) + 1e-15)
        norm = torch.clamp(d / torch.sqrt(torch.clamp(self.c, min=1e-5)), min=1e-5)
        log_map = torch.cat([torch.zeros_like(norm), norm * direction], dim=-1)
        return log_map

    def proj(self, x):
        # 投影到洛伦兹流形
        x_ = x.clone()
        x_[..., 0] = torch.sqrt(torch.clamp(1.0 + torch.sum(x_[..., 1:] ** 2, dim=-1), min=1e-5))
        return x_

    def distance(self, x, y):
        # 流形上的测地线距离
        dot = -self.minkowski_dot(x, y)
        return torch.acosh(torch.clamp(dot, min=1.0))

    def mobius_add(self, x, y):
        # Hyperboloid下的加法（近似或用poincare近似）
        raise NotImplementedError("Mobius addition is not usually defined for Lorentz model directly.")

    def to_tangent(self, x, base=None):
        # 投影到切空间
        return x

    def inner(self, x, y=None):
        if y is None:
            y = x
        return self.minkowski_dot(x, y)

    def zero(self, shape, device=None, dtype=None):
        x = torch.zeros(shape, device=device, dtype=dtype)
        x[..., 0] = 1.0
        return x
