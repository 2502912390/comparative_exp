import torch
import torch.nn as nn

class EuclideanManifold(nn.Module):
    """
    欧氏空间：适用于MRiemGNN的流形接口
    """
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c
        self.name = 'Euclidean'

    def expmap(self, x, base=None):
        # 欧氏空间上的指数映射，退化为平移
        if base is None:
            return x
        return base + x

    def logmap(self, x, base=None):
        # 对应expmap的逆映射
        if base is None:
            return x
        return x - base

    def mobius_add(self, x, y):
        # 欧氏空间莫比乌斯加法=普通加法
        return x + y

    def distance(self, x, y):
        # 欧氏距离
        return torch.norm(x - y, dim=-1)

    def proj(self, x):
        # 投影到流形（欧氏空间恒等）
        return x

    def to_tangent(self, x, base=None):
        # 投影到切空间（欧氏空间恒等）
        return x

    def inner(self, x, y=None):
        # 欧氏空间的内积
        if y is None:
            y = x
        return torch.sum(x * y, dim=-1)

    def zero(self, shape, device=None, dtype=None):
        return torch.zeros(shape, device=device, dtype=dtype)
