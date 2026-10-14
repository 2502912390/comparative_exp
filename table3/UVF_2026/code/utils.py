import torch
import torch.nn.functional as F

def empirical_copula_transform(X):
    """
    将 Tensor 数据进行 Copula 变换，输出范围为 [0,1]
    输入: X [n_samples, n_features]
    输出: u [n_samples, n_features]
    """
    n, d = X.shape
    ranks = torch.argsort(torch.argsort(X, dim=0), dim=0)
    u = (ranks + 1).float() / (n + 1)
    return u


def chebyshev_distance(U):
    """
    Chebyshev 距离：max(|x_i - y_i|)
    输入: X, Y [n_samples, n_features]
    输出: [n_samples, n_neighbors]
    """
    n = U.shape[0]
    m = U.shape[0]
    X_exp = U.unsqueeze(1).expand(n, m, -1)
    Y_exp = U.unsqueeze(0).expand(n, m, -1)
    return torch.max(torch.abs(X_exp - Y_exp), dim=2)[0]


def copula_entropy_knn_torch(X, k=64):
    """
    使用 PyTorch 实现的 Copula Entropy 估计器
    输入: X [n_samples, n_features] - 原始样本张量
    返回: CE 值（标量）
    """
    u = empirical_copula_transform(X)
    n, d = u.shape
    k = (int)(X.size(0) / 2)


    # 计算所有样本间的 Chebyshev 距离
    dists = chebyshev_distance(u)  # [n, n]
    dists += torch.eye(n, device=u.device) * 1e6  # 排除自身距离

    # 找到每个样本的第 k 近邻距离
    kth_distances, _ = torch.kthvalue(dists, k, dim=1)
    kth_distances = kth_distances + 1e-10  # 防止 log(0)

    # Kozachenko-Leonenko 熵估计公式
    # entropy = -digamma(k) + digamma(n) + d * torch.mean(torch.log(kth_distances))
    entropy = -torch.digamma(torch.tensor(k)) + torch.digamma(torch.tensor(n)) + d * torch.mean(torch.log(kth_distances))

    # CE 是负的熵
    return -entropy

def decorrelation_loss(features):
    """
    输入: features [batch_size, feature_dim]
    输出: 去相关损失
    """
    # 特征中心化
    features = features - features.mean(dim=0, keepdim=True)
    # 协方差矩阵
    cov = (features.T @ features) / (features.shape[0] - 1)
    # 去掉对角线，只关注特征之间的相关性
    off_diag = cov - torch.diag(torch.diag(cov))
    # L2范数或平方和作为损失
    loss = torch.norm(off_diag, p='fro')
    return loss

def mse_constraint(x, x_gated):
    """
    MSE约束，鼓励门控后特征与原始特征保持接近
    :param x: 原始特征 [B, D]
    :param x_gated: 门控后特征 [B, D]
    :return: MSE损失
    """
    return F.mse_loss(x_gated, x, reduction='mean')

