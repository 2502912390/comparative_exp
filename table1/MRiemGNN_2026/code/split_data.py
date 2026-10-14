# import numpy as np
# import scipy.io
#
# # 路径请根据实际情况调整
# mat = scipy.io.loadmat('data/small_alibaba_1_10/small_alibaba_1_10.mat')
#
# # 1. 获取标签（假设label为一维或二维稠密数组，自动适配）
# labels = mat['label']
# if hasattr(labels, 'toarray'):  # 若是稀疏矩阵
#     labels = labels.toarray()
# labels = np.array(labels).reshape(-1)
# num_nodes = len(labels)
#
# # 2. 设置比例
# train_ratio = 0.5
# val_ratio = 0.25
# test_ratio = 0.25
#
# # 3. 按类别分层划分
# train_idx, val_idx, test_idx = [], [], []
#
# for c in np.unique(labels):
#     idx = np.where(labels == c)[0]
#     np.random.shuffle(idx)
#     n_train = int(len(idx) * train_ratio)
#     n_val = int(len(idx) * val_ratio)
#     # 保证每一类都有数据
#     n_test = len(idx) - n_train - n_val
#
#     train_idx.append(idx[:n_train])
#     val_idx.append(idx[n_train:n_train + n_val])
#     test_idx.append(idx[n_train + n_val:])
#
# train_idx = np.concatenate(train_idx)
# val_idx = np.concatenate(val_idx)
# test_idx = np.concatenate(test_idx)
#
# # 4. 打印信息
# print(f"Train: {train_idx.shape}, Val: {val_idx.shape}, Test: {test_idx.shape}")
#
# # 5. 保存到新文件
# mat['train_idx'] = train_idx.reshape(-1, 1)
# mat['valid_idx'] = val_idx.reshape(-1, 1)
# mat['test_idx'] = test_idx.reshape(-1, 1)
# scipy.io.savemat('data/small_alibaba_1_10/small_alibaba_1_10_split.mat', mat)

import numpy as np
import scipy.io

mat = scipy.io.loadmat('data/small_alibaba_1_10/small_alibaba_1_10.mat')

# 保证label长度等于节点数
features = mat['feature']
num_nodes = features.shape[0]

labels = mat['label']
if hasattr(labels, 'toarray'):  # 稀疏
    labels = labels.toarray()
labels = np.array(labels).reshape(-1)[:num_nodes]

# 分层划分
train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
train_idx, val_idx, test_idx = [], [], []

for c in np.unique(labels):
    idx = np.where(labels == c)[0]
    np.random.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * val_ratio)
    train_idx.append(idx[:n_train])
    val_idx.append(idx[n_train:n_train+n_val])
    test_idx.append(idx[n_train+n_val:])

train_idx = np.concatenate(train_idx)
val_idx = np.concatenate(val_idx)
test_idx = np.concatenate(test_idx)

print("max train_idx", train_idx.max())
print("num_nodes", num_nodes)

# 检查所有索引不越界
assert train_idx.max() < num_nodes
assert val_idx.max() < num_nodes
assert test_idx.max() < num_nodes

mat['train_idx'] = train_idx.reshape(-1, 1)
mat['valid_idx'] = val_idx.reshape(-1, 1)
mat['test_idx'] = test_idx.reshape(-1, 1)
scipy.io.savemat('data/small_alibaba_1_10/small_alibaba_1_10_split.mat', mat)

