import numpy as np
import scipy.io
import scipy.sparse

# # ----------载入原始数据Alibaba-smll ----------
# mat = scipy.io.loadmat('data/small_alibaba_1_10/small_alibaba_1_10.mat')
#
# # 1. 合并所有IUI_*非零边为正样本
# IUI_keys = [k for k in mat.keys() if k.startswith('IUI')]
# pos_edges = set()
# for k in IUI_keys:
#     adj = mat[k]
#     if scipy.sparse.issparse(adj):
#         coo = adj.tocoo()
#         for i, j in zip(coo.row, coo.col):
#             if i != j:   # 防止自环
#                 pos_edges.add((i, j))
#     else:
#         nz = np.transpose(np.nonzero(adj))
#         for i, j in nz:
#             if i != j:
#                 pos_edges.add((i, j))
# pos_edges = np.array(list(pos_edges))
#
# num_nodes = mat['feature'].shape[0]
#
# # 2. 根据节点划分边（train_idx/valid_idx/test_idx）
# def filter_edges(edges, node_idx):
#     node_set = set(node_idx.flatten())
#     mask = np.array([(i in node_set) and (j in node_set) for i, j in edges])
#     return edges[mask]
#
# train_idx = mat['train_idx'].flatten()
# valid_idx = mat['valid_idx'].flatten()
# test_idx = mat['test_idx'].flatten()
#
# train_pos = filter_edges(pos_edges, train_idx)
# valid_pos = filter_edges(pos_edges, valid_idx)
# test_pos = filter_edges(pos_edges, test_idx)
#
# # 3. 负采样：从所有可能节点对里随机采样，不重叠正例
# def sample_neg_edges(num_nodes, pos_set, num_samples):
#     all_edges = set((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)
#     neg_candidates = np.array(list(all_edges - pos_set))
#     if len(neg_candidates) < num_samples:
#         idx = np.random.choice(len(neg_candidates), num_samples, replace=True)
#     else:
#         idx = np.random.choice(len(neg_candidates), num_samples, replace=False)
#     return neg_candidates[idx]
#
# train_neg = sample_neg_edges(num_nodes, set(map(tuple, train_pos)), len(train_pos))
# valid_neg = sample_neg_edges(num_nodes, set(map(tuple, valid_pos)), len(valid_pos))
# test_neg = sample_neg_edges(num_nodes, set(map(tuple, test_pos)), len(test_pos))
#
# # 4. 保存
# mat['train_edges'] = train_pos
# mat['train_edges_neg'] = train_neg
# mat['valid_edges'] = valid_pos
# mat['valid_edges_neg'] = valid_neg
# mat['test_edges'] = test_pos
# mat['test_edges_neg'] = test_neg
#
# scipy.io.savemat('data/small_alibaba_1_10/small_alibaba_1_10_linkpred_split.mat', mat)
# print('lp_split hase been saved to small_alibaba_1_10_lp_split.mat')


# ----------载入原始数据IMDB----------
import scipy.io
import scipy.sparse
import numpy as np

mat = scipy.io.loadmat('data/IMDB/imdb_1_10.mat')
edges = mat['edges']

# 融合所有关系，得到一个全局邻接矩阵
adj = None
for i in range(edges.shape[1]):
    e = edges[0, i]
    if adj is None:
        adj = e.copy()
    else:
        adj = adj + e
adj.data[:] = 1
adj = adj - scipy.sparse.diags(adj.diagonal())  # 去自环

num_nodes = adj.shape[0]
row, col = adj.nonzero()
mask = row < col
pos_edges = np.vstack([row[mask], col[mask]]).T

# 随机划分正样本
np.random.shuffle(pos_edges)
n_total = len(pos_edges)
n_train = int(n_total * 0.5)
n_valid = int(n_total * 0.25)
train_edges = pos_edges[:n_train]
valid_edges = pos_edges[n_train:n_train + n_valid]
test_edges = pos_edges[n_train + n_valid:]

# 负样本（全0区间采样，同样数量）
all_pairs = set((i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes))
pos_set = set(map(tuple, pos_edges))
neg_candidates = np.array(list(all_pairs - pos_set))
np.random.shuffle(neg_candidates)
train_edges_neg = neg_candidates[:n_train]
valid_edges_neg = neg_candidates[n_train:n_train + n_valid]
test_edges_neg = neg_candidates[n_train + n_valid:n_train + n_valid + len(test_edges)]

print("train_edges.shape:", train_edges.shape)
print("train_edges_neg.shape:", train_edges_neg.shape)
print("valid_edges.shape:", valid_edges.shape)
print("valid_edges_neg.shape:", valid_edges_neg.shape)
print("test_edges.shape:", test_edges.shape)
print("test_edges_neg.shape:", test_edges_neg.shape)

# 保存到mat文件
scipy.io.savemat("data/IMDB/imdb_1_10_linkpred_split.mat", {
    "train_edges": train_edges, "train_edges_neg": train_edges_neg,
    "valid_edges": valid_edges, "valid_edges_neg": valid_edges_neg,
    "test_edges": test_edges,   "test_edges_neg": test_edges_neg
})

