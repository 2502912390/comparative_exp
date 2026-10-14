import scipy.io

# 这里改成你的实际路径
mat_path = 'data/small_alibaba_1_10/small_alibaba_1_10_linkpred_split.mat'

mat = scipy.io.loadmat(mat_path)

print("All fields in the .mat file:")
for k in mat.keys():
    print("  ", k)

# 重点关注 train_edges、train_edges_neg、valid_edges、valid_edges_neg、test_edges、test_edges_neg
for split in ["train", "valid", "test"]:
    pos_name = f"{split}_edges"
    neg_name = f"{split}_edges_neg"
    pos = mat.get(pos_name, None)
    neg = mat.get(neg_name, None)
    print(f"{pos_name}: type={type(pos)}, shape={None if pos is None else pos.shape}")
    print(f"{neg_name}: type={type(neg)}, shape={None if neg is None else neg.shape}")
    # 打印前5行看看
    if pos is not None:
        print(f"  {pos_name} head:", pos[:5])
    if neg is not None:
        print(f"  {neg_name} head:", neg[:5])
