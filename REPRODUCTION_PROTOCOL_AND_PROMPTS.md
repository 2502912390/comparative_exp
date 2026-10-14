# 表 1–3 对比方法复现协议与逐方法 Prompt

基准代码：`/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005`。协议标识：`coe1005_fixedsplit_v1`；制定日期：2026-10-14。本文依据实际源码及正式五种子产物编写，没有运行新基线训练，也没有修改 `coe` 环境。

18 个论文方法共对应 103 个“方法 × 数据集”组合，正式评估合计 **515 次训练**；开发调参和冒烟检查另计。MvCGE 暂无已核实的作者源码，其 25 次属于计划量。共享方法只保留一个目录，一次适配全部对应 Table。

## 1. 已核实的基准数据处理

### 1.1 数据与划分

默认数据根目录为：

- `COE_DATA_ROOT=/home/chenzhichao/Research/CoE2/data/CoE_dataset`。
- `COE_MULTIVIEW_DATA_ROOT=/home/chenzhichao/Research/CoE2/data/Multi-view datasets`。

必须显式设置这两个位置，避免复制 loader 后其相对路径默认值改变。数据只读；不得重新下载同名但不同版本的数据替换当前文件。

| Table | CLI 数据集 ID | 样本数 | 类别 | 视图数 | 基准输入维度 | train / val / test |
|---|---|---:|---:|---:|---|---|
| 1 | acm | 3025 | 3 | 2 | 1902 / 1902 | 600 / 300 / 2125 |
| 1 | dblp | 2957 | 4 | 2 | 334 / 334 | 600 / 300 / 2057 |
| 1 | yelp | 2614 | 3 | 3 | 82 / 82 / 82 | 300 / 300 / 2014 |
| 1 | amazon | 11944 | 2 | 3 | 25 / 25 / 25 | 4777 / 2388 / 4779 |
| 1 | mag | 113919 | 4 | 2 | 128 / 128 | 40000 / 10000 / 63919 |
| 2 | esp | 11032 | 7 | 2 | 100 / 100 | 5514 / 2754 / 2764 |
| 2 | flickr | 12154 | 6 | 2 | 100 / 100 | 6076 / 3037 / 3041 |
| 2 | iapr | 7855 | 6 | 2 | 100 / 100 | 3926 / 1961 / 1968 |
| 2 | nus | 20000 | 8 | 2 | 100 / 100 | 10000 / 5000 / 5000 |
| 3 | mv_acm | 3025 | 3 | 5 | 1870 × 5 | 1511 / 755 / 759 |
| 3 | mv_citeseer | 3312 | 6 | 2 | 3703 / 3703 | 1655 / 827 / 830 |
| 3 | mv_prokaryotic | 551 | 4 | 3 | 393 / 3 / 438 | 274 / 136 / 141 |
| 3 | mv_reuters1200 | 1200 | 6 | 5 | 2000 × 5 | 600 / 300 / 300 |
| 3 | mv_wikipedia | 2866 | 10 | 2 | 128 / 10 | 1734 / 439 / 693 |

以上维度是 **1005 loader/prepare_data 之后的输入维度**，不一定等于 MAT 文件中每个原始矩阵的形状。具体样本身份、类别映射、数据和划分指纹见 [registry/reference_datasets.json](registry/reference_datasets.json)，已核对每个数据集五个正式种子的相关元数据一致；本轮还用当前 loader 在 CPU 上实际读取全部 14 个数据集，数据、划分、标签和类别映射指纹全部匹配。

源码依据：

- [data_loader.py:20](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/data_loader.py:20)：数据路径。
- [data_loader.py:72](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/data_loader.py:72)：表 1 读取保存的 masks/indices，**不按训练 seed 重新划分**；MAG 实际目录为 `mag-4`。
- [data_loader.py:211](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/data_loader.py:211)：表 2 读取 `L.mat/txt.mat/img.mat`，视图顺序 **文本、图像**，标签 `L−1`。ESP/Flickr/IAPR/NUS 分别来自 `esp-game/flickr30k/iaprtc12/nus-wide`。
- [data_loader.py:391](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/data_loader.py:391)：表 2 使用 `np.random.RandomState(42)`，逐类取 floor(50%) 训练、floor(25%) 验证、其余测试。
- [data_loader.py:424](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/data_loader.py:424)：表 3 常规划分使用 `np.random.default_rng(42)`，逐类约 50%/25%/25%，并保留代码的小样本保护规则。与表 2 **不是同一个 RNG**，不能用统一 sklearn split 替代。
- [data_loader.py:465](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/data_loader.py:465)：Wikipedia 通过全部原始视图及类别身份把 `Wikipedia-test.mat` 的 693 行映射回主文件，保留测试集；剩余样本按类约 80%/20% 划分 train/val。匹配只用于冻结样本身份，测试标签不得进入训练或调参。
- [data_loader.py:583](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/data_loader.py:583)：MAT 标签按 `np.unique(..., return_inverse=True)` 映射为连续 0-based ID，所有视图保留相同节点顺序。

### 1.2 特征、关系和构图：必须保留的区别

1. 表 1 使用每个关系图对应的同一份节点特征；Yelp/Amazon 在各自 loader 中做 row-L2。ACM/DBLP/MAG 沿用读取后的特征，**不能因为配置写了 row_l2 就对五个网络全部追加归一化**。
2. 表 2、常规特征视图使用 float32 + row-L2；这一步按行独立，不需要拟合全数据统计量。不得额外做全数据 StandardScaler/PCA，或换用 BERT/CLIP 等新特征。原方法确需统计变换时仅在训练数据拟合，单独披露。
3. 表 1 图预处理不完全相同：ACM/DBLP/Yelp 去旧自环、加自环、二值化后对称归一化；MAG 在稀疏图上加自环、二值化、对称归一化；Amazon 对现有稀疏图做对称归一化。应复用 loader 输出，不能把所有图粗略改为同一种稠密邻接。
4. 表 2 与 Prokaryotic/Reuters/Wikipedia 的 loader 返回空图，占位空图并不是已有邻域。1005 随后使用视图内特征构图。
5. **MV-ACM 的第 0 个矩阵是内容，1–4 是关系；MV-Citeseer 的第 1 个矩阵是内容，第 0 个是关系。** 1005 把关系视图转成图，并给这些视图复制内容特征；关系设置自环后按行归一化。只取 `JointData.features` 而丢弃图，会把这两个数据集变成重复的内容输入，使多数特征融合基线失去关系信息。
6. `prepare_data` 对有关系的视图产生 `row_l2(A^2 X)` 的无标签上下文；无关系视图使用 row-L2 特征。1005 多数任务再在上下文上学习视图内 kNN，MAG 直接用提供图。学到的边权和模型参数属于 **CCoE 自身机制**，不得把 CCoE checkpoint 生成的图或表示提供给其他方法。
7. 图方向约定为邻接行=接收节点、列=发送节点；转 PyG/DGL 时核对方向。`preserve_scale` 保留已有正自环，没有自环时以同尺度入边权补齐，然后构成行归一化 prior。不要误将已归一化图再次按同一规则归一化。
8. 1005 学习 max-input 对齐并使用 256 维表示，这是模型设计，**不是公共数据预处理**。不得把训练过的 CCoE 对齐器、编码器或可靠性头借给基线。

源码依据：[joint_data.py:69](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/joint_data.py:69)、[joint_data.py:141](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/joint_data.py:141)、[joint_graph.py:60](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/joint_graph.py:60)、[joint_graph.py:97](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/joint_graph.py:97)、[joint_graph.py:115](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/joint_graph.py:115)。

## 2. 统一公平比较规则

### 2.1 共同数据契约与方法适配

公共模块输出 `node_ids、class_mapping、labels、train/val/test masks、features[V]、provided_adjs[V]`。规范化特征和提供图的指纹、划分指纹、标签及类别映射哈希必须与登记值一致；若源码或数据版本改变，先核查差异，不得绕过检查。每个方法自己的学习图、单图转换及编码后的输入另存哈希，**不要求它们与 CCoE 的模型图指纹相同**。

共享 loader 应作为有名称空间的模块使用；不要随意向 `sys.path` 添加多个作者源码目录，以免同名 `utils.py/model.py/data.py` 被错误导入。公共模块从上述代码提取纯数据及无标签工具，记录源哈希；不得导入会解析 CLI、启动训练或拟合 CCoE 的入口。不得修改作为参照的 1005 目录。

默认适配规则须在开发阶段固定，不能对各测试集分别挑最好的一种：

- **原生多关系方法**：保留所有关系和节点特征，适配稀疏接口及原模型所需的图算子。
- **单图方法 GCN/NodeFormer/Mowst**：默认使用所有关系的无标签并图 `A_union[i,j]=1` 当至少一个原关系存在该非自环边；只加一次自环，再按原模型的图算子处理。三者共用同一并图策略，不按测试 Acc 挑关系。报告 `adaptation=relation_union`。
- **表 2 特征融合方法**：直接使用相同的文本/图像向量，保留其原生编码器、融合规则及损失。
- **表 3 只有特征接口的方法**：为 MV-ACM/Citeseer 的关系视图使用统一、无参数、无标签的 `graph_to_feature_context_v1`：`X_eff=concat(X, row_l2(A^2 X))`，其中 A 是公共 loader 的该视图提供图；纯内容视图与无图数据集仍用 X。使用稀疏乘法，并记录变换后维度。该适配使方法获得来自相同关系的结构信息；它是输入接口适配，并不等价于 CCoE 的可学习图编码。QMF/ECML/LMF/UVF/AssoDMVC/TUNED 均采用这一统一入口，不得只给其中部分方法增加关系信息。
- TUNED 等方法在上述入口上保留自己的邻域机制；不会获得 CCoE 的学习图。graph-to-feature 的规则本轮已预先指定，后续不得依据测试结果更换。
- 若希望严格隔离“融合模块”的增益，另设所有方法 **包括 CCoE** 共用同一从头训练的视图编码器的补充实验；其 CCoE 结果必须重新运行，不能沿用当前 1005 行。主表仍应披露各方法原生/适配编码器和图输入差别。

主比较统一信息来源、划分、标签预算和评估规则，保留各方法的关键机制。方法本身选择使用哪种图算子或表示是设计差异，不能把所有方法替换成同一个融合网络后仍称为原方法。

### 2.2 五次运行、验证选择与统计

- 固定 `split_seed=42`，正式训练 seed **1024、1025、1026、1027、1028**；表 1 原始 masks 直接冻结。五次随机初始化、dropout、训练采样及方法内随机性独立，**五次不换数据划分**。
- 训练使用全部外层训练标签；不得采用原始 CoE 旧流程的 fit/calibration 标签缩减，再把结果当成 1005 同协议。
- 所有监督损失、标签依赖的类关系、历史难度和原型只使用训练标签。无标签的全图特征与边可用于传导式学习，与 1005 保持一致。需要标签的邻域统计不能读取验证/测试标签。
- 验证集只用于配置和 checkpoint 选择：按 **验证 ACC 严格改善**保存 best；并列取最早 epoch。测试前恢复该 best checkpoint；不得每个 epoch 展示测试准确率，亦不得取最高测试 epoch。
- 监督模型默认上限 400 epochs，每 epoch 验证，patience=200 次验证不改善；以正式 1005 配置为依据。不同方法允许保留自身优化器、必要阶段与训练结构。无监督表示方法默认预训练上限 400 + 下游头上限 400，明确记录两个阶段的更新数和耗时，不能宣称它与单阶段模型有相同算力预算。开发验证显示未收敛时，应预先扩展并记录预算，不能根据测试分数延长训练。
- HDMI/DMG 的无监督表示训练不加入额外 CE 改成监督骨干；下游头只用外层训练标签。每个正式 seed 使用一个已固定评估流程，不保留官方脚本内部额外 10 次 probe 的平均，也不选择最好的 probe 测试结果。
- 主指标为单标签节点/样本分类 `ACC=correct/N_test`。F1 可作为附加指标；聚类 ACC、回归误差、交叉验证 fold 均值均不能填主表。
- 每个正式 seed 保存一个测试 ACC。统计百分比：`mean=mean(100*acc_seed)`，`std=std(100*acc_seed, ddof=1)`。不是总体标准差、标准误或跨 batch 标准差。
- 五个种子的 **源代码、数据与划分、最终配置、协议版本一致** 才能汇总。缺少 seed 或失败时状态为 incomplete，不输出“5 seeds”主表值；不得丢弃低分 seed。
- OOM/崩溃重试仍用同一 seed 和数学模型。优先稀疏化、等价分块和激活 checkpoint；不得偷偷降维、删节点、删关系、换小数据集。失败记录不删除，正式汇总只纳入该 seed 的一次有效重跑。
- 推理若含 MC dropout/随机采样（如 UVF），固定采样次数及独立评估 RNG，保存其状态，并能从 best checkpoint 复算同一 Acc。不得在测试上反复抽样挑分数。
- 设置 Python/NumPy/框架/CUDA seed；PyTorch 关闭 cuDNN benchmark 并启用 deterministic 设置。明确记录稀疏算子中不能完全保证确定性的部分，不把它们说成严格逐位确定。

源码依据：[joint_train.py:31](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/joint_train.py:31)、[joint_train.py:164](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/joint_train.py:164)、[joint_train.py:207](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/joint_train.py:207)、[summarize_joint.py:43](/home/chenzhichao/Research/CoE2/conditional-joint-acm-tuning-1005/summarize_joint.py:43)。

### 2.3 公平调参与模型容量

默认开发预算为每个“方法 × 数据集”**12 个预先登记的候选配置**（包含官方/论文合理起点），用 seed1024 的验证 Acc 选择；并列按预先登记顺序。开发运行与正式运行分开保存。选定后冻结配置，再从头运行全部五个正式 seed，不复用开发权重，不按每个 seed 的测试结果重新选参数。

学习率、weight decay、dropout、方法原生损失系数/秩/邻居数等允许适当搜索；不要直接复制 CCoE 的可靠性损失、对比损失或其全部超参数。主表表示宽度以 256 为首选，与 1005 保持衔接；原生机制存在多头、公共/私有分支、几何维度或固定宽度限制时保留其结构，记录有效维度、参数量和原因。不得用无用参数补齐参数量。

**12 个候选是今后基线的统一默认预算，不是已审计的历史 CCoE 搜索预算。** 当前 1005 已有按数据集调参；若要声称“严格相同 HPO 预算”，还须核查其历史 trial 数并让 CCoE 及基线共同遵循同一预算重选。没有该证据时只声称数据/划分/标签/评估一致，并披露调参预算。开发种子与正式种子重合属于当前预先指定的验证调参流程，也不能当作独立未参与选择的确认性检验。

### 2.4 环境规则

先用 `/home/chenzhichao/miniconda3/envs/coe/bin/python` 做必要 import 检查、小样本前向/反向、真实小数据集短训练、checkpoint 保存与重放检查。**依赖清单能 import，不等于方法已经能跑。**

当前只读探查：Python 3.10.21、torch 2.1.1+cu121、DGL 2.4.0+cu121、NumPy 1.26.4、SciPy 1.15.3、sklearn 1.5.2；主机 CUDA 可用，8 张 GPU。TensorFlow、PyG、torch_sparse、torch_scatter 尚不可 import。沙箱中 CUDA 不可见不算真实环境故障，须在同一主机可访问 GPU 的执行环境复核。记录见 [registry/environment_probe.json](registry/environment_probe.json)。

`coe` **保持只读，不在原环境 pip/conda install 或降级**。缺包或兼容性检查确实失败后才 clone：

```bash
/home/chenzhichao/miniconda3/bin/conda create -y -n coe_baseline_<method> --clone /home/chenzhichao/miniconda3/envs/coe
/home/chenzhichao/miniconda3/envs/coe_baseline_<method>/bin/python -m pip install <实际需要且兼容的包>
```

替换占位符后执行。记录触发 clone 的具体错误、环境名和包差异；安装前核对 Python/PyTorch/CUDA 的 wheel 兼容性。优先只安装方法实际需要的库，避免照搬包括无关音视频/LLM 任务的整份 requirements。框架迁移或公式等价实现也须披露，不能伪称未经改动的官方代码。

GCN/HAN 官方代码为旧 TensorFlow；先尝试导入并记录失败，再 clone coe，在副本尝试兼容层。若 TF1 API/算子无法兼容 Python3.10，可在副本调整 Python/框架，或提供逐模块对照的 PyTorch 等价移植；不强装不兼容 TF1 包。PyG 扩展版本须匹配实际 torch/CUDA。MCR/SynIB 优先提取向量特征任务需要的最小核心，避免为了未使用的预训练模型升级整套 torch。

### 2.5 目录、结果与 Git

每个方法唯一目录下统一使用：

```text
<method>/
  code/                     # 作者源码，根仓库追踪核心文件
  reproduce/                # 新的适配器、CLI、验证/重放/汇总代码
  configs/                  # 最终配置、开发搜索空间
  environment/              # 精简 requirements/环境说明，追踪
  ADAPTATION.md              # 原方法与适配版的差异，追踪
  sources.json              # 作者来源及原始提交，追踪
  log/<run_id>/<dataset>/seed1024/console.log
  exp/<run_id>/<dataset>/seed1024/
    best.pt                 # TF 可使用其 checkpoint 格式
    predictions.npz
    manifest.json
    result.json
    training.jsonl
    source_snapshot/
    environment.json
  exp/<run_id>/summary.json
  exp/<run_id>/summary.csv
  exp/<run_id>/table_rows.tex
```

开发运行使用 `dev_<run_id>`，冒烟使用 `smoke_<run_id>`，正式使用 `formal_<run_id>`，即使输入相同也不能覆盖旧目录。可统一提供 `run.py / run_matrix.py / replay.py / summarize.py`，CLI 包含 `--dataset、--seed、--device、--run-id、--phase、--output-root`。output-root 默认该方法的 exp；log 写到同方法的 log。框架产生的其他缓存、曲线、权重也必须重定向到该方法目录，不能写作者源码的默认 saved_model/pig 或工作目录外。方法内 canonical 数据缓存放 `exp/_cache/`，不会提交 Git。

`manifest.json` 至少记录 method、Table、dataset、seed、phase、完整配置、源码/配置 SHA256、root Git commit/dirty 状态、作者原始 commit、协议与适配 ID、数据/划分/类别身份、节点数/维度/类别数、图输入转换、参数量、环境版本、设备、各阶段预算/耗时和选择规则。

`predictions.npz` 至少保存测试 `node_ids、y_true、y_pred`、test-mask/indices、class mapping 与对应指纹；能够提供分类分数时保存 logits/probabilities 并说明语义，禁止把 Dirichlet evidence 直接当 logits。优先保存全部节点的预测及 masks，至少不能缺少测试节点身份。保存模型自身多视图预测可用于诊断；不支持时不伪造。`result.json` 保留正确数、测试数、ACC、best epoch、best validation ACC、checkpoint/预测文件哈希。

根仓库用 allowlist 只追踪核心源码、配置、依赖、许可证、必要说明与来源元数据；PDF、数据、图片、视频、日志、权重、缓存、运行快照不追踪。保留作者许可证。作者代码原有的 17 份 Git 元数据归档于忽略目录 `.upstream_git/<method>/`，源码作为普通文件纳入根仓库，避免根仓库仅记录 gitlink 而没有源码。原始作者 commit 仍由 `sources.json` 与 [registry/methods.json](registry/methods.json) 保存。

后续不要用 `git add -f` 强行加入产物，也不要再在 code/ 初始化嵌套仓库。提交核心修改前用 `git diff --cached --stat` 检查；方法运行产物保留在磁盘中。

### 2.6 填表前的完成条件

必须具备五个正式 seed、相同协议和冻结配置、验证选择 best、可重放 checkpoint、逐节点复算 Acc 一致、完整汇总及来源。完成后输出可供复制的 `mean ± sample_std` 表格行，保留完整精度 CSV，最终 LaTeX 显示两位小数。

尚未运行、缺种子、OOM 或缺源码保持空白并写状态，不填 0 或估计值。本地复现应替换相应文献参考单元格，并在填表时去掉该单元格/整行不再适用的 dagger；如果一行混合文献值与本地值，改为逐单元格来源标记，不能沿用整行 dagger 让来源混淆。本任务的 prompt 默认只生成候选表格行，**不自行改论文 PDF**。

CoE 已按要求不存放此库，但仍是论文中的前作基线。当前 CoE local 为单种子且其训练标签预算/冻结评估流程与 1005 不同，不能把它称为同协议五次结果；未来应在原始 CoE 工作区另行做明确适配的五种子评估，保留原生机制并解释流程差别。

## 3. 使用方式与优先顺序

先执行 P00 建立公共实现，然后逐个复制下面的完整代码块给新的复现会话。每个 prompt 指向本文件，公共约束因此不需要重复粘贴。新会话必须实际读文件，不得只看方法名。

建议先做数据契约及简单控制，再做 MMDynamics/ECML/LMF、GCN/HAN、HDMI/InfoMGF/DMG、其余方法。最后扩展到大图，不能把小数据集冒烟结果当成完整复现。

**后续 prompt 的范围是实现并运行指定方法，不是本轮已经完成了这些训练。** 使用当前会话模型；不启动子代理或切换付费模型，除非用户在对应复现会话明确授权。

## P00：公共协议实现（先执行一次）

```text
请先完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，并核对 registry/reference_datasets.json。
在 other_exp/shared 下实现有名称空间的 benchmark 数据适配、数据指纹检查、seed、验证 checkpoint 选择、结果导出和五种子统计工具；从 conditional-joint-acm-tuning-1005 提取必要纯数据逻辑，保持其 14 个数据集的样本身份、划分、类别映射和特征处理，不改参考目录，不调用其训练或权重。
落实协议指定的 relation_union 和 graph_to_feature_context_v1，公开标明它们是适配规则，不是 1005 的模型机制。公共接口同时支持 Table 1/2/3，保留稀疏图。
先核对当前参考源码 SHA256。验证全部 14 个数据集的 canonical source、split、label/class mapping 指纹，区分 CCoE 模型图哈希。测试 Wikipedia 固定测试集以及 MV-ACM/Citeseer 关系视图；不得重新划分、补矩形关系列或删视图。公共函数不得导入作者的同名 utils/model 模块。
用 coe 环境只做数据与必要功能检查，不跑 515 次训练。检查产物放 other_exp/shared/exp 与 shared/log；更新公共模块使用说明和协议版本，提交核心代码，不提交数据/产物。若当前数据与历史指纹不一致，具体定位并记录，不放宽断言。
之后给出可供各方法调用的 API 和单任务/矩阵/重放/汇总约定。使用当前模型，不启动子代理。
```

## P01：GCN（Table 1，25 次）

```text
请先完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，按全部公共约束复现 GCN。
唯一目录：/home/chenzhichao/Research/other_exp/table1/GCN_2017。
一次支持 acm dblp yelp amazon mag；正式 seeds=1024 1025 1026 1027 1028。
核对 sources.json 对应官方 tkipf/gcn；先读 code/gcn/models.py、layers.py、utils.py、train.py。保留两层 GCN、图归一化和监督目标，统一采用公共 relation_union，不能按测试分数选某个关系图。实现准确的 train/val/test mask 接口。
官方为 TF1：先记录 coe 导入/冒烟情况，失败才 clone coe_baseline_gcn。若选择 PyTorch 等价移植，在 ADAPTATION.md 对照图卷积公式、激活、dropout、正则项和初始化；不能把其他 GNN 名为 GCN。
依协议开发验证并冻结配置，完成五个数据集五种子，重放 best 及预测 Acc，输出各数据集 mean±sample std。MAG 用稀疏计算，不缩小图。log/exp 全部在该目录。只提交核心代码/配置，输出候选表格行，不改论文。使用当前模型，不启动子代理。
```

## P02：HAN（Table 1，25 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，按公共约束复现 HAN。
目录：/home/chenzhichao/Research/other_exp/table1/HAN_2019。
一次支持 acm dblp yelp amazon mag，seeds=1024–1028；不是作者自带 ACM3025/DBLP 的数据版本。
先读 code/ex_acm3025.py、models/gat.py、models/base_gattn.py、utils/layers.py，核对邻域节点注意力和跨关系语义注意力。用公共数据每个关系作为输入通道；不重建其他版本的元路径、节点或 masks，不丢关系。
官方为旧 TensorFlow。先测试 coe，确实失败再 clone coe_baseline_han；必要的 PyTorch 移植须与论文及官方机制逐项对应，保留多头和语义聚合，披露总输出维度/参数量。
避免官方稠密 NxN attention/bias 在 MAG 造成内存爆炸；等价稀疏实现须在小图上核对，不能用截断样本换取能跑。验证 Acc 选择，五次完整汇总和 checkpoint 重放。所有 log/exp 在本方法目录，提交核心实现和适配说明，不改论文。使用当前模型，不启动子代理。
```

## P03：HDMI（Table 1，25 次）

```text
请阅读并严格执行 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 HDMI。
目录：/home/chenzhichao/Research/other_exp/table1/HDMI_2021。
一次支持 acm dblp yelp amazon mag，正式 seeds=1024–1028。
先读 code/models/hdmi.py、layers/discriminator.py、embedder.py、evaluate.py。保留多关系编码、HDMI 高阶互信息和聚合机制，以公共图/特征替换自带 loader。无监督表示阶段不引入额外标签 CE。
重写下游评估为 train 标签拟合、val Acc 选择并保存线性分类头、最后一次 test。取消官方 evaluate 的内部额外多次 probe 平均及逐 epoch test 检查；每个 seed 保存表示骨干和选定头，重放不重新拟合。
先测试 coe，不兼容才 clone coe_baseline_hdmi。saved_model 等输出重定向到本方法 exp，MAG 稀疏化且保留全部节点关系。冻结配置后跑全 25 次，报告分类 ACC sample std，附适配差异。使用当前模型，不启动子代理。
```

## P04：NodeFormer（Table 1，25 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 NodeFormer。
目录：/home/chenzhichao/Research/other_exp/table1/NodeFormer_2022。
一次支持 acm dblp yelp amazon mag，seeds=1024–1028。
先读 code/nodeformer.py、main.py、main-batch.py、parse.py、data_utils.py，保留随机特征近似、Gumbel 图学习、关系/边正则和监督目标。对现有多关系采用公共 relation_union；明确其结构先验，不凭测试表现选择 use_graph 等开关。
coe 已缺 PyG/torch_sparse/torch_scatter，仍先验证实际需要的 import/短训练；失败后 clone coe_baseline_nodeformer，匹配 torch/CUDA 扩展。不要直接更换 NodeFormer 为其后续模型 SGFormer/DIFFormer。
保存随机特征和必要推理随机状态，确保 best checkpoint 可复算；大图使用官方可扩展路径或经验证的等价分块，不删 MAG 节点。验证集开发、冻结配置、完成 25 次及 summary。产物只写本目录 log/exp，提交核心代码。使用当前模型，不启动子代理。
```

## P05：DMG（Table 1，25 次）

```text
请阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，按公共协议复现 DMG。
目录：/home/chenzhichao/Research/other_exp/table1/DMG_2023。
一次支持 acm dblp yelp amazon mag，正式 seeds=1024–1028。
读 code/models/DMG.py、embedder.py、utils/process.py、evaluate.py、args.yaml。保留公共/私有表示、重构、独立性/对比目标及其交替优化；用现有关系和特征，不使用作者不同节点数的数据。
检查是否有固定双视图/数据集限定/稠密操作，并泛化到 2 或 3 个关系。按协议实现持久保存的 train-only 下游分类头和验证 Acc checkpoint 选择，取消官方额外十次 probe 平均和反复 test；输出节点分类 ACC，不把聚类 ACC 或 F1 填表。
先验证 coe，缺实际依赖再 clone coe_baseline_dmg；不要把旧 requirements 整份强装进 coe。冻结开发配置后完成 25 次，重放全部有效结果，产物在本目录 log/exp，提交核心源码和机制对照。使用当前模型，不启动子代理。
```

## P06：InfoMGF（Table 1 + 2，45 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，一次性适配 InfoMGF 的两个 Table。
唯一目录：/home/chenzhichao/Research/other_exp/table1/InfoMGF_2024，不再新建 table2 副本。
数据集：acm dblp yelp amazon mag esp flickr iapr nus，正式每个 seeds=1024–1028。
先读 code/InfoMGF/main.py、model.py、graph_learners.py、data_loader.py、params.py。保留特有/融合图学习、冗余控制、对比目标和原生下游 GCN；Table1 输入全部提供关系，Table2 使用公共文本/图像向量并从每个视图自身特征构图，不能把占位空图当真实图。
先核查官方主循环是否在多次评估下游头时查看或保存测试最好值；改为 train 拟合、val 选表示/图和分类头、最终冻结一次 test，保存整个推理链，重放不重新训练评估 GCN。Flickr 是当前 6 类版本。
coe 的 DGL 与官方老版本不同，先前向/反向及稀疏算子冒烟；有实质失败才 clone coe_baseline_infomgf。MAG 不生成稠密全图相似度，采用验证过的稀疏/分块等价实现；若数学机制无法保持则报告未完成，不删其核心图学习。
按协议开发并冻结每数据集配置，再完成 45 次，按 Table 分组导出 mean±std。log/exp 全部在唯一目录，提交核心代码和 ADAPTATION.md。使用当前模型，不启动子代理。
```

## P07：Mowst（Table 1，25 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 Mowst。
目录：/home/chenzhichao/Research/other_exp/table1/Mowst_2024。
一次支持 acm dblp yelp amazon mag，seeds=1024–1028。
读 code/model.py、train_test_engine.py、main.py。保留弱 MLP/强 GNN 专家及置信度 gate、方法要求的预训练/训练顺序；多关系使用公共 relation_union，不自行换成 CCoE 式多专家融合。
提前固定主表 variant（建议论文 Mowst-GCN 对应配置，名称写清 Mowst 而非混用 Mowst*）；如验证选择 variant，它必须计入统一开发预算并在所有正式种子冻结。关闭原有 ten-run、不同 split 循环与默认 wandb 远程上传。
先尝试 coe，PyG 实质缺失/不兼容后 clone coe_baseline_mowst；保留任务所需最小依赖。保存所有专家/gate 的 checkpoint，不能重放时重做专家预训练。完整 25 次、val 选 best、导出 Acc sample std。产物在本目录 log/exp，提交核心修改。使用当前模型，不启动子代理。
```

## P08：MvCGE（Table 1，25 次；源码待取得）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，推进 MvCGE 的 Table1 复现。
目录：/home/chenzhichao/Research/other_exp/table1/MvCGE_2025。
目标 acm dblp yelp amazon mag，seeds=1024–1028。本地目前没有已核实官方源码，先读 CODE_NOT_FOUND.md、paper.pdf/paper.txt、supplement.pdf/supplement.txt。
重新核实作者/会议主页是否公开源码，有则记录作者依据、URL 和 commit 并使用；不要联系作者或发送邮件，除非用户明确授权。没有源码时可按论文和补充材料实现，必须标注 paper_based_reimplementation，逐模块列出图感知路由、逐层协作专家、训练目标和不能确定的实现细节；不能用普通 MoE 或 CCoE 代替后仍称为官方 MvCGE。
先把具体实现和不确定点写入 ADAPTATION.md，并进行机制/数值检查；无法确定会改变算法的关键公式时报告缺口，不编造实现或数值。可实现部分继续推进。
统一公共数据，保留全部关系。先试 coe，失败才 clone coe_baseline_mvcge。按验证协议冻结配置并完成可行的 25 次；未完成的单元格留空。产物在本目录 log/exp，提交实现及来源。使用当前模型，不启动子代理。
```

## P09：MRiemGNN（Table 1，25 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 MRiemGNN。
目录：/home/chenzhichao/Research/other_exp/table1/MRiemGNN_2026。
一次支持 acm dblp yelp amazon mag，seeds=1024–1028。
读 code/data.py、train.py、model_mriemgnn.py、layer_mriemgnn.py、kernels.py、manifolds/ 和 config.yaml。保留关系感知核、欧氏/双曲空间及协同损失，只运行节点分类分支；不用 link prediction 分支或原文 OGB-MAG 120 类子集。
直接适配公共节点、图关系和 masks，禁用 split_data.py 重新划分。MAG 必须是 113919 节点/4 类，Amazon 为 11944/2 类，不运行 prepare_ogb_mag.py 换数据。
检查几何投影、曲率、参数优化和损失数值稳定性；不把几何机制简化成普通 GCN，也不在测试上挑 saved logits ensemble。先验证 coe，实际 PyG 依赖失败后 clone coe_baseline_mriemgnn。
开发验证、冻结配置、完成 25 次及重放；输出节点 ACC mean±sample std。产物在本目录 log/exp，追踪核心代码/配置/适配说明。使用当前模型，不启动子代理。
```

## P10：LMF（Table 2 + 3，45 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，一次性适配 LMF 的两个 Table。
唯一目录：/home/chenzhichao/Research/other_exp/table2/LMF_2018。
数据集 esp flickr iapr nus mv_acm mv_citeseer mv_prokaryotic mv_reuters1200 mv_wikipedia，seeds=1024–1028。
先读 code/model.py 的 LMF/SubNet/TextSubNet 与论文公式。官方 forward 固定 audio/video/text 三模态，当前是静态向量和 2/3/5 视图；使用每视图独立向量投影，将低秩因子乘积泛化到任意 V，保留追加常数项、低秩乘性交互及融合权重。不制造虚假音频模态，不把静态向量伪造为文本序列，不用拼接 MLP 冒充 LMF。
对 Table3 的关系视图使用协议 graph_to_feature_context_v1；Prokaryotic 的 393/3/438 维全部保留。替换原回归/情感任务 head 为 C 类分类 head 和 CE，披露静态特征/多视图适配。搜索 rank 等原生参数并冻结。
先尝试 coe，包括小 batch 的 BatchNorm 行为；有真实错误才 clone coe_baseline_lmf。按公共验证选择 best，完成 45 次、重放并按 Table 输出 Acc mean±sample std。log/exp 在唯一目录，提交核心代码。使用当前模型，不启动子代理。
```

## P11：MMDynamics（Table 2，20 次）

```text
请阅读并执行 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 MMDynamics。
目录：/home/chenzhichao/Research/other_exp/table2/MMDynamics_2022。
一次支持 esp flickr iapr nus，seeds=1024–1028。
读 code/model.py 的 MMDynamic、train_test.py、main.py。保留特征级信息量、逐模态置信度和融合损失，用公共文本/图像特征替换原多组学 CSV，支持当前两个视图；不得重新归一化或换 ROSMAP/BRCA。
官方循环反复打印测试指标且主要保存最终状态：必须增加独立验证集、val Acc best checkpoint、冻结后 test，修复保存/加载一致性。标签只进训练分支，推理通过 infer 路径，不用测试标签计算辅助置信度监督。
先使用 coe 做真实小数据集短训练及 checkpoint 重放，有错误才 clone coe_baseline_mmdynamics；不加载作者已有模型 checkpoint。冻结配置后完成 20 次并输出 mean±sample std。产物在本目录 log/exp，提交核心代码和适配说明。使用当前模型，不启动子代理。
```

## P12：QMF（Table 2 + 3，45 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，一次适配 QMF 的 Table2/3。
唯一目录：/home/chenzhichao/Research/other_exp/table2/QMF_2023。
数据集 esp flickr iapr nus mv_acm mv_citeseer mv_prokaryotic mv_reuters1200 mv_wikipedia，seeds=1024–1028。
读 code/text-image-classification/train_qmf.py 的 rank_loss/model_forward、src 中 latefusion 模型与 History；结合 RGBD 路径核对论文的置信度加权、逐模态监督和历史难度排序。不能用 TMC 或普通概率平均代替 QMF。
当前输入均为现有静态向量，替换原 BERT/ResNet 原始模态入口，保留 QMF 核心并披露 encoder 适配。把双模态质量权重/历史表泛化到 V=2/3/5；训练历史只更新 train 节点，val/test 不能更新 History。Table3 关系使用 graph_to_feature_context_v1。
先用 coe 测实际最小依赖；缺包/版本错误才 clone coe_baseline_qmf，不为了未使用的 raw-image/raw-text 任务强装全部库。验证选择、冻结配置、完整 45 次、重放并按 Table 汇总 sample std。产物在唯一目录 log/exp，提交核心代码。使用当前模型，不启动子代理。
```

## P13：ECML（Table 2 + 3，45 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，一次适配 ECML 的 Table2/3。
唯一目录：/home/chenzhichao/Research/other_exp/table2/ECML_2024。
数据集 esp flickr iapr nus mv_acm mv_citeseer mv_prokaryotic mv_reuters1200 mv_wikipedia，seeds=1024–1028。
官方仓库叫 RCML：读 code/model.py、loss_function.py、data.py、main.py，明确与 ECML 论文的对应，保留证据提取及冲突意见组合，不换成其他 evidence 平均方法。原生 num_views 接口应支持 2/3/5 个视图和不等输入维度，当前 Table3 关系用 graph_to_feature_context_v1。
禁止沿用作者 conflict 模式给主表添加人工冲突、换 split 或使用测试标签；主表固定 normal 当前数据。训练阶段保留原生证据损失，验证/测试取组合后的最终类别判断，保存 evidence/score 的语义并能复算 Acc。
先尝试 coe，实质失败才 clone coe_baseline_ecml。新增 val 选 best 的统一 runner，开发冻结后完成 45 次、重放并按 Table 汇总。产物在唯一目录 log/exp，只提交核心代码/配置与 ADAPTATION.md。使用当前模型，不启动子代理。
```

## P14：MCR（Table 2，20 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 MCR。
目录：/home/chenzhichao/Research/other_exp/table2/MCR_2025。
一次支持 esp flickr iapr nus，seeds=1024–1028。
优先读 code/models/MCR_Model_StandAlone.py、MCR_Models.py 与原生 Trainer 中 loss 路径，核对论文各正则项、系数和梯度边界。使用公共两模态静态向量的独立投影代替音视频/大预训练 encoder，保留 MCR 训练目标，不用 generic CE/InfoNCE 代替其机制。
另提供同一 backbone/fusion、关闭 MCR 正则的 base_fusion 对照，以明确训练目标增益；该对照额外 20 次，单独命名与汇总，不算入 MCR 的五次。监督对比项/样本关系只使用训练标签，完整记录 backbone、各损失和参数量。
先在 coe 做最小 MCR 前向反向检查；需要 pytorch_metric_learning 等才 clone coe_baseline_mcr 并安装最小依赖，关闭 wandb 外部上传。预先冻结 MCR variant/config，不依据 test 选版本。
完成 MCR 的 20 次及 base_fusion 对照，val 选择 checkpoint、test 最终评估和重放。产物在本目录 log/exp，提交核心代码并披露 feature-level adaptation。使用当前模型，不启动子代理。
```

## P15：SynIB（Table 2，20 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 SynIB。
目录：/home/chenzhichao/Research/other_exp/table2/SynIB_2026。
一次支持 esp flickr iapr nus，seeds=1024–1028。
优先读 code/src/synib/models/vlm/synib_mask_model.py 中 SynIB/FusionIBModel_Mask、README objective 和对应配置。用公共静态文本/图像向量建立 feature-level 入口，保留原生模态遮蔽、反事实前向、KL 置信度惩罚及其梯度/掩码机制，不把 masking-only 或普通 dropout 写成 SynIB。
正式主 variant 在验证开发阶段固定并写明；若采用 learned mask，完整保存其参数及训练状态，不能为了省事换成未登记的 random mask。推理使用原生完整模态预测，不用测试标签定义 synergy 子集再只对该子集报主 Acc。
另跑相同 backbone/fusion、正则关闭的 base_fusion 对照，额外 20 次单独汇总；若与 MCR 对照复用，必须代码、配置、seed、checkpoint 和协议逐项相同并记录来源，不能只因名称相同就复用。
先尝试 coe 核心路径，不为无关 VLM 强制升级 torch2.9.1。确有缺失/不兼容才 clone coe_baseline_synib；惰性化未使用依赖需披露，核心公式不改。
完成 20 次及匹配对照，val 选 best、冻结 test、重放与 sample std。产物在本目录 log/exp，提交核心代码。使用当前模型，不启动子代理。
```

## P16：TUNED（Table 3，25 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 TUNED。
目录：/home/chenzhichao/Research/other_exp/table3/TUNED_2025。
一次支持 mv_acm mv_citeseer mv_prokaryotic mv_reuters1200 mv_wikipedia，seeds=1024–1028；不跑 PIE 等原实验后移植分数。
读 code/train_script.py、data.py、loss_function.py、model/handwritten.py、scene.py、pie_normal.py，先核对不同数据集脚本的差异并登记通用变体。保留局部特征-邻域、共享/共识证据及 selective MRF 跨视图聚合；泛化到 V=2/3/5 与 393/3/438 等不等维。
统一采用公共 Table3 特征入口（关系视图 graph_to_feature_context_v1），再按 TUNED 自身无标签邻域规则处理。官方 build_graphs 按 batch 构图与全图 kNN 不是同一算法，必须核实论文的训练/推理定义并记录采用哪一种，不能未经说明把它换成 CCoE 学习图。若保留 batch 邻域，验证/测试 batch 组成与顺序必须预先冻结并可重放。
主表不人工制造冲突。检查原循环测试集评估和邻居数据组织，改为 val 选择 checkpoint、最后 test，邻域与证据不得读取 test 标签。
先尝试 coe，实质失败才 clone coe_baseline_tuned。冻结配置后完成 25 次，保存所有证据/邻域相关推理状态，重放并汇总 sample std。产物在本目录 log/exp，提交机制对照说明和核心代码。使用当前模型，不启动子代理。
```

## P17：AssoDMVC（Table 3，25 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 AssoDMVC。
目录：/home/chenzhichao/Research/other_exp/table3/AssoDMVC_2025。
一次支持 mv_acm mv_citeseer mv_prokaryotic mv_reuters1200 mv_wikipedia，seeds=1024–1028。
读 code/Code/Model.py、engine.py、layers.py、loss.py、dataset.py、train.py。保留视图关联、类关系及动态融合；修复 engine 中固定 view1/view2/view3 的 batch 组织，使用任意 V=2/3/5 并支持不同输入维度。Table3 关系视图共用 graph_to_feature_context_v1。
重点核查 sym_conditional_prob 及所有按类别/标签建立的关系，只用 train 样本及标签；禁止让 val/test 标签进入关联计算。官方 Learn 接口把 X_test/y_test 用于选择 best_acc：拆成 val checkpoint 选择与最终 test，不能保持该测试选择行为。
禁用 cross_validation.py 自行换 fold；用冻结公共 masks。先试 coe，需要额外库才 clone coe_baseline_assodmvc。每个 checkpoint 持久保存关联状态，重放不重新估计标签关系。
验证开发冻结后完成 25 次及 mean±sample std。产物在本目录 log/exp，提交泛化后的核心实现和适配差异。使用当前模型，不启动子代理。
```

## P18：UVF（Table 3，25 次）

```text
请完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，复现 UVF。
目录：/home/chenzhichao/Research/other_exp/table3/UVF_2026。
一次支持 mv_acm mv_citeseer mv_prokaryotic mv_reuters1200 mv_wikipedia，seeds=1024–1028。
读 code/model.py、layers.py、trainKL.py、dataset.py、utils.py，保留不确定性估计、强弱视图差异化特征选择与融合。trainKL.py 固定三视图、model.py 存在固定 dim=128：泛化任意 V=2/3/5 和原始不等维，将宽度配置化并检查所有相关层，不能裁剪 Prokaryotic 的特征。使用公共 graph_to_feature_context_v1。
官方主循环评估并选择 best_acc 且 epochs=10：建立统一 train/val/test runner，开发阶段检查收敛，按 val 保存 best；不得用原 test 选择或跨验证 folds 的最好分数填表。
MC dropout 是原生不确定性机制，不能为普通 eval() 方便将其关闭。固定 MC 次数、推理 RNG、batch 组成与顺序；不要反复抽样挑测试分数。删除/重定向 forward 中 ./pig 等输出到本方法 exp，保持核心数学机制。
先尝试 coe，实质失败才 clone coe_baseline_uvf。冻结配置后完成 25 次，重放 checkpoint 和预测、汇总 sample std。产物在本目录 log/exp，提交核心代码与适配说明。使用当前模型，不启动子代理。
```

## P19：三个简单控制（建议补齐，额外 210 次）

```text
请先完整阅读 /home/chenzhichao/Research/other_exp/REPRODUCTION_PROTOCOL_AND_PROMPTS.md，为 Best-view、Mean、Concat+MLP 实现统一的 controls/ 入口，同时支持全部 14 个数据集。
沿用公共数据、标签、split 和五个 seed。表1/含关系的表3 使用共同的从头训练视图编码器，并写明是否与 CCoE 编码器一致；不要加载 CCoE 权重。Best-view 的视图选择仅使用验证 Acc，Mean 平均各视图概率，Concat+MLP 拼接共同编码器输出后训练分类头。
预先固定独立视图头的训练/选择规则；Best-view/Mean 可共享同一组视图模型产物，必须登记来源而不是把共享结果计为独立重复。Concat+MLP 单独从头训练，披露参数量及训练预算。
按协议进行配置冻结、checkpoint 保存、最终 test 与逐节点复算，导出每控制每数据集五次 mean±sample std。210 是三控制×14×5 的名义评估数；共享 Best-view/Mean 视图训练后，实际训练任务数应按实现真实计数。所有产物在 other_exp/controls/log 与 controls/exp，核心代码由根仓库追踪。先试 coe，实质失败才 clone。使用当前模型，不启动子代理。
```

