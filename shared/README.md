# shared.coe_benchmark 公共基准模块

协议 `coe1005_fixedsplit_v1.1`，实现版本 `1.0.0`。数据身份协议仍为 `coe1005_fixedsplit_v1`：样本、标签、划分、特征和提供图的处理规则均未改变。v1.1 增加可执行公共 API、来源核验、输入适配、checkpoint 与结果门槛。

代码从 `conditional-joint-acm-tuning-1005` 提取纯数据逻辑，使用相对导入及每次调用独立的数据根目录。不导入或执行参考训练/模型代码，不读取参考权重，不借用 CCoE 学习图。原函数位置及允许的接口改动见 [provenance.json](coe_benchmark/provenance.json)。

## 运行和导入

使用现有 coe，不需要安装包或创建新环境：

```bash
cd /home/chenzhichao/Research/other_exp
/home/chenzhichao/miniconda3/envs/coe/bin/python -m shared.coe_benchmark audit --run-id p00_new_id --self-tests
```

务必换新的 run-id。检查产物自动写到 `shared/exp/audit_<run_id>/audit.json`、`shared/log/audit_<run_id>/console.log` 和 `unit_tests.log`。单元测试另保留各自的合成检查文件，以及一个 **两步 CPU 线性分类器 API fixture**，用于验证 checkpoint/导出/模型重放；该 fixture 是 smoke，不能进入正式汇总或论文表格，不是任一基线结果。

从作者方法目录启动时，把 **other_exp 根目录**放入 PYTHONPATH，保持名称空间，不添加参考目录或多个作者 code 目录：

```bash
PYTHONPATH=/home/chenzhichao/Research/other_exp /home/chenzhichao/miniconda3/envs/coe/bin/python reproduce/run.py --dataset acm --seed 1024 --device cuda:0 --phase formal --run-id formal_new_id --output-root /home/chenzhichao/Research/other_exp/table1/GCN_2017/exp
```

这是各方法今后实现的 CLI 约定；P00 不创建某方法模型或训练入口，也不会启动矩阵计划中的任务。

## 数据 API 与图适配

```python
from shared.coe_benchmark import (
    load_dataset, DataRoots, verify_identity,
    relation_union, graph_to_feature_context_v1, adaptation_metadata,
)
from pathlib import Path

data = load_dataset("mv_acm")
# 也可明确传入保持相同内容的只读数据根目录：
# data = load_dataset("mv_acm", roots=DataRoots(Path(coe_root), Path(mv_root)))

data.features               # tuple[V] float32 [N, d_v]
data.provided_adjs           # tuple[V] sparse COO [N, N]，全部关系保留
data.labels                 # int64 [N]，连续 class ID
data.class_mapping          # 原始类别 ID 与 encoded ID 的对应
data.masks                  # boolean train/val/test，固定、互斥且覆盖全部节点
data.indices("train")       # 全部外层训练节点
data.node_ids               # arange(N)，不可重排
data.to("cuda:0")           # 显式移动，不静默 CPU fallback
verify_identity(data)       # 重算 canonical source/split/label/mapping，篡改即报错

# GCN / NodeFormer / Mowst 的 Table1 单图适配：
# union = relation_union(data.provided_adjs)
# 模型自行执行其原生图算子；union 只做正边并集、二值化、单次单位自环。

# Table3 特征接口：
effective = graph_to_feature_context_v1(data)
metadata = adaptation_metadata(data, "graph_to_feature_context_v1")
```

`relation_union` 与 `graph_to_feature_context_v1` 是协议预先指定的 **无参数输入适配**，不是 1005 的可学习图机制。后者仅对 MV-ACM 的视图 1–4、MV-Citeseer 的视图 0 拼接 `X` 与 `row_l2(A²X)`；其他视图原样保留。适配后的维度分别是 `[1870,3740,3740,3740,3740]` 与 `[7406,3703]`。

图方向为行接收、列发送。`relation_union` 不额外强制对称化，不保留归一化边权，不选择关系。其作用是把全部提供关系编码成统一的单图支持；模型再执行自己的归一化。不要把并图的 hash 当成 canonical source hash。

历史 source 指纹包含图的原始 dense/sparse 序列化方式。公共图始终 sparse；重算历史 dense hash 时按行分块，不重建完整 NxN 图。`fingerprints["provided_graphs_sparse"]` 是公共稀疏存储 hash，`registry.reference_graph_fingerprint` 是 CCoE 模型图 hash；两者有不同用途，不要求相等。

读取数据先核验 7 份参考源码和配置哈希；身份不一致时抛出 `IdentityMismatch`，明确指出字段及 expected/actual。不存在跳过检查或放宽精度的参数。数据根目录不存在时明确报错，不偷偷回退到另一份数据。维度不同的视图不会截断，矩形关系列没有 verified node IDs 时不能补零。

## seed、选择与 checkpoint API

```python
from shared.coe_benchmark import (
    SEEDS, seed_everything, BestCheckpoint, load_best,
    ValidationSelector, write_selection_record, capture_rng, restore_rng,
)

seed_everything(1024)         # Python、NumPy、Torch/CUDA；cuDNN deterministic
# sparse/CUDA 算子不默认宣称逐位确定；严格模式 strict_algorithms=True 可显式启用。
selector = BestCheckpoint(job.exp / "best.pt", patience_checks=200)

# 由方法自己的训练器提供真实验证 Acc；只能在验证严格改善时保存。
selector.consider(epoch, validation_acc, {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "config_sha256": manifest["config_sha256"],
})
if selector.should_stop:
    pass                     # 训练器退出自己的循环

best = load_best(job.exp / "best.pt")
model.load_state_dict(best["payload"]["model_state"], strict=True)
# 测试必须使用此状态；模型重放也不重新训练或拟合分类头。
```

并列保留最早 epoch；非有限或非法 Acc、重复/倒退 epoch 会报错。现存 checkpoint 不由新的 selector 覆盖。`BestCheckpoint` 是 PyTorch 序列化实现；其他框架可使用 `ValidationSelector` 决策，并在写好单个完整推理状态文件后调用 `write_selection_record(path, selector)`。多文件框架需由方法适配层保存一个包/描述文件及全部组件校验值，确保最终状态文件能完整恢复；当前公共 exporter 对所登记的单个文件校验 SHA256，不自动管理 TF 多 shard。

`capture_rng/restore_rng` 用于原生随机推理机制的重放，不改变算法。需要 CUDA 的 RNG 状态不能在无 CUDA 的环境静默恢复。

## 单任务：产物 API

```python
from shared.coe_benchmark import (
    RunPaths, build_manifest, export_result, verify_model_replay, recount_result,
)

job = RunPaths.create(method_dir, "formal_unique_id", "acm", 1024, "formal")
manifest = build_manifest(
    data, job, method="GCN",
    config=resolved_config,
    source_files=all_method_core_files,  # 列出模型/损失/runner/配置等实际使用文件
    adaptation=adaptation_metadata(data, "relation_union"),
    parameter_count=trainable_parameters, device="cuda:0",
    author_commit=author_source_commit,
)

# 使用 best checkpoint，完成方法原生的推理；输出全部节点或完整测试节点。
result = export_result(
    job, data, checkpoint=job.exp / "best.pt",
    best_epoch=selector.best_epoch,
    best_validation_acc=selector.best_acc,
    node_ids=test_ids, predictions=test_predictions,
    scores=test_scores, score_kind="logits", timing=measured_timing,
)
# score_kind 可为 logits / probabilities / evidence；没有分数时不提供 scores。
# evidence 原样保存，不冒充 probability；任何 scores 的 argmax 必须等于预测。

# 方法自己的 replay.py 重新构造并加载模型，从 checkpoint 新推理：
# 2026-10-15 修订：以新推理的逐节点预测和 Acc 重放一致为门槛。
# 分数仍保存并检查有限性；微小 logits 差异另作诊断，不要求逐元素严格一致。
verify_model_replay(job.exp, node_ids=fresh_ids, predictions=fresh_predictions)
recount_result(job.exp)      # 仅复算已保存预测、数据身份、文件与源码快照哈希
```

`RunPaths` 固定产物到该方法 `exp/<run_id>/<dataset>/seed<seed>` 和 `log/<run_id>/<dataset>/seed<seed>`；run-id 不带 phase 时自动加前缀。训练器把 stdout/stderr 写到 `job.log/console.log`，把训练轨迹写到 `job.exp/training.jsonl`。公共 helper 不会改变作者模型的默认路径，适配器必须主动把剩余输出重定向到这两个目录。

`build_manifest` 自动记录协议、数据身份、配置 hash、共享源码及显式提供的方法源码 hash、Git 状态、环境与参数量，并保存核心源码快照。不得漏列模型/损失实现文件。checkpoint 和 selection sidecar 在导出时必须一致；正式结果不能覆盖既有的最终测试文件。

`predictions.npz` 保存完整 labels/masks/class mapping、导出的节点 ID、预测和原生分数语义，供独立重算历史指纹与 Acc。`result.json` 保存正确数/总数、选择 epoch、校验值和重放状态。

`recount` 不运行模型，不能替代 checkpoint replay。`verify_model_replay` 接收各方法 **从保存模型重新推理的**输出，比较预测及可选分数；不能直接把旧 predictions.npz 当成新推理输出。它仅提供比较/记录工具，模型重建与推理由各方法实现，公共代码无法替调用方证明推理来源。

### 浮点分数与重放（2026-10-15 修订）

后续方法默认保持作者原生精度，通常为 FP32，不为微小分数误差额外切换 FP64 或重跑训练。新进程必须恢复真实 checkpoint、输入/随机推理状态并重新推理；节点身份、逐节点类别预测、正确数/测试数及 Acc 保持一致。分数的形状、有限性和 argmax 语义仍须核对，微小 logits 差异单独记录，不作为停止队列的理由。

现有 `verify_model_replay` 在显式提供 `scores` 时仍使用严格逐元素比较。按本次修订，后续调用采用上面的预测比较接口，并将新旧分数的差异另记诊断日志；这一接口原已支持，不需要更改共享实现或模型。不得直接把历史 `predictions.npz` 作为新推理来源。正在运行的冻结批次保持原实现，避免修改共享源码触发来源不一致，也不回写历史 manifest 或混用不同批次。

## 矩阵：计划与各方法 CLI 约定

```bash
# 只生成计划；InfoMGF 为 Table1+2 的 9 数据集 × 5 seeds = 45 jobs：
/home/chenzhichao/miniconda3/envs/coe/bin/python -m shared.coe_benchmark plan --method InfoMGF --run-id formal_infomgf_new --devices cuda:0 cuda:1
# 需要冻结到该方法 exp/formal_infomgf_new/planned_jobs.json 时加 --save。
```

Python API 是 `plan_method(method, run_id, devices=..., phase="formal")` 与 `save_plan(plan)`。正式 seed 不可改为其他集合，dataset 范围来自方法登记，不能从成功任务反推实验范围。计划包含 argv 和日志/产物根目录约定，实际调度由各方法的 `reproduce/run_matrix.py` 执行；没有提供通用模型训练器。

方法入口统一：

| 入口 | 约定 |
|---|---|
| `reproduce/run.py` | `--dataset --seed --device --phase --run-id --output-root --config`，使用共同 API、方法原生训练 |
| `reproduce/run_matrix.py` | `--run-id --devices --phase --seeds`，使用 plan_method 范围；每次失败完整记录，同 seed 等价重跑 |
| `reproduce/replay.py` | `--job-dir --device`，验证快照/配置，重建全部推理状态，新推理后调用 verify_model_replay |
| `reproduce/summarize.py` | `--run-root --datasets`，调用 summarize_run/write_summary，不自行丢弃失败或低分种子 |

`--output-root` 在协议中固定为该方法的 exp；目前 RunPaths 以 method_dir 推导此位置，方法 runner 应拒绝不一致的 output-root，以免产物跑到共享或其他方法目录。

## 五种子汇总与门槛

```bash
/home/chenzhichao/miniconda3/envs/coe/bin/python -m shared.coe_benchmark summarize --run-root /home/chenzhichao/Research/other_exp/table1/InfoMGF_2024/exp/formal_infomgf_new --datasets acm dblp yelp amazon mag esp flickr iapr nus
/home/chenzhichao/miniconda3/envs/coe/bin/python -m shared.coe_benchmark recount --job-dir /path/to/method/exp/formal_run/acm/seed1024
```

`summarize_run(run_root, expected_datasets=...)` 或读取已冻结的 planned_jobs.json，逐个核验：

- 正式 phase、5 个唯一 seed、完整且预先登记的数据集范围。
- 数据、配置、来源源码、适配、环境、参数量和选择规则在同数据集五次一致。
- 文件/源码快照没有篡改，保存预测复算的 Acc 和正确数一致。
- 五个 seed 都有与当前 checkpoint/预测匹配的模型重放记录。

缺 seed 或未重放时标 incomplete，mean/std 为 null；不输出可填 LaTeX 行。发现重复 seed、混用数据/代码/配置或篡改文件时明确报错。五次 Acc 的百分比均值和 **样本 std，ddof=1** 写入 summary.json/CSV，完整行以两位小数写入 table_rows.tex。这是逐数据集统计，不自动改论文或与文献值混合。

## 核验范围

`audit --self-tests` 检查全部 14 个真实数据集的 canonical source/split/label/class mapping 指纹、稀疏提供图、并图和关系转特征的输入范围；独立比较 Wikipedia 测试观测多重集；核对 MV-ACM/Citeseer 适配维度。单元检查覆盖历史 dense hash 的稀疏分块重建、关系方向/自环、矩形关系拒绝补零、Wikipedia 重复行身份、名称空间隔离、RNG、验证 tie、patience、完整五种子 sample std、跨 Table 计划，以及真实数据上的两步 CPU checkpoint/导出/新模型重放与篡改拒绝。

检查过程中仅训练上述 smoke fixture 两步，不运行 CCoE、不读取其权重、不实施基线调参或 515 个正式任务。参考目录与 coe 环境保持不变。所有检查产物留在 shared/log、shared/exp，Git 仅提交共享核心代码、测试和说明。

