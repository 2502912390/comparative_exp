# P00 公共基准模块核验记录

- 协议：coe1005_fixedsplit_v1.1；实现：shared.coe_benchmark 1.0.0。数据身份协议仍为 coe1005_fixedsplit_v1。
- 执行者：当前 Codex 会话；未使用子代理或其他模型。
- 环境：已有 coe；未安装/升级包、未创建 conda 环境。数据与功能检查在 CPU 上执行，功能测试进程隐藏 CUDA 设备，不占用训练 GPU。
- 7 份参考源码和配置 SHA256 全部匹配。参考目录未修改；未调用参考训练/模型代码，未读取参考权重。
- 全部 14 个真实数据集的 source/split/label/class mapping 指纹、规模、维度、类别和划分均匹配。原 registry 的全部 dataset records 保持逐字段不变，仅增加协议实现版本字段。
- 提供图作为 sparse COO 输出；历史 dense hash 在分块重建中保持一致。CCoE 模型图 hash 不用于公共输入检查。
- 无参数 relation_union 与 graph_to_feature_context_v1 已实现并标明是输入适配。MV-ACM 适配维度 [1870,3740,3740,3740,3740]；MV-Citeseer [7406,3703]。
- Wikipedia 的 693 个固定测试观测通过独立原始多视图/标签多重集比较。合成重复行与矩形关系检查也通过；没有重新划分、补零或删视图。
- 14 项功能测试全部通过：dense/sparse 历史身份、并图方向/自环、关系转特征、矩形关系拒绝补零、Wikipedia 重复行身份、名称空间隔离、RNG、validation tie/patience、非法验证拒绝、五种子 sample std、目录不覆盖、跨表计划、数据篡改拒绝、checkpoint/导出/重建模型新推理及源码篡改拒绝。
- 首次完整检查后补全了已安装包版本记录；最后对结果接口及全部 14 项功能做针对性复核。每轮功能测试只运行一个两步 CPU 线性分类器 smoke fixture，不是任一论文方法，不生成正式主表统计。515 个正式任务未启动。
- 已保留参考纯数值工具使用的 torch.sparse.FloatTensor；当前 torch 提示构造器弃用警告，未抑制警告。coe 当前环境的全部身份和功能检查通过；更换框架版本时仍需重新核验。

## 可复核产物

- [全部 14 数据集核验](exp/audit_p00_full14_20261014_1801/audit.json)及[日志](log/audit_p00_full14_20261014_1801/console.log)。
- [最后结果接口复核](exp/audit_p00_api_final_20261014_1810/audit.json)及[14 项功能日志](log/audit_p00_api_final_20261014_1810/unit_tests.log)。
- [公共 API 与运行约定](README.md)，包含单任务、矩阵计划、模型重放、已保存预测复算和严格五种子汇总。

检查产物均在 shared/exp、shared/log，Git 忽略这些目录；提交只包含核心代码、测试、来源和使用说明。
