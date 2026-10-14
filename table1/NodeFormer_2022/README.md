# NodeFormer (2022)

NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification

- 对应表格：1
- 论文：https://proceedings.neurips.cc/paper_files/paper/2022/file/af790b7ae573771689438bbcfc5933fe-Paper-Conference.pdf
- PDF 状态：ok
- 作者代码：https://github.com/qitianwu/NodeFormer
- 代码状态：ok
- 固定下载提交：`5390b6908326905b2eb840f65a70b0f4082aef43`

paper.pdf 为论文；paper.txt 为检索用文本；code/ 为作者仓库。来源、校验值及下载状态见 sources.json。

固定配置复现入口见 [REPRODUCE.md](REPRODUCE.md)、
[参数来源](configs/author_sources.json) 与 [适配说明](ADAPTATION.md)。
正式配置为官方通用默认值；启用原生 Gumbel/LayerNorm/残差，固定 relation_union
边正则先验，验证 ACC 选 best，500 epochs，无超参数搜索。
