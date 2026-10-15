"""Record only a complete, source-consistent, replay-verified formal batch."""
import argparse
import fcntl
from pathlib import Path
from shared.coe_benchmark.statistics import summarize_run, write_summary
from .common import METHOD, REPO, DATASETS


def record(run_root):
    run_root = Path(run_root).resolve()
    if run_root.parent != (METHOD/'exp').resolve():
        raise ValueError('Result batch must belong to this method')
    summary = summarize_run(run_root, expected_datasets=DATASETS)
    if summary['status'] != 'complete' or summary['method'] != 'NodeFormer':
        raise ValueError('All 25 formal jobs must complete and pass replay before recording')
    write_summary(run_root, summary)
    rel = run_root.relative_to(REPO).as_posix()
    values = [f"{r['mean_percent']:.2f} ± {r['sample_std_percent']:.2f}" for r in summary['rows']]
    row = '| NodeFormer (2022) | '+' | '.join(values)+f' | [summary.csv]({rel}/summary.csv) / [summary.json]({rel}/summary.json) |'
    lines = [
        '### NodeFormer', '',
        f'- 正式批次：`{run_root.name}`；25/25 完成，五种子独立进程重放与保存预测 ACC 复算通过。',
        '- 固定作者通用配置：隐藏32、2层、4头、M=30、K=10、tau=0.25、lr=0.01、weight_decay=0.005、dropout=0、500 epochs；Gumbel／LayerNorm／残差开启，未进行超参数搜索。按验证 ACC 严格改善保存 best，作者无早停；测试只在恢复 best 后评估。',
        '- 结构先验：全部关系采用公共 relation_union，lambda=0.1 的边正则启用；rb_order=0 沿用通用默认，因此加性关系偏置未启用。MAG 保留全部113919节点和11319755条并图边。',
        '- 数值修复：训练 Gumbel 指数前按每个 head/sample 的全节点最大值平移；同一正缩放在注意力分子分母抵消，不裁剪／重抽噪声、不改变 tau 或模型。官方正常输入前向与梯度，以及极端噪声的 FP64 稠密参照检查通过。旧 v1 批次3次成功、22次非有限损失，整体排除，不与本批次混用。',
        '- 专用环境 coe_baseline_nodeformer（torch 2.1.1+cu121、PyG 2.5.3 及匹配扩展）；原 coe 未改动。保留官方随机投影生成机制，best 保存每层实际投影、种子和推理 RNG。',
        f'- 文件：[完整精度 CSV]({rel}/summary.csv)；[五种子 JSON]({rel}/summary.json)；[候选 LaTeX 行]({rel}/table_rows.tex)；[总日志](table1/NodeFormer_2022/log/{run_root.name}/matrix.log)；[适配说明](table1/NodeFormer_2022/ADAPTATION.md)。', '',
    ]
    for r in summary['rows']:
        name = r['dataset']
        links = '、'.join(f'[seed{s}/result.json]({rel}/{name}/seed{s}/result.json)' for s in r['seeds'])
        seeds = '、'.join(f"{r['per_seed_percent'][str(s)]:.6f}" for s in r['seeds'])
        lines.append(f'- **{name.upper() if name != "amazon" else "Amazon"}**：{links}。逐种子 ACC（%，seeds 1024–1028）：`{seeds}`。')
    block = '\n'.join(lines)+'\n\n'
    target = REPO/'EXPERIMENT_RESULTS.md'
    with target.open('r+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        content = stream.read()
        matches = [line for line in content.splitlines() if line.startswith('| NodeFormer (2022) |')]
        if len(matches) != 1:
            raise ValueError('Expected exactly one Table1 NodeFormer row')
        if '### NodeFormer\n' in content:
            raise FileExistsError('NodeFormer already recorded; refuse to replace a prior batch')
        anchor = '## 后续登记约定'
        if content.count(anchor) != 1:
            raise ValueError('Results registry anchor missing or ambiguous')
        content = content.replace(matches[0], row)
        content = content.replace('本次录入 GCN、HAN、HDMI；', '已录入 GCN、HAN、HDMI、NodeFormer；')
        content = content.replace(anchor, block+anchor)
        stream.seek(0)
        stream.write(content)
        stream.truncate()
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
    print(f'RECORDED {target}', flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args()
    record(args.run_root)

if __name__=='__main__': main()
