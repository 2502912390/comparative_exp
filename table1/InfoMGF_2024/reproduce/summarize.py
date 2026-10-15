"""Strict replayed five-seed Table 1 and Table 2 statistics."""
import argparse
from pathlib import Path
from shared.coe_benchmark import summarize_run,write_summary
from .run import DATASETS


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--datasets',nargs='+',default=list(DATASETS))
    a=p.parse_args();root=Path(a.run_root)
    summary=summarize_run(root,expected_datasets=a.datasets)
    write_summary(root,summary)
    for table,ids in [(1,DATASETS[:5]),(2,DATASETS[5:])]:
        selected={**summary,'expected_datasets':list(ids),'rows':[r for r in summary['rows'] if r['dataset'] in ids]}
        selected['status']='complete' if all(r['status']=='complete' for r in selected['rows']) else 'incomplete'
        (root/f'table{table}').mkdir(exist_ok=True)
        write_summary(root/f'table{table}',selected)
