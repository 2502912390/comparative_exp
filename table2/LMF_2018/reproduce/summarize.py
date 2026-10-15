import argparse
from pathlib import Path
from shared.coe_benchmark import summarize_run,write_summary
from .common import DATASETS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    summary=summarize_run(a.run_root,expected_datasets=DATASETS)
    if summary['status']!='complete':raise ValueError('Incomplete formal matrix')
    write_summary(a.run_root,summary)
    for table,ids in ((2,DATASETS[:4]),(3,DATASETS[4:])):
        selected={**summary,'expected_datasets':list(ids),'rows':[r for r in summary['rows'] if r['dataset'] in ids]}
        (a.run_root/f'table{table}').mkdir(exist_ok=True);write_summary(a.run_root/f'table{table}',selected)
    print(summary)
