import argparse
from pathlib import Path
from shared.coe_benchmark.statistics import summarize_run, write_summary
from .common import DATASETS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',default=list(DATASETS))
    a=p.parse_args()
    summary=summarize_run(a.run_root,expected_datasets=a.datasets)
    write_summary(a.run_root,summary)
    print(summary,flush=True)
    return summary

if __name__=='__main__': main()
