"""Explicit user-triggered strict five-seed aggregation; never called by matrix."""
import argparse
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.coe_benchmark import summarize_run,write_summary
from reproduce.common import DATASETS,METHOD_DIR


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-root",required=True,type=Path)
    p.add_argument("--datasets",nargs="+",default=list(DATASETS));args=p.parse_args()
    if not args.run_root.resolve().is_relative_to(METHOD_DIR/"exp") or tuple(args.datasets)!=DATASETS:
        raise ValueError("Primary summary must cover all five HAN datasets in method exp")
    summary=summarize_run(args.run_root,expected_datasets=args.datasets)
    write_summary(args.run_root,summary);print(summary)


if __name__=="__main__":main()
