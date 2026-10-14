"""Manual-only statistics, invoked after the user asks to summarize."""
import argparse
from pathlib import Path
from shared.coe_benchmark import summarize_run, write_summary
from gcn_baseline.common import DATASETS, METHOD_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = parser.parse_args()
    if args.run_root.resolve().parent != METHOD_DIR / 'exp' or tuple(args.datasets) != DATASETS:
        parser.error('Provide this method formal root and all five datasets')
    summary = summarize_run(args.run_root, expected_datasets=DATASETS)
    write_summary(args.run_root, summary)
    print(summary)
