"""Strict four-dataset, five-seed formal summary."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.coe_benchmark import summarize_run, write_summary
from table2.MMDynamics_2022.reproduce.common import DATASETS


def summarize(run_root):
    summary = summarize_run(run_root, expected_datasets=DATASETS)
    write_summary(run_root, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    if tuple(args.datasets) != DATASETS:
        parser.error("Formal summaries require all four preregistered datasets")
    print(json.dumps(summarize(args.run_root), indent=2))
