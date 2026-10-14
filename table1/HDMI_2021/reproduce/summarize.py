"""Strict fixed-five-seed HDMI statistics and candidate LaTeX row."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.coe_benchmark import summarize_run, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "yelp", "amazon", "mag"])
    args = parser.parse_args()
    summary = summarize_run(args.run_root, expected_datasets=args.datasets)
    write_summary(args.run_root, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
