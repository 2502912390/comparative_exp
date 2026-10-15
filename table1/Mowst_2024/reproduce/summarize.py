"""Strict five-training-seed statistics for native 50-draw Mowst ACC."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from table1.Mowst_2024.reproduce.artifacts import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root), indent=2))


if __name__ == "__main__":
    main()
