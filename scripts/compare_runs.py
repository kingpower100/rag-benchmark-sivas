from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print rows from Pipeline 2 summary files.")
    parser.add_argument(
        "summary",
        nargs="*",
        default=["data/eval/runs/pipeline2/eval_officeqa_treasury_hybrid_rrf_bge_small_qwen25_7b/summary_by_experiment.csv"],
    )
    args = parser.parse_args()
    for raw in args.summary:
        path = Path(raw)
        if not path.exists():
            print(f"missing: {path}")
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                print(row)


if __name__ == "__main__":
    main()
