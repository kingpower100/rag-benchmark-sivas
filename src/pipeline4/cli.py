from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline4",
        description="Pipeline 4: Aggregation and Ranking of RAG benchmark experiments",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to Pipeline 4 YAML config file",
    )
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    from src.pipeline4.orchestrator import Pipeline4Orchestrator

    orchestrator = Pipeline4Orchestrator()
    run_dir = orchestrator.run(config_path)
    print(f"\nDone. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
