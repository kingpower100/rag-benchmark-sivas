from __future__ import annotations

import argparse

from src.pipeline3.orchestrator import Pipeline3Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline 3 - Advanced Semantic Evaluation"
    )
    parser.add_argument("--config", required=True, help="Path to pipeline3 YAML config")
    args = parser.parse_args()
    Pipeline3Orchestrator().run(args.config)


if __name__ == "__main__":
    main()
