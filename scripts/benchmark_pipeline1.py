from __future__ import annotations

import argparse

from src.pipeline1.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the default Pipeline 1 benchmark.")
    parser.add_argument("--config", default="configs/pipeline1/experiments/officeqa_treasury_hybrid_rrf_bge_small_qwen25_7b.yaml")
    args = parser.parse_args()
    print(run_pipeline(args.config))


if __name__ == "__main__":
    main()
