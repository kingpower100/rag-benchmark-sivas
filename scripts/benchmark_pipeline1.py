from __future__ import annotations

import argparse

from src.pipeline1.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SIVAS Pipeline 1 baseline.")
    parser.add_argument(
        "--config",
        default="configs/pipeline1/experiments/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml",
    )
    args = parser.parse_args()
    print(run_pipeline(args.config))


if __name__ == "__main__":
    main()
