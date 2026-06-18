# Clean Project State

This repository contains no real experiment results yet.

Real Pipeline 1 and Pipeline 2 benchmark results must be generated on the remote DGX machine. Local synthetic fixtures are allowed only inside the test suite under `tests/`; they are not benchmark evidence and must not be copied into real output folders.

Any audit marked `skipped` is not a real benchmark validation. It only means the evaluator could not find the required real Pipeline 1 artifacts on the machine where it was run.

A run is considered real only if it has all of the following artifacts:

- `results.jsonl`
- `events.jsonl`
- `manifest.json` or `run_manifest.json`
- valid SHA256 hashes for all required inputs and outputs
- `audit_report.json`
- `audit_report.md`

Repository output directories are intentionally empty or contain only placeholder files such as `.gitkeep` until real DGX experiments are run.
