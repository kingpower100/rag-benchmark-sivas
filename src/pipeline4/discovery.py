from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.pipeline4.loaders import P2Summary, P3Summary, load_p2_summary, load_p3_summary


def _is_smoke_run(identifier: str) -> bool:
    return identifier.startswith("smoke_") or identifier.endswith("_smoke")


def discover_p2_experiments(p2_runs_dir: Path) -> list[P2Summary]:
    summaries: list[P2Summary] = []
    if not p2_runs_dir.exists():
        print(f"  P2 runs directory not found: {p2_runs_dir}")
        return summaries
    for run_dir in sorted(p2_runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "summary_metrics.json").exists():
            continue
        try:
            summary = load_p2_summary(run_dir)
            if _is_smoke_run(summary.experiment_id) or _is_smoke_run(run_dir.name):
                continue
            summaries.append(summary)
        except Exception as exc:
            print(f"  Warning: skipping P2 run {run_dir.name}: {exc}")
    return summaries


def discover_p3_experiments(p3_runs_dir: Path) -> list[P3Summary]:
    summaries: list[P3Summary] = []
    if not p3_runs_dir.exists():
        return summaries
    for run_dir in sorted(p3_runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "evaluation_manifest.json").exists():
            continue
        try:
            summary = load_p3_summary(run_dir)
            if _is_smoke_run(summary.experiment_id) or _is_smoke_run(run_dir.name):
                continue
            summaries.append(summary)
        except Exception as exc:
            print(f"  Warning: skipping P3 run {run_dir.name}: {exc}")
    return summaries


def match_p3_to_p2(
    p2_summaries: list[P2Summary], p3_summaries: list[P3Summary]
) -> dict[str, Optional[P3Summary]]:
    p3_by_experiment: dict[str, P3Summary] = {p3.experiment_id: p3 for p3 in p3_summaries}
    return {p2.experiment_id: p3_by_experiment.get(p2.experiment_id) for p2 in p2_summaries}
