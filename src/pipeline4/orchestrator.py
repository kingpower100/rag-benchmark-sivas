from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline4.discovery import (
    discover_p2_experiments,
    discover_p3_experiments,
    match_p3_to_p2,
)
from src.pipeline4.manifest import write_manifest
from src.pipeline4.ranking import rank_retrieval, rank_rqi
from src.pipeline4.reporting import (
    build_records,
    write_comparison_report,
    write_full_summary,
    write_leaderboard_json,
    write_retrieval_leaderboard,
    write_rqi_leaderboard,
    write_validation_report,
)
from src.pipeline4.schemas import Pipeline4Config
from src.pipeline4.scoring import compute_retrieval_score, compute_rqi
from src.pipeline4.validation import build_comparison_groups, validate_all


class Pipeline4Orchestrator:
    def run(self, config_path: str) -> Path:
        start_ts = datetime.now(timezone.utc)
        start_time = time.time()

        cfg = Pipeline4Config.from_yaml(config_path)
        project_root = Path(__file__).resolve().parents[2]

        def _resolve(rel_or_abs: str) -> Path:
            p = Path(rel_or_abs)
            return p if p.is_absolute() else project_root / p

        p2_runs_dir = _resolve(cfg.pipeline2_runs_dir)
        p3_runs_dir = _resolve(cfg.pipeline3_runs_dir)
        run_dir = _resolve(cfg.output_dir) / cfg.run_id

        print(f"Pipeline 4 — Aggregation and Reporting")
        print(f"  Config: {config_path}")
        print(f"  Run ID: {cfg.run_id}")
        print(f"  Ranking mode: {cfg.ranking_mode}")
        print(f"  Output dir: {run_dir}")
        print()

        run_dir.mkdir(parents=True, exist_ok=True)

        print("Discovering P2 experiments...")
        p2_summaries = discover_p2_experiments(p2_runs_dir)
        print(f"  Found {len(p2_summaries)} P2 experiment(s)")

        print("Discovering P3 experiments...")
        p3_summaries = discover_p3_experiments(p3_runs_dir)
        print(f"  Found {len(p3_summaries)} P3 experiment(s)")

        p3_map = match_p3_to_p2(p2_summaries, p3_summaries)
        p3_matched = sum(1 for v in p3_map.values() if v is not None)
        print(f"  P3 matched to {p3_matched}/{len(p2_summaries)} P2 experiment(s)")
        print()

        print("Validating experiments...")
        validations = validate_all(p2_summaries, p3_map, cfg.validation)
        excluded = sum(1 for v in validations.values() if v.p2_excluded)
        print(f"  Excluded: {excluded}, Valid: {len(validations) - excluded}")

        print("Building comparison groups...")
        comparison_groups = build_comparison_groups(
            p2_summaries, p3_map, validations, cfg.ranking_mode
        )
        print(f"  {len(comparison_groups)} comparison group(s)")
        print()

        print("Building experiment records and computing scores...")
        records = build_records(p2_summaries, p3_map, validations, comparison_groups, cfg)

        print("Ranking experiments...")
        retrieval_scores = {
            r.experiment_id: r.retrieval_score
            for r in records
            if validations[r.experiment_id].retrieval_leaderboard_eligible
        }
        rqi_scores = {
            r.experiment_id: r.rqi
            for r in records
            if validations[r.experiment_id].overall_leaderboard_eligible
        }

        retrieval_ranks = rank_retrieval(retrieval_scores, comparison_groups)
        rqi_ranks = rank_rqi(rqi_scores, comparison_groups, cfg.ranking_mode)

        for rec in records:
            rec.retrieval_rank = retrieval_ranks.get(rec.experiment_id)
            rec.rqi_rank = rqi_ranks.get(rec.experiment_id)

        print("Writing output files...")

        ret_leaderboard_path = run_dir / "retrieval_leaderboard.csv"
        rqi_leaderboard_path = run_dir / "overall_rqi_leaderboard.csv"
        full_summary_path = run_dir / "full_experiment_summary.csv"
        leaderboard_json_path = run_dir / "leaderboard.json"
        comparison_report_path = run_dir / "comparison_report.md"
        validation_report_path = run_dir / "validation_report.json"

        write_retrieval_leaderboard(records, ret_leaderboard_path)
        write_rqi_leaderboard(records, rqi_leaderboard_path)
        write_full_summary(records, full_summary_path)
        write_leaderboard_json(records, comparison_groups, cfg, leaderboard_json_path)
        write_comparison_report(records, comparison_groups, cfg, comparison_report_path)
        write_validation_report(records, comparison_groups, cfg, validation_report_path)

        end_ts = datetime.now(timezone.utc)

        output_files = {
            "retrieval_leaderboard": ret_leaderboard_path,
            "overall_rqi_leaderboard": rqi_leaderboard_path,
            "full_experiment_summary": full_summary_path,
            "leaderboard_json": leaderboard_json_path,
            "comparison_report": comparison_report_path,
            "validation_report": validation_report_path,
        }

        manifest_path = write_manifest(
            records, output_files, cfg, config_path, start_ts, end_ts, run_dir
        )
        print(f"  Manifest: {manifest_path}")
        print()

        elapsed = time.time() - start_time
        ranked_ret = sum(1 for r in records if r.retrieval_rank is not None)
        ranked_rqi = sum(1 for r in records if r.rqi_rank is not None)
        print(f"Pipeline 4 complete in {elapsed:.1f}s")
        print(f"  Retrieval ranking: {ranked_ret} experiment(s)")
        print(f"  RQI ranking: {ranked_rqi} experiment(s)")
        print(f"  Output: {run_dir}")

        return run_dir
