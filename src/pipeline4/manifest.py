from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline4.reporting import ExperimentRecord
from src.pipeline4.schemas import Pipeline4Config


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    records: list[ExperimentRecord],
    output_files: dict[str, Path],
    cfg: Pipeline4Config,
    cfg_path: str,
    start_ts: datetime,
    end_ts: datetime,
    run_dir: Path,
) -> Path:
    output_hashes: dict[str, str] = {}
    for label, path in output_files.items():
        if path.exists():
            output_hashes[label] = _sha256_file(path)

    p2_inputs = [
        {
            "experiment_id": r.experiment_id,
            "p2_run_dir": r.p2_run_dir,
            "qa_hash": r.qa_hash,
        }
        for r in records
    ]
    p3_inputs = [
        {
            "experiment_id": r.experiment_id,
            "p3_run_dir": r.p3_run_dir,
            "judge_model": r.judge_model,
            "prompt_version": r.prompt_version,
        }
        for r in records
        if r.has_p3
    ]

    manifest: dict[str, Any] = {
        "run_id": cfg.run_id,
        "pipeline_version": "1.0.0",
        "start_timestamp_utc": start_ts.isoformat(),
        "end_timestamp_utc": end_ts.isoformat(),
        "duration_seconds": round((end_ts - start_ts).total_seconds(), 3),
        "config_path": cfg_path,
        "ranking_mode": cfg.ranking_mode,
        "retrieval_score_weights": {
            "recall_at_5": cfg.retrieval_score_weights.recall_at_5,
            "mrr_at_5": cfg.retrieval_score_weights.mrr_at_5,
            "ndcg_at_5": cfg.retrieval_score_weights.ndcg_at_5,
            "context_precision_at_5": cfg.retrieval_score_weights.context_precision_at_5,
        },
        "rqi_weights": {
            "correctness": cfg.rqi_weights.correctness,
            "faithfulness": cfg.rqi_weights.faithfulness,
            "context_relevance": cfg.rqi_weights.context_relevance,
            "recall_at_5": cfg.rqi_weights.recall_at_5,
            "no_unknown": cfg.rqi_weights.no_unknown,
        },
        "validation_thresholds": {
            "max_generation_failure_rate": cfg.validation.max_generation_failure_rate,
            "min_judge_success_rate": cfg.validation.min_judge_success_rate,
            "max_ragas_nan_rate": cfg.validation.max_ragas_nan_rate,
        },
        "inputs": {
            "pipeline2_runs_dir": cfg.pipeline2_runs_dir,
            "pipeline3_runs_dir": cfg.pipeline3_runs_dir,
            "p2_experiments_loaded": len(records),
            "p3_experiments_loaded": len(p3_inputs),
            "p2_inputs": p2_inputs,
            "p3_inputs": p3_inputs,
        },
        "outputs": {
            label: str(path) for label, path in output_files.items()
        },
        "output_hashes": output_hashes,
        "summary": {
            "total_experiments": len(records),
            "ranked_retrieval": sum(1 for r in records if r.retrieval_rank is not None),
            "ranked_rqi": sum(1 for r in records if r.rqi_rank is not None),
            "excluded": sum(1 for r in records if r.p2_status != "VALID"),
        },
    }

    manifest_path = run_dir / "pipeline4_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path
