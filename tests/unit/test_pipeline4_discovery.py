from __future__ import annotations

import json

from src.pipeline4.discovery import discover_p2_experiments, discover_p3_experiments


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discovery_excludes_smoke_runs(tmp_path):
    p2_dir = tmp_path / "pipeline2"
    p3_dir = tmp_path / "pipeline3"

    for experiment_id in (
        "91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0",
        "smoke_pgvector_dense",
    ):
        _write_json(
            p2_dir / f"{experiment_id}_eval" / "summary_metrics.json",
            {
                "summary_by_experiment": [
                    {
                        "experiment_id": experiment_id,
                        "n_questions": 1,
                        "run_valid": True,
                        "generation_failure_rate": 0.0,
                        "mean_recall_at_5": 1.0,
                        "mean_mrr_at_5": 1.0,
                        "mean_ndcg_at_5": 1.0,
                        "mean_context_precision_at_5": 1.0,
                    }
                ]
            },
        )
        _write_json(
            p3_dir / experiment_id / "evaluation_manifest.json",
            {
                "run_id": experiment_id,
                "summary": {
                    "n_questions": 1,
                    "judge_success_rate": 1.0,
                    "judge_failure_count": 0,
                    "mean_judge_correctness": 5.0,
                    "mean_judge_faithfulness": 5.0,
                    "mean_judge_context_relevance": 5.0,
                },
                "inputs": {},
                "ragas_stats": {},
                "reproducibility": {},
            },
        )

    p2_ids = [summary.experiment_id for summary in discover_p2_experiments(p2_dir)]
    p3_ids = [summary.experiment_id for summary in discover_p3_experiments(p3_dir)]

    assert p2_ids == ["91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0"]
    assert p3_ids == ["91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0"]
