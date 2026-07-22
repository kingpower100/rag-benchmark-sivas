import json
from pathlib import Path

import pytest

from src.pipeline2.metrics.chunk_retrieval_metrics import (
    ChunkGroundTruthError,
    ChunkGroundTruthLoader,
    compute_chunk_retrieval_metrics_for_ks,
)
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_retrieval_evaluation_defaults_preserve_legacy_behavior():
    cfg = EvalConfig.model_validate({"evaluation": {"eval_run_id": "eval"}, "inputs": {"rag_outputs": []}})

    assert cfg.retrieval_evaluation is None


def test_retrieval_evaluation_document_default_and_chunk_default():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval"},
            "inputs": {"rag_outputs": []},
            "retrieval_evaluation": {},
        }
    )

    assert cfg.retrieval_evaluation.document_level.enabled is True
    assert cfg.retrieval_evaluation.chunk_level.enabled is False


def test_enabled_chunk_evaluation_requires_ground_truth_path():
    with pytest.raises(ValueError, match="ground_truth_path is required"):
        EvalConfig.model_validate(
            {
                "evaluation": {"eval_run_id": "eval"},
                "inputs": {"rag_outputs": []},
                "retrieval_evaluation": {"chunk_level": {"enabled": True}},
            }
        )


def test_both_retrieval_levels_disabled_is_rejected():
    with pytest.raises(ValueError, match="enable at least one"):
        EvalConfig.model_validate(
            {
                "evaluation": {"eval_run_id": "eval"},
                "inputs": {"rag_outputs": []},
                "retrieval_evaluation": {
                    "document_level": {"enabled": False},
                    "chunk_level": {"enabled": False},
                },
            }
        )


def test_chunk_ground_truth_loader_loads_and_merges_multiple_records(tmp_path):
    path = tmp_path / "gold.jsonl"
    _write_jsonl(
        path,
        [
            {
                "question_id": "Q1",
                "chunk_config_id": "cfg",
                "gold_relevant_chunk_ids": ["A", "B", "B"],
            },
            {
                "question_id": "Q1",
                "chunk_config_id": "cfg",
                "gold_relevant_chunk_ids": ["C"],
            },
            {
                "question_id": "Q2",
                "chunk_config_id": "cfg",
                "gold_relevant_chunk_ids": ["D"],
            },
        ],
    )

    loaded = ChunkGroundTruthLoader(path).load()

    assert loaded.by_question == {"Q1": {"A", "B", "C"}, "Q2": {"D"}}
    assert loaded.question_count == 2
    assert loaded.gold_chunk_count == 4
    assert loaded.chunk_config_ids == {"cfg"}


def test_chunk_ground_truth_loader_rejects_invalid_json(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text("{bad\n", encoding="utf-8")

    with pytest.raises(ChunkGroundTruthError, match="Malformed chunk-level JSONL"):
        ChunkGroundTruthLoader(path).load()


def test_chunk_ground_truth_loader_rejects_missing_question_id(tmp_path):
    path = tmp_path / "gold.jsonl"
    _write_jsonl(path, [{"gold_relevant_chunk_ids": ["A"]}])

    with pytest.raises(ChunkGroundTruthError, match="missing required field 'question_id'"):
        ChunkGroundTruthLoader(path).load()


def test_chunk_ground_truth_loader_rejects_missing_chunk_ids(tmp_path):
    path = tmp_path / "gold.jsonl"
    _write_jsonl(path, [{"question_id": "Q1", "gold_relevant_chunk_ids": []}])

    with pytest.raises(ChunkGroundTruthError, match="contains no non-empty gold chunk IDs"):
        ChunkGroundTruthLoader(path).load()


def test_chunk_ground_truth_loader_rejects_empty_file(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ChunkGroundTruthError, match="is empty"):
        ChunkGroundTruthLoader(path).load()


def test_chunk_ground_truth_loader_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChunkGroundTruthLoader(tmp_path / "missing.jsonl").load()


def test_e00g_annotation_package_loads_all_96_questions():
    path = (
        "data/ground_truth/chunk_level/E00-G_sentence512_overlap200/"
        "gold_chunk_annotations_E00-G_sentence512_overlap200.jsonl"
    )

    loaded = ChunkGroundTruthLoader(Path(path)).load()

    assert loaded.question_count == 96
    assert loaded.gold_chunk_count > 96
    assert loaded.chunk_config_ids == {"E00-G_sentence512_overlap200"}
    assert loaded.package_metadata["integration_package.json"]["questions"] == 96


def test_chunk_metrics_exact_values_for_relevant_at_rank_1():
    metrics = compute_chunk_retrieval_metrics_for_ks(["GOLD", "A"], {"GOLD"}, [1, 3])

    assert metrics["chunk_hit_at_1"] == 1.0
    assert metrics["chunk_recall_at_1"] == 1.0
    assert metrics["chunk_mrr_at_1"] == 1.0
    assert metrics["chunk_ndcg_at_1"] == 1.0
    assert metrics["chunk_ndcg_at_3"] == 1.0


def test_chunk_metrics_exact_values_for_relevant_at_rank_3():
    metrics = compute_chunk_retrieval_metrics_for_ks(["A", "B", "GOLD"], {"GOLD"}, [3])

    assert metrics["chunk_hit_at_3"] == 1.0
    assert metrics["chunk_recall_at_3"] == 1.0
    assert metrics["chunk_mrr_at_3"] == 1 / 3
    assert metrics["chunk_ndcg_at_3"] == pytest.approx(0.5)


def test_chunk_metrics_no_relevant_chunks():
    metrics = compute_chunk_retrieval_metrics_for_ks(["A", "B"], {"GOLD"}, [5])

    assert metrics["chunk_hit_at_5"] == 0.0
    assert metrics["chunk_recall_at_5"] == 0.0
    assert metrics["chunk_mrr_at_5"] == 0.0
    assert metrics["chunk_ndcg_at_5"] == 0.0


def test_chunk_metrics_multiple_relevant_and_more_gold_than_k():
    metrics = compute_chunk_retrieval_metrics_for_ks(["A", "G1", "G2"], {"G1", "G2", "G3"}, [2])

    assert metrics["chunk_hit_at_2"] == 1.0
    assert metrics["chunk_recall_at_2"] == 1 / 3
    assert metrics["chunk_mrr_at_2"] == 0.5


def test_chunk_metrics_duplicate_nonrelevant_preserves_rank_for_ndcg_regression():
    metrics = compute_chunk_retrieval_metrics_for_ks(["A", "A", "GOLD"], {"GOLD"}, [3])

    assert metrics["chunk_hit_at_3"] == 1.0
    assert metrics["chunk_recall_at_3"] == 1.0
    assert metrics["chunk_ndcg_at_3"] == pytest.approx(0.5)
    assert metrics["chunk_mrr_at_3"] == 1 / 3


def test_chunk_metrics_duplicate_relevant_gets_recall_credit_once():
    metrics = compute_chunk_retrieval_metrics_for_ks(["GOLD", "GOLD"], {"GOLD", "OTHER"}, [2])

    assert metrics["chunk_hit_at_2"] == 1.0
    assert metrics["chunk_recall_at_2"] == 0.5
    assert metrics["chunk_ndcg_at_2"] == pytest.approx(1 / (1 + 1 / 1.584962500721156))
