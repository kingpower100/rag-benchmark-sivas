"""Tests verifying scientific correctness of generation metric implementations.

Covers:
- OfficialBertScorer uses the bert-score library, not custom logic
- BERTScore model is multilingual
- _validate_production_config rejects deterministic_hash without offline_mode
- embedding_similarity and hashed_embedding_cosine_similarity are mutually exclusive
- bert_score_model_metadata records official implementation details
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline2.metrics.answer_metrics import (
    OfficialBertScorer,
    bert_score_model_metadata,
)
from src.pipeline2.orchestrator import EvaluationOrchestrator
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def _make_cfg(**embedding_overrides):
    base = {
        "evaluation": {
            "eval_run_id": "test",
            "retrieval_eval_field": "retrieved_original_context_ids",
        },
        "inputs": {"rag_outputs": []},
    }
    if embedding_overrides:
        base["embedding_similarity"] = embedding_overrides
    return EvalConfig.model_validate(base)


def _install_fake_bert_score(monkeypatch, scorer):
    mock_cls = MagicMock(return_value=scorer)
    fake_module = types.ModuleType("bert_score")
    fake_module.BERTScorer = mock_cls
    monkeypatch.setitem(sys.modules, "bert_score", fake_module)
    return mock_cls


# ─── _validate_production_config ──────────────────────────────────────────────

def test_production_config_rejects_deterministic_hash_without_offline_mode():
    cfg = _make_cfg(provider="deterministic_hash", offline_mode=False)
    with pytest.raises(ValueError, match="deterministic_hash"):
        EvaluationOrchestrator()._validate_production_config(cfg)


def test_production_config_allows_deterministic_hash_in_offline_mode():
    cfg = _make_cfg(provider="deterministic_hash", offline_mode=True)
    EvaluationOrchestrator()._validate_production_config(cfg)  # must not raise


def test_production_config_allows_sentence_transformers_without_offline_mode():
    cfg = _make_cfg(provider="sentence_transformers", model_name="intfloat/multilingual-e5-large")
    EvaluationOrchestrator()._validate_production_config(cfg)  # must not raise


def test_production_config_allows_disabled_embedding_similarity():
    cfg = _make_cfg(provider="deterministic_hash", enabled=False, offline_mode=False)
    EvaluationOrchestrator()._validate_production_config(cfg)  # must not raise


def test_default_config_fails_validation_because_hash_is_default():
    """Schema default provider=deterministic_hash with offline_mode=False must be rejected."""
    cfg = EvalConfig.model_validate({
        "evaluation": {"eval_run_id": "test", "retrieval_eval_field": "retrieved_original_context_ids"},
        "inputs": {"rag_outputs": []},
        # No embedding_similarity section — defaults to deterministic_hash, offline_mode=False
    })
    with pytest.raises(ValueError, match="deterministic_hash"):
        EvaluationOrchestrator()._validate_production_config(cfg)


# ─── OfficialBertScorer ───────────────────────────────────────────────────────

def test_official_bert_scorer_uses_bert_score_library(monkeypatch):
    """OfficialBertScorer must delegate to bert_score.BERTScorer, not custom logic."""
    mock_bert_scorer = MagicMock()
    mock_bert_scorer.device = "cpu"

    mock_cls = _install_fake_bert_score(monkeypatch, mock_bert_scorer)
    scorer = OfficialBertScorer("bert-base-multilingual-cased", device="cpu")
    mock_cls.assert_called_once_with(
        model_type="bert-base-multilingual-cased",
        idf=False,
        rescale_with_baseline=False,
        device="cpu",
    )
    assert scorer._scorer is mock_bert_scorer


def test_official_bert_scorer_model_name_is_multilingual(monkeypatch):
    _install_fake_bert_score(monkeypatch, MagicMock(device="cpu"))
    scorer = OfficialBertScorer("bert-base-multilingual-cased", device="cpu")
    assert "multilingual" in scorer.model_name.lower()


def test_official_bert_scorer_returns_official_bertscore_keys(monkeypatch):
    """score() must return official_bertscore_* keys; custom_bertscore_* must not appear."""
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            return self.values[index]

    mock_bert_scorer = MagicMock()
    mock_bert_scorer.device = "cpu"
    mock_bert_scorer.score.return_value = (
        FakeTensor([0.9]),
        FakeTensor([0.85]),
        FakeTensor([0.875]),
    )

    _install_fake_bert_score(monkeypatch, mock_bert_scorer)
    scorer = OfficialBertScorer("bert-base-multilingual-cased", device="cpu")
    result = scorer.score("generated answer", "reference answer")

    assert "official_bertscore_f1" in result
    assert "official_bertscore_precision" in result
    assert "official_bertscore_recall" in result
    assert "custom_bertscore_f1" not in result
    assert result["official_bertscore_f1"] == pytest.approx(0.875)


def test_official_bert_scorer_empty_input_short_circuits_without_model_call(monkeypatch):
    """Empty inputs must return zeros without calling BERTScorer.score."""
    mock_bert_scorer = MagicMock()
    mock_bert_scorer.device = "cpu"

    _install_fake_bert_score(monkeypatch, mock_bert_scorer)
    scorer = OfficialBertScorer("bert-base-multilingual-cased", device="cpu")
    result = scorer.score("", "reference answer")

    mock_bert_scorer.score.assert_not_called()
    assert result == {
        "official_bertscore_precision": 0.0,
        "official_bertscore_recall": 0.0,
        "official_bertscore_f1": 0.0,
    }


def test_official_bert_scorer_idf_and_rescale_flags_are_forwarded(monkeypatch):
    mock_bert_scorer = MagicMock()
    mock_bert_scorer.device = "cpu"

    mock_cls = _install_fake_bert_score(monkeypatch, mock_bert_scorer)
    OfficialBertScorer(
        "bert-base-multilingual-cased",
        device="cpu",
        idf=True,
        rescale_with_baseline=True,
    )
    mock_cls.assert_called_once_with(
        model_type="bert-base-multilingual-cased",
        idf=True,
        rescale_with_baseline=True,
        device="cpu",
    )


# ─── bert_score_model_metadata ────────────────────────────────────────────────

def test_bert_score_metadata_records_official_implementation():
    meta = bert_score_model_metadata(
        None, "bert-base-multilingual-cased", idf=False, rescale_with_baseline=False
    )
    assert meta["implementation"] == "official_bert_score_library"
    assert "library_version" in meta
    assert meta["idf"] is False
    assert meta["rescale_with_baseline"] is False


def test_bert_score_metadata_idf_and_rescale_flags_are_recorded():
    meta = bert_score_model_metadata(
        None, "bert-base-multilingual-cased", idf=True, rescale_with_baseline=True
    )
    assert meta["idf"] is True
    assert meta["rescale_with_baseline"] is True


def test_bert_score_metadata_has_no_custom_implementation_label():
    meta = bert_score_model_metadata(None, "bert-base-multilingual-cased")
    assert meta.get("implementation") != "custom"
    assert "transformers" not in meta.get("implementation", "")


# ─── Embedding metric mutual exclusivity ──────────────────────────────────────

def test_hashed_embedding_not_reported_as_semantic_embedding_similarity():
    """When provider=deterministic_hash, embedding_similarity must be None."""
    cfg = _make_cfg(
        provider="deterministic_hash",
        offline_mode=True,
        model_name="unit-test",
        dimensions=64,
    )
    rows = [{
        "question_id": "q1",
        "experiment_id": "exp",
        "generated_answer": "net income 100",
        "question": "Q?",
        "retrieved_original_context_ids": ["c1"],
    }]
    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"q1": {"id": "q1", "answer": "100"}},
        {"q1": ["c1"]},
        cfg,
    )
    row = evaluated[0]
    assert row["embedding_similarity"] is None, (
        "embedding_similarity must not be filled by deterministic_hash embedder"
    )
    assert row["hashed_embedding_cosine_similarity"] is not None, (
        "hashed_embedding_cosine_similarity must be filled when provider=deterministic_hash"
    )


def test_sentence_transformer_route_fills_embedding_similarity_not_hash():
    """When provider=sentence_transformers, embedding_similarity is filled; hash column is None."""
    cfg = _make_cfg(
        provider="sentence_transformers",
        model_name="intfloat/multilingual-e5-large",
    )

    mock_embedder = MagicMock()
    mock_embedder.metric_name = "embedding_similarity"
    mock_embedder.is_semantic = True
    mock_embedder.encode.return_value = [1.0, 0.0]

    with patch("src.pipeline2.orchestrator.build_answer_embedder", return_value=mock_embedder):
        rows = [{
            "question_id": "q1",
            "experiment_id": "exp",
            "generated_answer": "net income 100",
            "question": "Q?",
            "retrieved_original_context_ids": ["c1"],
        }]
        evaluated = EvaluationOrchestrator()._evaluate_rows(
            rows,
            {"q1": {"id": "q1", "answer": "100"}},
            {"q1": ["c1"]},
            cfg,
        )

    row = evaluated[0]
    assert row["embedding_similarity"] is not None, (
        "embedding_similarity must be filled by sentence_transformers embedder"
    )
    assert row["hashed_embedding_cosine_similarity"] is None, (
        "hashed_embedding_cosine_similarity must be None when provider=sentence_transformers"
    )


def test_embedding_similarity_columns_are_mutually_exclusive():
    """Exactly one of embedding_similarity / hashed_embedding_cosine_similarity must be non-null."""
    cfg = _make_cfg(
        provider="deterministic_hash",
        offline_mode=True,
        model_name="unit-test",
        dimensions=64,
    )
    rows = [
        {
            "question_id": f"q{i}",
            "experiment_id": "exp",
            "generated_answer": f"answer {i}",
            "question": "Q?",
            "retrieved_original_context_ids": ["c1"],
        }
        for i in range(5)
    ]
    qa = {f"q{i}": {"id": f"q{i}", "answer": str(i)} for i in range(5)}
    gold = {f"q{i}": ["c1"] for i in range(5)}

    evaluated = EvaluationOrchestrator()._evaluate_rows(rows, qa, gold, cfg)

    for row in evaluated:
        emb = row.get("embedding_similarity")
        hsh = row.get("hashed_embedding_cosine_similarity")
        assert (emb is None) != (hsh is None), (
            f"Exactly one column must be non-null per row; got "
            f"embedding_similarity={emb}, hashed_embedding_cosine_similarity={hsh}"
        )
