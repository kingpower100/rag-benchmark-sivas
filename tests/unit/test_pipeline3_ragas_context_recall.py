"""
Tests for RAGAS Context Recall integration in Pipeline 3.

RAGAS 0.1.22 — ContextRecall uses EvaluationMode.qcg:
  inputs: question, contexts, ground_truth
  output column: context_recall  → stored as ragas_context_recall
  requires: LLM only (no embeddings)

All tests mock the RAGAS evaluator — Ollama is not required.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline3.aggregation.summarizer import summarize_semantic_metrics, _RAGAS_PREFIXES
from src.pipeline3.io.writer import PER_QUESTION_FIELDS, _build_report_markdown
from src.pipeline3.metrics.ragas_metrics import RagasEvaluator, RagasResults, RagasRow
from src.pipeline3.schemas.pipeline3_config_schema import (
    P3RagasConfig,
    P3RagasMetricsConfig,
    Pipeline3Config,
)
from src.pipeline4.loaders import P3Summary, load_p3_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_BASE_CONFIG = {
    "pipeline3": {"run_id": "test_run"},
    "inputs": {"pipeline1_results_path": "data/runs/pipeline1/test/results.jsonl"},
}


def _make_ragas_cfg(
    enabled: bool = True,
    faithfulness: bool = False,
    answer_relevancy: bool = False,
    context_recall: bool = True,
) -> P3RagasConfig:
    return P3RagasConfig(
        enabled=enabled,
        embeddings_device="cpu",
        require_cuda=False,
        fail_on_ragas_error=False,
        metrics=P3RagasMetricsConfig(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall,
        ),
    )


def _make_rows(n: int = 1) -> list[RagasRow]:
    return [
        RagasRow(
            question_id=f"q{i}",
            question="Was ist die Standardpreisgruppe?",
            answer="Die Standardpreisgruppe ist PG1.",
            contexts=["Preisgruppe PG1 ist die Standardpreisgruppe."],
            ground_truth="PG1 ist die Standardpreisgruppe.",
        )
        for i in range(n)
    ]


def _fake_ragas_result_df(rows: list[RagasRow], score: float) -> Any:
    """Return a minimal pandas DataFrame mimicking RAGAS output."""
    import pandas as pd

    data = {
        "question_id": [r.question_id for r in rows],
        "context_recall": [score] * len(rows),
    }
    return pd.DataFrame(data)


def _mock_ragas_evaluate(rows: list[RagasRow], score: float):
    """Patch ragas.evaluate to return a controlled result."""

    class _FakeResult:
        def to_pandas(self_inner):
            return _fake_ragas_result_df(rows, score)

    return _FakeResult()


def _make_evaluator_with_mocked_run(monkeypatch, score: float, rows: list[RagasRow]):
    """Create a RagasEvaluator whose _run_ragas is mocked to return score."""
    cfg = _make_ragas_cfg(context_recall=True)
    evaluator = RagasEvaluator(cfg)

    def _fake_run(inner_rows):
        return RagasResults(
            rows=[
                {"question_id": r.question_id, "ragas_context_recall": score}
                for r in inner_rows
            ],
            enabled_metrics=["context_recall"],
            nan_counts={"ragas_context_recall": 0},
            valid_counts={"ragas_context_recall": len(inner_rows)},
        )

    monkeypatch.setattr(evaluator, "_run_ragas", _fake_run)
    return evaluator


# ===========================================================================
# 1–3: Configuration
# ===========================================================================

class TestContextRecallConfig:
    def test_disabled_by_default(self):
        """context_recall defaults to False — opt-in only."""
        cfg = Pipeline3Config.model_validate(_VALID_BASE_CONFIG)
        assert cfg.ragas.metrics.context_recall is False

    def test_can_be_enabled_via_config(self):
        config = {
            **_VALID_BASE_CONFIG,
            "ragas": {
                "embeddings_device": "cpu",
                "require_cuda": False,
                "metrics": {"faithfulness": True, "answer_relevancy": True, "context_recall": True},
            },
        }
        cfg = Pipeline3Config.model_validate(config)
        assert cfg.ragas.metrics.context_recall is True

    def test_dict_value_rejected(self):
        """Genuinely invalid types (dict) must be rejected."""
        with pytest.raises(Exception):
            P3RagasMetricsConfig(
                faithfulness=True, answer_relevancy=True, context_recall={"enabled": True}  # type: ignore[arg-type]
            )

    def test_list_value_rejected(self):
        """Genuinely invalid types (list) must be rejected."""
        with pytest.raises(Exception):
            P3RagasMetricsConfig(
                faithfulness=True, answer_relevancy=True, context_recall=[True]  # type: ignore[arg-type]
            )

    def test_old_config_without_context_recall_key_valid(self):
        """Config dict that omits context_recall loads with default False."""
        config = {
            **_VALID_BASE_CONFIG,
            "ragas": {
                "embeddings_device": "cpu",
                "require_cuda": False,
                "metrics": {"faithfulness": True, "answer_relevancy": True},
            },
        }
        cfg = Pipeline3Config.model_validate(config)
        assert cfg.ragas.metrics.context_recall is False

    def test_extra_metric_key_rejected(self):
        """Unknown keys in metrics block are rejected by Pydantic extra=forbid."""
        with pytest.raises(Exception):
            P3RagasMetricsConfig(
                faithfulness=True,
                answer_relevancy=True,
                context_recall=False,
                context_precision=True,  # type: ignore[call-arg]
            )


# ===========================================================================
# 3: Metric registration
# ===========================================================================

class TestContextRecallMetricRegistration:
    def test_context_recall_class_registered_when_enabled(self):
        """_build_metrics must return a ContextRecall instance when enabled."""
        from ragas.metrics import ContextRecall

        evaluator = RagasEvaluator(_make_ragas_cfg(context_recall=True))
        metrics = evaluator._build_metrics(llm=None, embeddings=None)

        assert any(isinstance(m, ContextRecall) for m in metrics)

    def test_context_recall_not_registered_when_disabled(self):
        """_build_metrics must not include ContextRecall when config disables it."""
        from ragas.metrics import ContextRecall

        evaluator = RagasEvaluator(_make_ragas_cfg(context_recall=False))
        metrics = evaluator._build_metrics(llm=None, embeddings=None)

        assert not any(isinstance(m, ContextRecall) for m in metrics)

    def test_context_recall_uses_llm_not_embeddings(self):
        """ContextRecall construction must not receive embeddings."""
        from ragas.metrics import ContextRecall

        evaluator = RagasEvaluator(_make_ragas_cfg(context_recall=True))
        metrics = evaluator._build_metrics(llm=None, embeddings=None)

        cr_metrics = [m for m in metrics if isinstance(m, ContextRecall)]
        assert len(cr_metrics) == 1
        # ContextRecall has no embeddings attribute — it's LLM-only
        assert not hasattr(cr_metrics[0], "embeddings")

    def test_context_recall_metric_name_is_context_recall(self):
        """The metric's .name determines the output column prefix."""
        from ragas.metrics import ContextRecall

        cr = ContextRecall()
        assert cr.name == "context_recall"

    def test_context_recall_only_does_not_build_embeddings(self, monkeypatch):
        """ContextRecall-only RAGAS runs must not import/build sentence-transformer embeddings."""
        cfg = _make_ragas_cfg(context_recall=True, answer_relevancy=False)
        evaluator = RagasEvaluator(cfg)

        monkeypatch.setitem(sys.modules, "ragas", types.SimpleNamespace(evaluate=MagicMock()))
        monkeypatch.setitem(sys.modules, "ragas.run_config", types.SimpleNamespace(RunConfig=MagicMock()))
        monkeypatch.setitem(sys.modules, "ragas.llms", types.SimpleNamespace(LangchainLLMWrapper=lambda llm: llm))
        monkeypatch.setitem(sys.modules, "ragas.embeddings", types.SimpleNamespace(LangchainEmbeddingsWrapper=lambda emb: emb))
        monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=MagicMock()))
        monkeypatch.setattr(evaluator, "_build_embeddings", MagicMock(side_effect=AssertionError("should not build embeddings")))
        monkeypatch.setattr(evaluator, "_build_metrics", MagicMock(return_value=[]))

        result = evaluator._run_ragas(_make_rows(1))

        assert result.skipped is True
        evaluator._build_embeddings.assert_not_called()

    def test_faithfulness_and_context_recall_together(self):
        """Both metrics register without conflict."""
        from ragas.metrics import ContextRecall, Faithfulness

        evaluator = RagasEvaluator(
            _make_ragas_cfg(faithfulness=True, context_recall=True)
        )
        metrics = evaluator._build_metrics(llm=None, embeddings=None)

        assert any(isinstance(m, Faithfulness) for m in metrics)
        assert any(isinstance(m, ContextRecall) for m in metrics)


# ===========================================================================
# 4: Input field mapping — dataset already has question/contexts/ground_truth
# ===========================================================================

class TestContextRecallInputMapping:
    def test_ragas_evaluator_disabled_returns_skipped(self):
        """When ragas.enabled=False, evaluate() returns skipped result."""
        cfg = _make_ragas_cfg(enabled=False)
        evaluator = RagasEvaluator(cfg)
        result = evaluator.evaluate(_make_rows(3))
        assert result.skipped is True
        assert result.rows == []

    def test_evaluate_returns_context_recall_per_row(self, monkeypatch):
        """Valid rows produce ragas_context_recall in each result row."""
        rows = _make_rows(2)
        evaluator = _make_evaluator_with_mocked_run(monkeypatch, score=0.85, rows=rows)

        result = evaluator.evaluate(rows)

        assert result.skipped is False
        assert len(result.rows) == 2
        for row in result.rows:
            assert "ragas_context_recall" in row

    def test_ground_truth_used_not_generated_answer(self, monkeypatch):
        """Context Recall must use ground_truth, not generated answer."""
        rows = [
            RagasRow(
                question_id="q1",
                question="Test?",
                answer="WRONG answer that must not affect context recall",
                contexts=["Correct evidence in context."],
                ground_truth="Correct reference answer.",
            )
        ]
        captured_rows: list[RagasRow] = []

        cfg = _make_ragas_cfg(context_recall=True)
        evaluator = RagasEvaluator(cfg)

        def _fake_run(inner_rows):
            captured_rows.extend(inner_rows)
            return RagasResults(
                rows=[{"question_id": "q1", "ragas_context_recall": 0.9}],
                enabled_metrics=["context_recall"],
                nan_counts={"ragas_context_recall": 0},
                valid_counts={"ragas_context_recall": 1},
            )

        monkeypatch.setattr(evaluator, "_run_ragas", _fake_run)
        evaluator.evaluate(rows)

        assert captured_rows[0].ground_truth == "Correct reference answer."
        assert captured_rows[0].answer == "WRONG answer that must not affect context recall"


# ===========================================================================
# 5–7: Score ranges
# ===========================================================================

class TestContextRecallScoreRanges:
    def test_perfect_evidence_high_score(self, monkeypatch):
        rows = _make_rows(1)
        evaluator = _make_evaluator_with_mocked_run(monkeypatch, score=1.0, rows=rows)
        result = evaluator.evaluate(rows)
        assert result.rows[0]["ragas_context_recall"] == pytest.approx(1.0)

    def test_partial_evidence_intermediate_score(self, monkeypatch):
        rows = _make_rows(1)
        evaluator = _make_evaluator_with_mocked_run(monkeypatch, score=0.6, rows=rows)
        result = evaluator.evaluate(rows)
        assert result.rows[0]["ragas_context_recall"] == pytest.approx(0.6)

    def test_irrelevant_evidence_low_score(self, monkeypatch):
        rows = _make_rows(1)
        evaluator = _make_evaluator_with_mocked_run(monkeypatch, score=0.0, rows=rows)
        result = evaluator.evaluate(rows)
        assert result.rows[0]["ragas_context_recall"] == pytest.approx(0.0)


# ===========================================================================
# 8–10: Edge cases
# ===========================================================================

class TestContextRecallEdgeCases:
    def test_missing_reference_answer_produces_none(self, monkeypatch):
        """Row with empty ground_truth produces None (NaN from RAGAS → None)."""
        rows = [
            RagasRow(
                question_id="q1",
                question="Was kostet das?",
                answer="100 EUR.",
                contexts=["Preis: 100 EUR."],
                ground_truth="",  # missing reference
            )
        ]
        cfg = _make_ragas_cfg(context_recall=True)
        evaluator = RagasEvaluator(cfg)

        def _fake_run_nan(inner_rows):
            return RagasResults(
                rows=[{"question_id": "q1", "ragas_context_recall": None}],
                enabled_metrics=["context_recall"],
                nan_counts={"ragas_context_recall": 1},
                valid_counts={"ragas_context_recall": 0},
            )

        monkeypatch.setattr(evaluator, "_run_ragas", _fake_run_nan)
        result = evaluator.evaluate(rows)

        assert result.rows[0]["ragas_context_recall"] is None

    def test_empty_context_list_does_not_crash(self, monkeypatch):
        """Empty context list is handled without exception."""
        rows = [
            RagasRow(
                question_id="q1",
                question="Test?",
                answer="Answer.",
                contexts=[],
                ground_truth="Reference answer.",
            )
        ]
        cfg = _make_ragas_cfg(context_recall=True)
        evaluator = RagasEvaluator(cfg)

        def _fake_run(inner_rows):
            return RagasResults(
                rows=[{"question_id": "q1", "ragas_context_recall": 0.0}],
                enabled_metrics=["context_recall"],
                nan_counts={"ragas_context_recall": 0},
                valid_counts={"ragas_context_recall": 1},
            )

        monkeypatch.setattr(evaluator, "_run_ragas", _fake_run)
        result = evaluator.evaluate(rows)

        assert result.skipped is False
        assert result.rows[0]["ragas_context_recall"] == pytest.approx(0.0)

    def test_nan_result_converted_to_none(self, monkeypatch):
        """NaN produced by RAGAS is stored as None, not as float NaN."""
        rows = _make_rows(1)
        cfg = _make_ragas_cfg(context_recall=True)
        evaluator = RagasEvaluator(cfg)

        def _fake_run(inner_rows):
            return RagasResults(
                rows=[{"question_id": "q0", "ragas_context_recall": None}],
                enabled_metrics=["context_recall"],
                nan_counts={"ragas_context_recall": 1},
                valid_counts={"ragas_context_recall": 0},
            )

        monkeypatch.setattr(evaluator, "_run_ragas", _fake_run)
        result = evaluator.evaluate(rows)

        val = result.rows[0]["ragas_context_recall"]
        assert val is None or not math.isnan(val)  # never a float NaN


# ===========================================================================
# 14–16: Failure handling
# ===========================================================================

class TestContextRecallFailureHandling:
    def test_ragas_exception_returns_skipped_with_error(self, monkeypatch):
        """When _run_ragas raises, evaluate() returns skipped=True with error."""
        cfg = _make_ragas_cfg(context_recall=True)
        evaluator = RagasEvaluator(cfg)

        def _raise(_rows):
            raise RuntimeError("RAGAS timed out")

        monkeypatch.setattr(evaluator, "_run_ragas", _raise)
        result = evaluator.evaluate(_make_rows(1))

        assert result.skipped is True
        assert result.error is not None
        assert "RAGAS timed out" in result.error

    def test_ragas_failure_does_not_remove_other_ragas_results(self, monkeypatch):
        """When Context Recall fails (stage skipped), Faithfulness is also None
        (both share one ragas_evaluate call). Judge metrics run separately
        and are always preserved."""
        cfg = _make_ragas_cfg(faithfulness=True, context_recall=True)
        evaluator = RagasEvaluator(cfg)

        def _raise(_rows):
            raise RuntimeError("ContextRecall error")

        monkeypatch.setattr(evaluator, "_run_ragas", _raise)
        result = evaluator.evaluate(_make_rows(1))

        assert result.skipped is True
        assert result.rows == []

    def test_per_row_failure_does_not_stop_evaluation(self, monkeypatch):
        """A row with NaN context_recall does not prevent other rows from succeeding."""
        rows = _make_rows(3)
        cfg = _make_ragas_cfg(context_recall=True)
        evaluator = RagasEvaluator(cfg)

        def _fake_run(inner_rows):
            results = []
            for i, r in enumerate(inner_rows):
                results.append({
                    "question_id": r.question_id,
                    "ragas_context_recall": None if i == 1 else 0.8,
                })
            return RagasResults(
                rows=results,
                enabled_metrics=["context_recall"],
                nan_counts={"ragas_context_recall": 1},
                valid_counts={"ragas_context_recall": 2},
            )

        monkeypatch.setattr(evaluator, "_run_ragas", _fake_run)
        result = evaluator.evaluate(rows)

        assert result.skipped is False
        assert len(result.rows) == 3
        assert result.rows[0]["ragas_context_recall"] == pytest.approx(0.8)
        assert result.rows[1]["ragas_context_recall"] is None
        assert result.rows[2]["ragas_context_recall"] == pytest.approx(0.8)

    def test_valid_nan_and_missing_counts_correct(self, monkeypatch):
        """nan_counts and valid_counts are reported per metric."""
        rows = _make_rows(3)
        cfg = _make_ragas_cfg(context_recall=True)
        evaluator = RagasEvaluator(cfg)

        def _fake_run(inner_rows):
            return RagasResults(
                rows=[
                    {"question_id": "q0", "ragas_context_recall": 0.9},
                    {"question_id": "q1", "ragas_context_recall": None},  # NaN
                    {"question_id": "q2", "ragas_context_recall": 0.7},
                ],
                enabled_metrics=["context_recall"],
                nan_counts={"ragas_context_recall": 1},
                valid_counts={"ragas_context_recall": 2},
            )

        monkeypatch.setattr(evaluator, "_run_ragas", _fake_run)
        result = evaluator.evaluate(rows)

        assert result.nan_counts["ragas_context_recall"] == 1
        assert result.valid_counts["ragas_context_recall"] == 2


# ===========================================================================
# 22–23: Summary aggregation
# ===========================================================================

class TestContextRecallSummarizer:
    def test_context_recall_in_ragas_prefixes(self):
        """The summarizer must recognize ragas_context_recall as a RAGAS column."""
        assert any(
            "ragas_context_recall".startswith(p) for p in _RAGAS_PREFIXES
        )

    def test_summarizer_computes_mean_context_recall(self):
        rows = [
            {"question_id": "q1", "judge_success": True, "ragas_context_recall": 0.8},
            {"question_id": "q2", "judge_success": True, "ragas_context_recall": 0.6},
        ]
        summary = summarize_semantic_metrics(rows)
        assert "mean_ragas_context_recall" in summary
        assert summary["mean_ragas_context_recall"] == pytest.approx(0.7)

    def test_summarizer_excludes_none_from_mean(self):
        rows = [
            {"question_id": "q1", "judge_success": True, "ragas_context_recall": 0.8},
            {"question_id": "q2", "judge_success": True, "ragas_context_recall": None},
        ]
        summary = summarize_semantic_metrics(rows)
        assert summary["mean_ragas_context_recall"] == pytest.approx(0.8)

    def test_summarizer_returns_none_when_all_none(self):
        rows = [
            {"question_id": "q1", "judge_success": False, "ragas_context_recall": None},
            {"question_id": "q2", "judge_success": False, "ragas_context_recall": None},
        ]
        summary = summarize_semantic_metrics(rows)
        assert summary.get("mean_ragas_context_recall") is None

    def test_existing_metrics_unaffected_by_context_recall(self):
        rows = [
            {
                "question_id": "q1",
                "judge_success": True,
                "judge_correctness": 4,
                "ragas_faithfulness": 0.9,
                "ragas_answer_relevancy": 0.8,
                "ragas_context_recall": 0.7,
            }
        ]
        summary = summarize_semantic_metrics(rows)
        assert summary["mean_judge_correctness"] == pytest.approx(4.0)
        assert summary["mean_ragas_faithfulness"] == pytest.approx(0.9)
        assert summary["mean_ragas_answer_relevancy"] == pytest.approx(0.8)
        assert summary["mean_ragas_context_recall"] == pytest.approx(0.7)


# ===========================================================================
# 29, 31: Output schema — writer constants
# ===========================================================================

class TestContextRecallWriter:
    def test_per_question_csv_includes_context_recall_field(self):
        """ragas_context_recall must be in the per-question CSV field list."""
        assert "ragas_context_recall" in PER_QUESTION_FIELDS

    def test_ragas_context_recall_after_answer_relevancy_in_fields(self):
        """Field order: ragas_faithfulness → ragas_answer_relevancy → ragas_context_recall."""
        idx_f = PER_QUESTION_FIELDS.index("ragas_faithfulness")
        idx_ar = PER_QUESTION_FIELDS.index("ragas_answer_relevancy")
        idx_cr = PER_QUESTION_FIELDS.index("ragas_context_recall")
        assert idx_f < idx_ar < idx_cr

    def test_report_includes_context_recall_section(self):
        """The Markdown report must mention ragas_context_recall in the RAGAS section."""
        manifest = {
            "run_id": "test",
            "end_timestamp_utc": "2026-01-01T00:00:00+00:00",
            "judge_model": "qwen2.5:14b",
            "prompt_version": "v2",
            "ragas_stats": {"nan_counts": {"ragas_context_recall": 2}},
            "reproducibility": {},
        }
        summary = {
            "n_questions": 5,
            "judge_success_rate": 1.0,
            "mean_ragas_context_recall": 0.75,
        }
        report = _build_report_markdown(manifest, summary)
        assert "Context Recall" in report
        assert "0.7500" in report

    def test_report_shows_na_when_context_recall_disabled(self):
        """When context_recall is None (disabled), report shows n/a."""
        manifest = {
            "run_id": "test",
            "end_timestamp_utc": "2026-01-01T00:00:00+00:00",
            "judge_model": "qwen2.5:14b",
            "prompt_version": "v2",
            "ragas_stats": {"nan_counts": {}},
            "reproducibility": {},
        }
        summary = {
            "n_questions": 5,
            "judge_success_rate": 1.0,
            "mean_ragas_context_recall": None,
        }
        report = _build_report_markdown(manifest, summary)
        assert "Context Recall" in report
        assert "n/a" in report

    def test_report_shows_nan_warning_when_applicable(self):
        """NaN counts for context_recall appear in the data quality warnings section."""
        manifest = {
            "run_id": "test",
            "end_timestamp_utc": "2026-01-01T00:00:00+00:00",
            "judge_model": "qwen2.5:14b",
            "prompt_version": "v2",
            "ragas_stats": {"nan_counts": {"ragas_context_recall": 3}},
            "reproducibility": {},
        }
        summary = {"n_questions": 5, "judge_success_rate": 1.0}
        report = _build_report_markdown(manifest, summary)
        assert "ragas_context_recall" in report
        assert "3" in report


# ===========================================================================
# 31: Orchestrator manifest integration
# ===========================================================================

class TestContextRecallOrchestratorManifest:
    def _run_p3_with_mock(
        self,
        tmp_path: Path,
        monkeypatch,
        context_recall_enabled: bool,
        context_recall_result: float | None = 0.85,
    ) -> dict:  # noqa: D401
        from src.pipeline3.judge.response_parser import JudgeResponse
        from src.pipeline3.orchestrator import Pipeline3Orchestrator
        from src.pipeline3.stages.judge_stage import JudgeRowResult, JudgeStageResult
        from src.pipeline3.stages.loader_stage import LoaderResult

        rag_path = tmp_path / "results.jsonl"
        qa_path = tmp_path / "qa.jsonl"
        questions_path = tmp_path / "questions.jsonl"
        for p in (rag_path, qa_path, questions_path):
            p.write_text("{}\n", encoding="utf-8")

        cr_flag = "true" if context_recall_enabled else "false"
        config_path = tmp_path / "p3.yaml"
        out_dir = tmp_path / "out"
        config_path.write_text(
            f"""
pipeline3:
  run_id: "test_cr"
  output_dir: "{out_dir.as_posix()}"
  overwrite: true
  save_csv: true
  version: "1.0.0"
  prompt_version: "v2"
inputs:
  pipeline1_results_path: "{rag_path.as_posix()}"
  questions_path: "{questions_path.as_posix()}"
  qa_path: "{qa_path.as_posix()}"
  gold_contexts_path: "{qa_path.as_posix()}"
ragas:
  enabled: true
  fail_on_ragas_error: false
  embeddings_device: "cpu"
  require_cuda: false
  metrics:
    faithfulness: false
    answer_relevancy: false
    context_recall: {cr_flag}
llm_judge:
  enabled: false
""",
            encoding="utf-8",
        )

        rag_rows = [
            {
                "question_id": "q1",
                "experiment_id": "exp",
                "question": "Was ist PG1?",
                "generated_answer": "PG1 ist die Standardpreisgruppe.",
                "retrieved_context_texts": ["PG1 ist die Standardpreisgruppe."],
            }
        ]
        qa_rows = [{"question_id": "q1", "answer": "PG1 ist die Standardpreisgruppe."}]
        questions_rows = [{"question_id": "q1", "frage": "Was ist PG1?"}]

        monkeypatch.setattr(
            "src.pipeline3.orchestrator.load_inputs",
            lambda cfg, project_root: LoaderResult(
                rag_rows=rag_rows,
                qa_rows=qa_rows,
                questions_rows=questions_rows,
                rag_path=rag_path,
                questions_path=questions_path,
                qa_path=qa_path,
                gold_contexts_path=qa_path,
            ),
        )

        cr_row: dict = {"question_id": "q1"}
        if context_recall_enabled:
            cr_row["ragas_context_recall"] = context_recall_result
            enabled_metrics = ["context_recall"]
        else:
            enabled_metrics = []

        monkeypatch.setattr(
            "src.pipeline3.orchestrator.run_ragas_stage",
            lambda rows, qa_by_id, evaluator: RagasResults(
                rows=[cr_row],
                enabled_metrics=enabled_metrics,
                nan_counts={"ragas_context_recall": 0} if context_recall_enabled else {},
                valid_counts={"ragas_context_recall": 1} if context_recall_enabled else {},
            ),
        )

        class _FakeOllamaClient:
            def __init__(self, *a, **kw):
                pass

            def get_model_info(self):
                return {"digest": "sha256:test"}

        monkeypatch.setattr("src.pipeline3.orchestrator.OllamaClient", _FakeOllamaClient)

        run_dir = Pipeline3Orchestrator().run(str(config_path))
        manifest_text = (run_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
        return json.loads(manifest_text)

    def test_manifest_records_context_recall_enabled(self, monkeypatch):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            manifest = self._run_p3_with_mock(
                tmp_path, monkeypatch, context_recall_enabled=True
            )
            assert "context_recall" in manifest["ragas_stats"]["enabled_metrics"]
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_manifest_records_context_recall_disabled(self, monkeypatch):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            manifest = self._run_p3_with_mock(
                tmp_path, monkeypatch, context_recall_enabled=False
            )
            assert "context_recall" not in manifest["ragas_stats"]["enabled_metrics"]
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_summary_contains_mean_when_enabled(self, monkeypatch):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            manifest = self._run_p3_with_mock(
                tmp_path, monkeypatch, context_recall_enabled=True, context_recall_result=0.85
            )
            summary = manifest["summary"]
            assert "mean_ragas_context_recall" in summary
            assert summary["mean_ragas_context_recall"] == pytest.approx(0.85)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_summary_omits_mean_when_disabled(self, monkeypatch):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            manifest = self._run_p3_with_mock(
                tmp_path, monkeypatch, context_recall_enabled=False
            )
            summary = manifest["summary"]
            assert "mean_ragas_context_recall" not in summary
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_csv_includes_context_recall_column(self, monkeypatch):
        import csv

        tmp_path = Path(tempfile.mkdtemp())
        try:
            self._run_p3_with_mock(
                tmp_path, monkeypatch, context_recall_enabled=True, context_recall_result=0.75
            )
            out_dirs = list((tmp_path / "out").iterdir())
            assert len(out_dirs) == 1
            csv_path = out_dirs[0] / "per_question_semantic_metrics.csv"
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
            assert len(rows) == 1
            assert "ragas_context_recall" in rows[0]
            assert rows[0]["ragas_context_recall"] == "0.75"
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


# ===========================================================================
# 32: Pipeline 4 compatibility
# ===========================================================================

class TestContextRecallP4Compatibility:
    def _make_manifest(self, tmp_path: Path, mean_cr: float | None) -> Path:
        """Write a minimal P3 evaluation_manifest.json for P4 loader tests."""
        manifest = {
            "run_id": "p3_test_exp",
            "judge_model": "qwen2.5:14b",
            "prompt_version": "v2",
            "summary": {
                "n_questions": 5,
                "judge_success_rate": 0.9,
                "judge_success_count": 4,
                "judge_failure_count": 1,
                "mean_judge_correctness": 4.0,
                "mean_judge_faithfulness": 3.5,
                "mean_judge_context_relevance": 4.2,
                "mean_judge_completeness": 3.8,
                "mean_judge_hallucination": 0.5,
                "mean_judge_overall_score": 3.9,
                "mean_ragas_faithfulness": 0.8,
                "mean_ragas_answer_relevancy": 0.75,
            },
            "reproducibility": {
                "judge_model": "qwen2.5:14b",
                "prompt_version": "v2",
            },
            "inputs": {
                "rag_rows": 5,
                "qa_sha256": "abc123",
            },
            "ragas_stats": {
                "nan_counts": {
                    "ragas_faithfulness": 0,
                    "ragas_answer_relevancy": 0,
                },
            },
        }
        if mean_cr is not None:
            manifest["summary"]["mean_ragas_context_recall"] = mean_cr
            manifest["ragas_stats"]["nan_counts"]["ragas_context_recall"] = 1

        run_dir = tmp_path / "p3_test_exp"
        run_dir.mkdir()
        (run_dir / "evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return run_dir

    def test_p4_loader_reads_context_recall_when_present(self):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            run_dir = self._make_manifest(tmp_path, mean_cr=0.72)
            summary = load_p3_summary(run_dir)
            assert summary.mean_ragas_context_recall == pytest.approx(0.72)
            assert summary.ragas_context_recall_nan_rate == pytest.approx(1 / 5)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_p4_loader_handles_missing_context_recall_gracefully(self):
        """Old manifests without context_recall load without error; field is None."""
        tmp_path = Path(tempfile.mkdtemp())
        try:
            run_dir = self._make_manifest(tmp_path, mean_cr=None)
            summary = load_p3_summary(run_dir)
            assert summary.mean_ragas_context_recall is None
            assert summary.ragas_context_recall_nan_rate is None
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_p4_loader_existing_ragas_fields_unaffected(self):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            run_dir = self._make_manifest(tmp_path, mean_cr=0.80)
            summary = load_p3_summary(run_dir)
            assert summary.mean_ragas_faithfulness == pytest.approx(0.8)
            assert summary.mean_ragas_answer_relevancy == pytest.approx(0.75)
            assert summary.mean_judge_correctness == pytest.approx(4.0)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_p4_summary_has_context_recall_fields(self):
        """P3Summary dataclass must expose the new optional fields."""
        assert hasattr(P3Summary, "__dataclass_fields__")
        fields = P3Summary.__dataclass_fields__
        assert "mean_ragas_context_recall" in fields
        assert "ragas_context_recall_nan_rate" in fields


# ===========================================================================
# Integration marker (excluded from default suite)
# ===========================================================================

@pytest.mark.integration
@pytest.mark.requires_ollama
def test_context_recall_live_integration():
    """
    Full end-to-end Context Recall evaluation against a real Ollama instance.
    Requires qwen2.5:7b-instruct to be available at localhost:11434.

    Run with: pytest -m 'integration and requires_ollama'
    """
    from src.pipeline3.schemas.pipeline3_config_schema import P3RagasConfig, P3RagasMetricsConfig

    cfg = P3RagasConfig(
        enabled=True,
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen2.5:7b-instruct",
        llm_temperature=0.0,
        embeddings_device="cpu",
        require_cuda=False,
        fail_on_ragas_error=True,
        timeout_seconds=300,
        metrics=P3RagasMetricsConfig(
            faithfulness=False,
            answer_relevancy=False,
            context_recall=True,
        ),
    )
    evaluator = RagasEvaluator(cfg)
    rows = [
        RagasRow(
            question_id="integration_q1",
            question="Was ist die Standardpreisgruppe?",
            answer="PG1 ist die Standardpreisgruppe.",
            contexts=["Die Standardpreisgruppe ist PG1 und wird für Neukunden verwendet."],
            ground_truth="PG1 ist die Standardpreisgruppe.",
        )
    ]
    result = evaluator.evaluate(rows)

    assert result.skipped is False
    assert len(result.rows) == 1
    score = result.rows[0].get("ragas_context_recall")
    assert score is None or 0.0 <= score <= 1.0
