from __future__ import annotations

import pytest

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import ValidationThresholds
from src.pipeline4.validation import (
    P2_EXCLUDED_FAILURE_RATE,
    P2_EXCLUDED_RUN_INVALID,
    P2_VALID,
    P3_JUDGE_WARNING,
    P3_NOT_AVAILABLE,
    P3_VALID,
    build_comparison_groups,
    validate_all,
    validate_p2,
    validate_p3,
)


def _thresholds() -> ValidationThresholds:
    return ValidationThresholds()


def _p2(
    exp_id="exp_a",
    run_valid=True,
    failure_rate=0.0,
    qa_hash="hash1",
    n_questions=96,
) -> P2Summary:
    return P2Summary(
        experiment_id=exp_id,
        n_questions=n_questions,
        run_valid=run_valid,
        generation_failure_rate=failure_rate,
        mean_recall_at_5=0.5,
        mean_mrr_at_5=0.8,
        mean_ndcg_at_5=0.6,
        mean_context_precision_at_5=0.4,
        unknown_rate=0.1,
        mean_embedding_similarity=0.88,
        mean_official_bertscore_f1=0.66,
        qa_hash=qa_hash,
        gold_contexts_hash=qa_hash,
        p2_run_dir=f"/fake/{exp_id}",
    )


def _p3(
    exp_id="exp_a",
    judge_success_rate=1.0,
    ragas_faith_nan_rate=0.0,
    ragas_rel_nan_rate=0.0,
    qa_sha256="hash1",
) -> P3Summary:
    return P3Summary(
        run_id=f"p3_{exp_id}",
        experiment_id=exp_id,
        n_questions=96,
        judge_model="qwen2.5:14b",
        prompt_version="v2",
        qa_sha256=qa_sha256,
        judge_success_rate=judge_success_rate,
        judge_failure_count=0,
        mean_judge_correctness=3.0,
        mean_judge_faithfulness=3.0,
        mean_judge_completeness=2.0,
        mean_judge_hallucination=1.0,
        mean_judge_context_relevance=4.0,
        mean_judge_overall_score=3.0,
        mean_ragas_faithfulness=0.65,
        mean_ragas_answer_relevancy=0.70,
        ragas_faithfulness_nan_rate=ragas_faith_nan_rate,
        ragas_answer_relevancy_nan_rate=ragas_rel_nan_rate,
        p3_run_dir=f"/fake/p3/{exp_id}",
    )


class TestValidateP2:
    def test_valid_run(self):
        p2 = _p2(run_valid=True, failure_rate=0.0)
        val = validate_p2(p2, _thresholds())
        assert val.p2_status == P2_VALID
        assert not val.p2_excluded

    def test_run_invalid(self):
        p2 = _p2(run_valid=False)
        val = validate_p2(p2, _thresholds())
        assert val.p2_status == P2_EXCLUDED_RUN_INVALID
        assert val.p2_excluded

    def test_high_failure_rate(self):
        p2 = _p2(run_valid=True, failure_rate=0.10)
        val = validate_p2(p2, _thresholds())
        assert val.p2_status == P2_EXCLUDED_FAILURE_RATE
        assert val.p2_excluded

    def test_failure_rate_at_threshold(self):
        p2 = _p2(run_valid=True, failure_rate=0.05)
        val = validate_p2(p2, _thresholds())
        assert val.p2_status == P2_VALID

    def test_failure_rate_just_above_threshold(self):
        p2 = _p2(run_valid=True, failure_rate=0.051)
        val = validate_p2(p2, _thresholds())
        assert val.p2_status == P2_EXCLUDED_FAILURE_RATE

    def test_run_invalid_takes_precedence(self):
        p2 = _p2(run_valid=False, failure_rate=0.10)
        val = validate_p2(p2, _thresholds())
        assert val.p2_status == P2_EXCLUDED_RUN_INVALID


class TestValidateP3:
    def test_none_returns_not_available(self):
        status, issues, warnings = validate_p3(None, _thresholds())
        assert status == P3_NOT_AVAILABLE
        assert not issues
        assert not warnings

    def test_valid_p3(self):
        p3 = _p3(judge_success_rate=1.0)
        status, issues, warnings = validate_p3(p3, _thresholds())
        assert status == P3_VALID
        assert not issues

    def test_low_judge_success_rate(self):
        p3 = _p3(judge_success_rate=0.80)
        status, issues, _w = validate_p3(p3, _thresholds())
        assert status == P3_JUDGE_WARNING
        assert len(issues) >= 1

    def test_ragas_nan_warning_does_not_exclude(self):
        p3 = _p3(judge_success_rate=1.0, ragas_faith_nan_rate=0.20)
        status, issues, ragas_warnings = validate_p3(p3, _thresholds())
        assert status == P3_VALID
        assert not issues
        assert len(ragas_warnings) >= 1

    def test_ragas_nan_at_threshold_no_warning(self):
        p3 = _p3(judge_success_rate=1.0, ragas_faith_nan_rate=0.10)
        status, issues, ragas_warnings = validate_p3(p3, _thresholds())
        assert status == P3_VALID
        assert not ragas_warnings


class TestComparisonGroups:
    def test_same_qa_hash_same_group_retrieval(self):
        p2_list = [
            _p2(exp_id="a", qa_hash="hash1", n_questions=96),
            _p2(exp_id="b", qa_hash="hash1", n_questions=96),
        ]
        p3_map = {"a": None, "b": None}
        validations = validate_all(p2_list, p3_map, _thresholds())
        groups = build_comparison_groups(p2_list, p3_map, validations, "retrieval_only")
        assert len(groups) == 1
        assert set(groups[0].experiment_ids) == {"a", "b"}

    def test_different_qa_hash_different_groups(self):
        p2_list = [
            _p2(exp_id="a", qa_hash="hash1"),
            _p2(exp_id="b", qa_hash="hash2"),
        ]
        p3_map = {"a": None, "b": None}
        validations = validate_all(p2_list, p3_map, _thresholds())
        groups = build_comparison_groups(p2_list, p3_map, validations, "retrieval_only")
        assert len(groups) == 2

    def test_excluded_not_in_any_group(self):
        p2_list = [
            _p2(exp_id="a", run_valid=True),
            _p2(exp_id="b", run_valid=False),
        ]
        p3_map = {"a": None, "b": None}
        validations = validate_all(p2_list, p3_map, _thresholds())
        groups = build_comparison_groups(p2_list, p3_map, validations, "retrieval_only")
        all_exp_ids = [eid for g in groups for eid in g.experiment_ids]
        assert "b" not in all_exp_ids

    def test_rag_mode_incomplete_p3_flagged(self):
        p2_list = [
            _p2(exp_id="a"),
            _p2(exp_id="b"),
        ]
        p3_map = {"a": _p3("a"), "b": None}
        validations = validate_all(p2_list, p3_map, _thresholds())
        groups = build_comparison_groups(p2_list, p3_map, validations, "overall_rag")
        # With overall_rag mode, judge_model differs: a has "qwen2.5:14b", b has "NO_P3"
        # so they end up in different groups
        assert len(groups) == 2
        group_with_a = next(g for g in groups if "a" in g.experiment_ids)
        assert group_with_a.has_complete_p3

    def test_rag_mode_complete_p3_has_complete_flag(self):
        p2_list = [_p2("a"), _p2("b")]
        p3_map = {"a": _p3("a"), "b": _p3("b")}
        validations = validate_all(p2_list, p3_map, _thresholds())
        groups = build_comparison_groups(p2_list, p3_map, validations, "overall_rag")
        for g in groups:
            assert g.has_complete_p3
