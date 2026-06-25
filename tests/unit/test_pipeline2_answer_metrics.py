import pytest

from src.pipeline2.metrics.answer_metrics import (
    answer_relevancy_score,
    compute_answer_metrics,
    compute_bert_score,
    is_abstention,
)


class FakeBertScorer:
    def score(self, generated_answer: str, ground_truth_answer: str) -> dict[str, float]:
        if not generated_answer.strip() or not ground_truth_answer.strip():
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        generated_tokens = set(generated_answer.lower().split())
        truth_tokens = set(ground_truth_answer.lower().split())
        overlap = len(generated_tokens & truth_tokens)
        precision = overlap / len(generated_tokens)
        recall = overlap / len(truth_tokens)
        f1 = 0.0 if precision + recall == 0.0 else (2 * precision * recall) / (precision + recall)
        return {"bertscore_precision": precision, "bertscore_recall": recall, "bertscore_f1": f1}


def test_bertscore_identical_answers_are_high():
    metrics = compute_bert_score("net income increased", "net income increased", FakeBertScorer())

    assert metrics["bertscore_precision"] == pytest.approx(1.0)
    assert metrics["bertscore_recall"] == pytest.approx(1.0)
    assert metrics["bertscore_f1"] == pytest.approx(1.0)


def test_bertscore_unrelated_answers_are_lower():
    identical = compute_bert_score("net income increased", "net income increased", FakeBertScorer())
    unrelated = compute_bert_score("net income increased", "warehouse delivery blocked", FakeBertScorer())

    assert unrelated["bertscore_f1"] < identical["bertscore_f1"]


def test_bertscore_empty_generated_answer_returns_safe_values():
    metrics = compute_bert_score("", "net income increased", FakeBertScorer())

    assert metrics == {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


def test_bertscore_empty_reference_answer_returns_safe_values():
    metrics = compute_bert_score("net income increased", "", FakeBertScorer())

    assert metrics == {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


def test_literal_exact_match_uses_minimal_text_normalization_only():
    assert compute_answer_metrics("2602", "2602")["literal_exact_match"] == 1.0
    assert compute_answer_metrics(" 2602 ", "2602")["literal_exact_match"] == 1.0
    assert compute_answer_metrics("2.602 billion", "2602 million")["literal_exact_match"] == 0.0
    assert compute_answer_metrics("Revenue was 2602", "2602")["literal_exact_match"] == 0.0


def test_non_empty_answer_rate_keeps_backward_compatible_alias():
    metrics = compute_answer_metrics("UNKNOWN", "100")

    assert metrics["non_empty_answer_rate"] == 1.0
    assert metrics["answer_coverage_rate"] == 1.0


def test_abstention_detection_handles_configured_variants():
    assert is_abstention("")
    assert is_abstention("UNKNOWN")
    assert is_abstention("cannot determine")
    assert is_abstention("not provided", ["not provided"])
    assert not is_abstention("1250")


def test_answer_relevancy_score_is_deterministic_overlap_baseline():
    score = answer_relevancy_score("What was total revenue in 2020?", "total revenue")

    assert score == 1.0
    assert answer_relevancy_score("What was total revenue in 2020?", "1250") == 0.0


def test_rouge_l_remains_available_as_compatibility_metric():
    metrics = compute_answer_metrics("net income increased in 2020", "net income increased")

    assert metrics["rouge_l"] == pytest.approx(0.75)
    assert compute_answer_metrics("", "net income")["rouge_l"] == 0.0
    assert compute_answer_metrics("UNKNOWN", "100")["rouge_l"] == 0.0
