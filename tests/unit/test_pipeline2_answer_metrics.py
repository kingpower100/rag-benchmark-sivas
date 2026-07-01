import pytest

from src.pipeline2.metrics.answer_metrics import (
    question_answer_lexical_f1,
    compute_answer_metrics,
    compute_bert_score,
    is_abstention,
)


class FakeBertScorer:
    def score(self, generated_answer: str, ground_truth_answer: str) -> dict[str, float]:
        if not generated_answer.strip() or not ground_truth_answer.strip():
            return {"custom_bertscore_precision": 0.0, "custom_bertscore_recall": 0.0, "custom_bertscore_f1": 0.0}
        generated_tokens = set(generated_answer.lower().split())
        truth_tokens = set(ground_truth_answer.lower().split())
        overlap = len(generated_tokens & truth_tokens)
        precision = overlap / len(generated_tokens)
        recall = overlap / len(truth_tokens)
        f1 = 0.0 if precision + recall == 0.0 else (2 * precision * recall) / (precision + recall)
        return {"custom_bertscore_precision": precision, "custom_bertscore_recall": recall, "custom_bertscore_f1": f1}


def test_bertscore_identical_answers_are_high():
    metrics = compute_bert_score("net income increased", "net income increased", FakeBertScorer())

    assert metrics["custom_bertscore_precision"] == pytest.approx(1.0)
    assert metrics["custom_bertscore_recall"] == pytest.approx(1.0)
    assert metrics["custom_bertscore_f1"] == pytest.approx(1.0)


def test_bertscore_unrelated_answers_are_lower():
    identical = compute_bert_score("net income increased", "net income increased", FakeBertScorer())
    unrelated = compute_bert_score("net income increased", "warehouse delivery blocked", FakeBertScorer())

    assert unrelated["custom_bertscore_f1"] < identical["custom_bertscore_f1"]


def test_bertscore_empty_generated_answer_returns_safe_values():
    metrics = compute_bert_score("", "net income increased", FakeBertScorer())

    assert metrics == {"custom_bertscore_precision": 0.0, "custom_bertscore_recall": 0.0, "custom_bertscore_f1": 0.0}


def test_bertscore_empty_reference_answer_returns_safe_values():
    metrics = compute_bert_score("net income increased", "", FakeBertScorer())

    assert metrics == {"custom_bertscore_precision": 0.0, "custom_bertscore_recall": 0.0, "custom_bertscore_f1": 0.0}


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


def test_question_answer_lexical_f1_uses_symmetric_f1_formula():
    # Partial question-vocabulary overlap: old formula gave 1.0 (parroting bug),
    # F1 gives 0.8 because the question has 3 content tokens and only 2 match.
    # Q content tokens: {total, revenue, 2020} (3)
    # A content tokens: {total, revenue} (2)  → intersection=2
    # F1 = 2*2 / (3+2) = 0.8
    score = question_answer_lexical_f1("What was total revenue in 2020?", "total revenue")
    assert score == pytest.approx(0.8)


def test_question_answer_lexical_f1_zero_for_no_overlap():
    assert question_answer_lexical_f1("What was total revenue in 2020?", "1250") == 0.0


def test_question_answer_lexical_f1_perfect_bidirectional_overlap():
    # When Q and A share all content tokens in both directions, score = 1.0.
    # Q: {total, revenue} A: {total, revenue} → F1 = 2*2/(2+2) = 1.0
    assert question_answer_lexical_f1("total revenue", "total revenue") == pytest.approx(1.0)


def test_question_answer_lexical_f1_empty_inputs():
    assert question_answer_lexical_f1("", "some answer") == 0.0
    assert question_answer_lexical_f1("What is the revenue?", "") == 0.0
