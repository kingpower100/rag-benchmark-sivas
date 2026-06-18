import pytest

from src.pipeline2.metrics.answer_metrics import answer_relevancy_score, compute_answer_metrics, is_abstention


def test_numeric_accuracy_matches_numbers_with_commas_and_currency():
    metrics = compute_answer_metrics("Revenue was $1,250.00.", "1250")

    assert metrics["numeric_accuracy"] == 1.0
    assert metrics["strict_numeric_accuracy"] == 1.0
    assert metrics["tolerant_numeric_accuracy"] == 1.0


def test_numeric_accuracy_is_none_without_numeric_ground_truth():
    metrics = compute_answer_metrics("Paris", "Paris")

    assert metrics["numeric_accuracy"] is None


def test_numeric_accuracy_detects_mismatch():
    metrics = compute_answer_metrics("42", "41")

    assert metrics["numeric_accuracy"] == 0.0
    assert metrics["strict_numeric_accuracy"] == 0.0


def test_yes_no_normalization_matches_one_zero():
    assert compute_answer_metrics("yes", "1")["numeric_accuracy"] == 1.0
    assert compute_answer_metrics("no", "0")["numeric_accuracy"] == 1.0


def test_percentage_decimal_normalization_matches_ratio():
    metrics = compute_answer_metrics("65.7%", "0.657")

    assert metrics["numeric_accuracy"] == 1.0
    assert metrics["generated_number"] == 0.657
    assert metrics["gold_number"] == 0.657


def test_numeric_debug_fields_explain_mismatch():
    metrics = compute_answer_metrics("2 million", "1000000")

    assert metrics["numeric_accuracy"] == 0.0
    assert metrics["strict_numeric_accuracy"] == 0.0
    assert metrics["tolerant_numeric_accuracy"] == 0.0
    assert metrics["generated_number"] == 2000000.0
    assert metrics["gold_number"] == 1000000.0
    assert metrics["absolute_error"] == 1000000.0
    assert metrics["answer_match_status"] == "mismatch"


def test_canonical_exact_match_normalizes_numeric_formatting_and_yes_no_variants():
    metrics = compute_answer_metrics("$1,250.00", "1250")

    assert metrics["literal_exact_match"] == 0.0
    assert metrics["exact_match"] == 0.0
    assert metrics["canonical_exact_match"] == 1.0
    assert compute_answer_metrics("yes", "1")["canonical_exact_match"] == 1.0


def test_strict_and_tolerant_numeric_metrics_are_separate():
    exact = compute_answer_metrics("2602", "2602")
    tolerant_only = compute_answer_metrics("2603", "2602")
    scaled = compute_answer_metrics("2.602 billion", "2602 million")
    empty = compute_answer_metrics("", "2602")

    assert exact["strict_numeric_accuracy"] == 1.0
    assert exact["tolerant_numeric_accuracy"] == 1.0
    assert tolerant_only["strict_numeric_accuracy"] == 0.0
    assert tolerant_only["tolerant_numeric_accuracy"] == 1.0
    assert tolerant_only["numeric_accuracy"] == 1.0
    assert scaled["strict_numeric_accuracy"] == 1.0
    assert empty["strict_numeric_accuracy"] == 0.0
    assert empty["tolerant_numeric_accuracy"] == 0.0


def test_literal_exact_match_uses_minimal_text_normalization_only():
    assert compute_answer_metrics("2602", "2602")["literal_exact_match"] == 1.0
    assert compute_answer_metrics(" 2602 ", "2602")["literal_exact_match"] == 1.0
    assert compute_answer_metrics("2.602 billion", "2602 million")["literal_exact_match"] == 0.0
    assert compute_answer_metrics("Revenue was 2602", "2602")["literal_exact_match"] == 0.0


def test_relative_error_and_numeric_parse_success_are_reported():
    metrics = compute_answer_metrics("90", "100")

    assert metrics["relative_error"] == pytest.approx(0.1)
    assert metrics["numeric_parse_success"] == 1.0


def test_numeric_parse_success_distinguishes_parse_failures():
    metrics = compute_answer_metrics("unknown", "100")

    assert metrics["numeric_accuracy"] == 0.0
    assert metrics["numeric_parse_success"] == 0.0


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


def test_numeric_accuracy_uses_configurable_tolerance():
    strict_default = compute_answer_metrics("100.2", "100")
    loose = compute_answer_metrics("100.2", "100", numeric_tolerance_abs=0.25, numeric_tolerance_rel=0.0)

    assert strict_default["numeric_accuracy"] == 0.0
    assert loose["numeric_accuracy"] == 1.0


def test_rouge_l_scores_lcs_token_overlap_and_empty_answers():
    metrics = compute_answer_metrics("net income increased in 2020", "net income increased")

    assert metrics["rouge_l"] == pytest.approx(0.75)
    assert compute_answer_metrics("", "net income")["rouge_l"] == 0.0
    assert compute_answer_metrics("UNKNOWN", "100")["rouge_l"] == 0.0
