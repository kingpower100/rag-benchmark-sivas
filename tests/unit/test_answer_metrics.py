from src.pipeline2.metrics.answer_metrics import (
    compute_answer_metrics,
    is_abstention,
)


def test_answer_metrics_match_status_and_non_empty():
    metrics = compute_answer_metrics("100", "100")

    assert metrics["answer_match_status"] == "match"
    assert metrics["non_empty_answer_rate"] == 1.0


def test_answer_metrics_mismatch():
    metrics = compute_answer_metrics("100", "200")

    assert metrics["answer_match_status"] == "mismatch"


def test_answer_metrics_no_gold():
    metrics = compute_answer_metrics("100", "")

    assert metrics["answer_match_status"] == "no_gold"


def test_german_abstention_patterns():
    assert is_abstention("UNBEKANNT")
    assert is_abstention("Nicht Verfügbar")
    assert is_abstention("k.a.")
    assert not is_abstention("Ja")
