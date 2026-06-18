from src.pipeline2.metrics.answer_metrics import (
    compute_answer_metrics,
    compute_rouge_1,
    compute_rouge_l,
    is_abstention,
)


def test_answer_metrics_exact_match():
    metrics = compute_answer_metrics("100", "100")

    assert metrics["exact_match"] == 1.0
    assert metrics["literal_exact_match"] == 1.0
    assert metrics["canonical_exact_match"] == 1.0
    assert metrics["answer_match_status"] == "match"
    assert metrics["non_empty_answer_rate"] == 1.0


def test_answer_metrics_mismatch():
    metrics = compute_answer_metrics("100", "200")

    assert metrics["exact_match"] == 0.0
    assert metrics["answer_match_status"] == "mismatch"


def test_answer_metrics_no_gold():
    metrics = compute_answer_metrics("100", "")

    assert metrics["exact_match"] == 0.0
    assert metrics["answer_match_status"] == "no_gold"


def test_german_canonical_exact_match_strips_trailing_punct():
    metrics = compute_answer_metrics("Ja.", "Ja")

    assert metrics["german_canonical_exact_match"] == 1.0
    assert metrics["exact_match"] == 0.0  # literal doesn't strip punct


def test_umlaut_expanded_exact_match():
    metrics = compute_answer_metrics("ue", "ü")

    assert metrics["umlaut_expanded_exact_match"] == 1.0


def test_german_abstention_patterns():
    assert is_abstention("UNBEKANNT")
    assert is_abstention("Nicht Verfügbar")
    assert is_abstention("k.a.")
    assert not is_abstention("Ja")


def test_rouge_1_unigram_overlap():
    # "die" and "das" overlap → F1 = 2*1/2*1/2 / (1/2 + 1/2) = 0.5
    score = compute_rouge_1("die katze", "die das")
    assert 0.0 < score <= 1.0


def test_rouge_l_exact():
    assert compute_rouge_l("hello world", "hello world") == 1.0


def test_rouge_l_no_overlap():
    assert compute_rouge_l("abc", "xyz") == 0.0
