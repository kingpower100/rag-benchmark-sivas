import pytest

from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics, normalize_source_id


def test_single_gold_hit_recall_precision_and_mrr_case_a():
    metrics = compute_retrieval_metrics(["A", "B", "C"], ["A"], k=3)

    assert metrics == {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_3": 1.0,
        "recall_at_3": 1.0,
        "mrr_at_3": 1.0,
        "context_precision_at_3": 1 / 3,
        "ndcg_at_3": 1.0,
    }


def test_multi_gold_full_recall_with_second_rank_first_hit_case_b():
    metrics = compute_retrieval_metrics(["X", "A", "B"], ["A", "B"], k=3)

    assert metrics == {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_3": 1.0,
        "recall_at_3": 1.0,
        "mrr_at_3": 1 / 2,
        "context_precision_at_3": 2 / 3,
        "ndcg_at_3": pytest.approx((1 / 1.584962500721156 + 1 / 2) / (1 + 1 / 1.584962500721156)),
    }


def test_multi_gold_partial_recall_case_c():
    metrics = compute_retrieval_metrics(["X", "A"], ["A", "B"], k=2)

    assert metrics == {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_2": 1.0,
        "recall_at_2": 0.5,
        "mrr_at_2": 1 / 2,
        "context_precision_at_2": 0.5,
        "ndcg_at_2": pytest.approx((1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)),
    }


def test_no_overlap_case_d():
    metrics = compute_retrieval_metrics(["X", "Y"], ["A", "B"], k=2)

    assert metrics == {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_2": 0.0,
        "recall_at_2": 0.0,
        "mrr_at_2": 0.0,
        "context_precision_at_2": 0.0,
        "ndcg_at_2": 0.0,
    }


def test_retrieval_metric_formulas_at_dynamic_k():
    metrics = compute_retrieval_metrics(["ctx_a", "ctx_b", "ctx_c"], ["ctx_b", "ctx_x"], k=5)

    assert metrics == {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_5": 1.0,
        "recall_at_5": 0.5,
        "mrr_at_5": 0.5,
        "context_precision_at_5": 1 / 5,
        "ndcg_at_5": pytest.approx((1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)),
    }


def test_retrieval_metric_names_use_configured_k():
    metrics = compute_retrieval_metrics(["ctx_a", "ctx_b"], ["ctx_c"], k=2)

    assert set(metrics) == {"duplicate_context_rate", "raw_duplicate_rate", "hit_at_2", "recall_at_2", "context_precision_at_2", "mrr_at_2", "ndcg_at_2"}
    assert all(value == 0.0 for key, value in metrics.items() if key != "raw_duplicate_rate")
    assert metrics["raw_duplicate_rate"] is None


def test_recall_is_none_when_no_gold_contexts():
    metrics = compute_retrieval_metrics(["ctx_a"], [], k=5)

    assert metrics["hit_at_5"] == 0.0
    assert metrics["recall_at_5"] is None
    assert metrics["context_precision_at_5"] == 0.0
    assert metrics["mrr_at_5"] == 0.0
    assert metrics["ndcg_at_5"] == 0.0


def test_retrieved_ids_are_deduplicated_after_top_k():
    metrics = compute_retrieval_metrics(["ctx_a", "ctx_a", "ctx_b"], ["ctx_a", "ctx_b"], k=3)

    assert metrics["hit_at_3"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["context_precision_at_3"] == 2 / 3
    assert metrics["mrr_at_3"] == 1.0
    assert metrics["duplicate_context_rate"] == 1 / 3
    assert metrics["ndcg_at_3"] == 1.0


def test_multi_k_retrieval_metrics_are_emitted():
    from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks

    metrics = compute_retrieval_metrics_for_ks(["A", "B", "C"], ["B"], [1, 3, 5])

    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr_at_3"] == 0.5
    assert metrics["context_precision_at_5"] == 1 / 5
    assert metrics["ndcg_at_5"] == pytest.approx(1 / 1.584962500721156)


def test_raw_duplicate_rate_uses_pre_dedup_candidates():
    metrics = compute_retrieval_metrics(["ctx_a", "ctx_b"], ["ctx_a"], k=2, raw_retrieved_ids=["ctx_a", "ctx_a", "ctx_b"])

    assert metrics["duplicate_context_rate"] == 0.0
    assert metrics["raw_duplicate_rate"] == 1 / 3


def test_source_id_normalization_matches_txt_and_non_txt_forms():
    assert normalize_source_id("treasury_bulletin_1941_01.txt") == "treasury_bulletin_1941_01"
    assert normalize_source_id("treasury_bulletin_1941_01") == "treasury_bulletin_1941_01"

    metrics = compute_retrieval_metrics(["treasury_bulletin_1941_01"], ["treasury_bulletin_1941_01.txt"], k=1)

    assert metrics["hit_at_1"] == 1.0
    assert metrics["recall_at_1"] == 1.0
