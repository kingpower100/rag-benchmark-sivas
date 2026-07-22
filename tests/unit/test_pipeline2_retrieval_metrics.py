import pytest

from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics, normalize_source_id


def test_single_gold_hit_recall_precision_and_mrr_case_a():
    metrics = compute_retrieval_metrics(["A", "B", "C"], ["A"], k=3)

    _assert_metric_subset(metrics, {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_3": 1.0,
        "recall_at_3": 1.0,
        "mrr_at_3": 1.0,
        "context_precision_at_3": 1 / 3,
        "ndcg_at_3": 1.0,
    })


def test_multi_gold_full_recall_with_second_rank_first_hit_case_b():
    metrics = compute_retrieval_metrics(["X", "A", "B"], ["A", "B"], k=3)

    _assert_metric_subset(metrics, {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_3": 1.0,
        "recall_at_3": 1.0,
        "mrr_at_3": 1 / 2,
        "context_precision_at_3": 2 / 3,
        "ndcg_at_3": pytest.approx((1 / 1.584962500721156 + 1 / 2) / (1 + 1 / 1.584962500721156)),
    })


def test_multi_gold_partial_recall_case_c():
    metrics = compute_retrieval_metrics(["X", "A"], ["A", "B"], k=2)

    _assert_metric_subset(metrics, {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_2": 1.0,
        "recall_at_2": 0.5,
        "mrr_at_2": 1 / 2,
        "context_precision_at_2": 0.5,
        "ndcg_at_2": pytest.approx((1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)),
    })


def test_no_overlap_case_d():
    metrics = compute_retrieval_metrics(["X", "Y"], ["A", "B"], k=2)

    _assert_metric_subset(metrics, {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_2": 0.0,
        "recall_at_2": 0.0,
        "mrr_at_2": 0.0,
        "context_precision_at_2": 0.0,
        "ndcg_at_2": 0.0,
    })


def test_retrieval_metric_formulas_at_dynamic_k():
    metrics = compute_retrieval_metrics(["ctx_a", "ctx_b", "ctx_c"], ["ctx_b", "ctx_x"], k=5)

    _assert_metric_subset(metrics, {
        "duplicate_context_rate": 0.0,
        "raw_duplicate_rate": None,
        "hit_at_5": 1.0,
        "recall_at_5": 0.5,
        "mrr_at_5": 0.5,
        "context_precision_at_5": 1 / 3,
        "ndcg_at_5": pytest.approx((1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)),
    })


def test_retrieval_metric_names_use_configured_k():
    metrics = compute_retrieval_metrics(["ctx_a", "ctx_b"], ["ctx_c"], k=2)

    assert {"duplicate_context_rate", "raw_duplicate_rate", "hit_at_2", "recall_at_2", "context_precision_at_2", "mrr_at_2", "ndcg_at_2"} <= set(metrics)
    assert metrics["hit_at_2"] == 0.0
    assert metrics["recall_at_2"] == 0.0
    assert metrics["context_precision_at_2"] == 0.0
    assert metrics["mrr_at_2"] == 0.0
    assert metrics["ndcg_at_2"] == 0.0
    assert metrics["raw_retrieved_count"] == 2
    assert metrics["unique_retrieved_document_count"] == 2
    assert metrics["duplicate_document_count"] == 0
    assert metrics["raw_duplicate_rate"] is None


def test_recall_is_none_when_no_gold_contexts():
    metrics = compute_retrieval_metrics(["ctx_a"], [], k=5)

    assert metrics["hit_at_5"] == 0.0
    assert metrics["recall_at_5"] is None
    assert metrics["context_precision_at_5"] == 0.0
    assert metrics["mrr_at_5"] == 0.0
    assert metrics["ndcg_at_5"] == 0.0


def test_retrieval_metrics_use_raw_ranking_and_report_duplicates():
    metrics = compute_retrieval_metrics(["ctx_a", "ctx_a", "ctx_b"], ["ctx_a", "ctx_b"], k=3)

    assert metrics["hit_at_3"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["context_precision_at_3"] == 2 / 3
    assert metrics["mrr_at_3"] == 1.0
    assert metrics["duplicate_context_rate"] == 1 / 3
    assert metrics["duplicate_document_count"] == 1
    assert metrics["duplicate_document_rate"] == 1 / 3
    assert metrics["duplicate_count_at_3"] == 1
    assert metrics["duplicate_rate_at_3"] == 1 / 3
    assert metrics["deduped_recall_at_3"] == 1.0


def test_multi_k_retrieval_metrics_are_emitted():
    from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks

    metrics = compute_retrieval_metrics_for_ks(["A", "B", "C"], ["B"], [1, 3, 5])

    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr_at_3"] == 0.5
    assert metrics["context_precision_at_5"] == 1 / 3
    assert metrics["ndcg_at_5"] == pytest.approx(1 / 1.584962500721156)


def test_raw_duplicate_rate_uses_pre_dedup_candidates():
    metrics = compute_retrieval_metrics(["ctx_a", "ctx_b"], ["ctx_a"], k=2, raw_retrieved_ids=["ctx_a", "ctx_a", "ctx_b"])

    assert metrics["duplicate_context_rate"] == 0.0
    assert metrics["raw_duplicate_rate"] == 1 / 3


def test_retrieval_metrics_do_not_dedupe_before_slicing_top_k():
    from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks

    metrics = compute_retrieval_metrics_for_ks(["A", "A", "A", "GOLD"], ["GOLD"], [3, 4])

    assert metrics["hit_at_3"] == 0.0
    assert metrics["recall_at_3"] == 0.0
    assert metrics["mrr_at_3"] == 0.0
    assert metrics["ndcg_at_3"] == 0.0
    assert metrics["hit_at_4"] == 1.0
    assert metrics["mrr_at_4"] == 0.25
    assert metrics["deduped_hit_at_3"] == 1.0
    assert metrics["deduped_mrr_at_3"] == 0.5
    assert metrics["duplicate_document_count"] == 2
    assert metrics["duplicate_document_rate"] == 0.5
    assert metrics["duplicate_count_at_3"] == 2
    assert metrics["duplicate_rate_at_3"] == 2 / 3


def test_ndcg_preserves_rank_when_duplicate_precedes_relevant_result():
    metrics = compute_retrieval_metrics(["A", "A", "GOLD"], ["GOLD"], k=3)

    assert metrics["hit_at_3"] == 1.0
    assert metrics["mrr_at_3"] == 1 / 3
    assert metrics["ndcg_at_3"] == pytest.approx(0.5)


def test_ndcg_duplicate_relevant_document_gets_zero_additional_gain():
    metrics = compute_retrieval_metrics(["GOLD", "GOLD", "OTHER"], ["GOLD"], k=3)

    assert metrics["ndcg_at_3"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["context_precision_at_3"] == 1 / 3


def test_ndcg_multiple_relevant_documents_keep_original_ranks_with_duplicate_interruption():
    metrics = compute_retrieval_metrics(["GOLD_A", "GOLD_A", "GOLD_B"], ["GOLD_A", "GOLD_B"], k=3)
    expected_dcg = 1.0 + 1.0 / 2.0
    expected_idcg = 1.0 + 1.0 / 1.584962500721156

    assert metrics["ndcg_at_3"] == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_no_relevant_results_is_zero():
    metrics = compute_retrieval_metrics(["A", "B", "C"], ["GOLD"], k=3)

    assert metrics["ndcg_at_3"] == 0.0


def test_ndcg_empty_identifiers_cannot_match():
    from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks

    metrics = compute_retrieval_metrics_for_ks(["", None, "GOLD"], ["", "GOLD"], [1, 3])  # type: ignore[list-item]

    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["ndcg_at_3"] == pytest.approx(0.5)


def test_ndcg_handles_k_smaller_and_larger_than_retrieved_list():
    from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks

    metrics = compute_retrieval_metrics_for_ks(["A", "GOLD"], ["GOLD"], [1, 5])

    assert metrics["ndcg_at_1"] == 0.0
    assert metrics["ndcg_at_5"] == pytest.approx(1 / 1.584962500721156)


def test_source_id_normalization_matches_txt_and_non_txt_forms():
    assert normalize_source_id("sivas_manual_01.md") == "sivas_manual_01.md"
    assert normalize_source_id("sivas_manual_01") == "sivas_manual_01"

    metrics = compute_retrieval_metrics(["sivas_manual_01"], ["sivas_manual_01.md"], k=1)

    assert metrics["hit_at_1"] == 0.0
    assert metrics["recall_at_1"] == 0.0


def test_source_id_normalization_strips_known_chunk_suffixes_to_document_id():
    expected = "sivas_manual_02.md"

    assert normalize_source_id("sivas_manual_02.md_chunk_17") == expected
    assert normalize_source_id("sivas_manual_02.md::chunk_17") == expected
    assert normalize_source_id("sivas_manual_02.md#chunk=17") == expected

    metrics = compute_retrieval_metrics(
        ["sivas_manual_02.md_chunk_17", "sivas_manual_02.md_chunk_18"],
        ["sivas_manual_02.md"],
        k=2,
    )

    assert metrics["hit_at_2"] == 1.0
    assert metrics["recall_at_2"] == 1.0
    assert metrics["context_precision_at_2"] == 0.5
    assert metrics["duplicate_document_count"] == 1


def _assert_metric_subset(metrics, expected):
    for key, value in expected.items():
        assert metrics[key] == value
