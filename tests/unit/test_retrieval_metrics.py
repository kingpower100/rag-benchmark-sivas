from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks


def test_true_ranked_top_k_metrics_do_not_dedupe_before_slicing():
    metrics = compute_retrieval_metrics_for_ks(["A", "A", "A", "GOLD"], ["GOLD"], [3, 4])

    assert metrics["hit_at_3"] == 0.0
    assert metrics["recall_at_3"] == 0.0
    assert metrics["mrr_at_3"] == 0.0
    assert metrics["hit_at_4"] == 1.0
    assert metrics["mrr_at_4"] == 0.25
