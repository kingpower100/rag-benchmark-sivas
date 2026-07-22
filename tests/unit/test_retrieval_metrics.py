from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks
import pytest


def test_raw_metrics_use_duplicate_aware_ranking_deduped_metrics_use_dedup():
    """Raw hit/mrr/recall use the original ranked list (duplicates block gold from top-3).
    Deduped variants remove duplicates first and show the gold is reachable at rank 2."""
    metrics = compute_retrieval_metrics_for_ks(["A", "A", "A", "GOLD"], ["GOLD"], [3, 4])

    # Raw metrics: duplicates fill top-3, GOLD is not in top-3
    assert metrics["hit_at_3"] == 0.0
    assert metrics["recall_at_3"] == 0.0
    assert metrics["mrr_at_3"] == 0.0
    # GOLD appears at raw position 4
    assert metrics["hit_at_4"] == 1.0
    assert metrics["mrr_at_4"] == 0.25

    # Deduped metrics: ["A", "GOLD"] after dedup, GOLD at deduped position 2
    assert metrics["deduped_hit_at_3"] == 1.0
    assert metrics["deduped_recall_at_3"] == 1.0
    assert metrics["deduped_mrr_at_3"] == 0.5


def test_document_metrics_required_duplicate_rank_case():
    metrics = compute_retrieval_metrics_for_ks(["A", "A", "GOLD"], ["GOLD"], [3])

    assert metrics["hit_at_3"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr_at_3"] == 1 / 3
    assert metrics["ndcg_at_3"] == pytest.approx(0.5)
