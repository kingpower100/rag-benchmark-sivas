from __future__ import annotations

import math
from pathlib import Path


DEFAULT_RETRIEVAL_KS = [1, 3, 5]


class EvaluationError(ValueError):
    """Raised when a computed metric violates an invariant (e.g. NDCG > 1)."""


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k: int,
    raw_retrieved_ids: list[str] | None = None,
) -> dict[str, float | None]:
    return compute_retrieval_metrics_for_ks(retrieved_ids, gold_ids, [k], raw_retrieved_ids)


def compute_retrieval_metrics_for_ks(
    retrieved_ids: list[str],
    gold_ids: list[str],
    ks: list[int] | None = None,
    raw_retrieved_ids: list[str] | None = None,
) -> dict[str, float | None]:
    normalized = _normalize_and_dedupe_documents(retrieved_ids)
    raw_normalized = [normalize_source_id(str(item)) for item in retrieved_ids if item is not None and str(item).strip()]
    # Deduplicate gold IDs before computing any metrics
    gold_ids_deduped = _dedupe_preserving_order(
        [normalize_source_id(str(g)) for g in gold_ids if g is not None and str(g).strip()]
    )
    duplicate_count = len(raw_normalized) - len(set(raw_normalized))
    output: dict[str, float | None] = {
        "raw_retrieved_count": len(raw_normalized),
        "unique_retrieved_document_count": len(normalized),
        "duplicate_document_count": duplicate_count,
        "duplicate_document_rate": duplicate_count / len(raw_normalized) if raw_normalized else 0.0,
        "duplicate_context_rate": duplicate_context_rate(retrieved_ids),
        "raw_duplicate_rate": None if raw_retrieved_ids is None else duplicate_context_rate(raw_retrieved_ids),
    }
    for k in ks or DEFAULT_RETRIEVAL_KS:
        raw_ranked_at_k = raw_normalized[:k]
        duplicate_count_at_k = len(raw_ranked_at_k) - len(set(raw_ranked_at_k))
        metrics = _metrics_at_k(raw_normalized, gold_ids_deduped, k, already_normalized=True)
        deduped_metrics = _metrics_at_k(normalized, gold_ids_deduped, k, already_normalized=True)
        metrics[f"duplicate_count_at_{k}"] = duplicate_count_at_k
        metrics[f"duplicate_rate_at_{k}"] = duplicate_count_at_k / len(raw_ranked_at_k) if raw_ranked_at_k else 0.0
        metrics[f"deduped_hit_at_{k}"] = deduped_metrics[f"hit_at_{k}"]
        metrics[f"deduped_recall_at_{k}"] = deduped_metrics[f"recall_at_{k}"]
        metrics[f"deduped_mrr_at_{k}"] = deduped_metrics[f"mrr_at_{k}"]
        metrics[f"deduped_ndcg_at_{k}"] = deduped_metrics[f"ndcg_at_{k}"]
        output.update(metrics)
        # Hard invariant: NDCG must be in [0, 1]
        ndcg_val = output.get(f"ndcg_at_{k}")
        if ndcg_val is not None and ndcg_val > 1.0 + 1e-9:
            raise EvaluationError(
                f"ndcg_at_{k}={ndcg_val:.6f} > 1.0. "
                f"Retrieved (first 10): {retrieved_ids[:10]}. "
                f"Gold (first 10): {gold_ids[:10]}."
            )
        deduped_ndcg_val = output.get(f"deduped_ndcg_at_{k}")
        if deduped_ndcg_val is not None and deduped_ndcg_val > 1.0 + 1e-9:
            raise EvaluationError(
                f"deduped_ndcg_at_{k}={deduped_ndcg_val:.6f} > 1.0."
            )
    return output


def _metrics_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k: int,
    already_normalized: bool = False,
) -> dict[str, float | None]:
    normalized = (
        [str(item) for item in retrieved_ids if item is not None and str(item).strip()]
        if already_normalized
        else _normalize_documents(retrieved_ids)
    )
    ranked = normalized[:k]
    gold_set = set(gold_ids) if already_normalized else {normalize_source_id(str(item)) for item in gold_ids if item is not None}
    overlap = len(set(ranked) & gold_set)

    hit = 1.0 if overlap > 0 else 0.0
    recall = None if not gold_set else overlap / len(gold_set)
    # Denominator is the raw ranked-list length (up to k), including any duplicate slots.
    # Numerator uses set intersection so duplicates never inflate the hit count.
    # This intentionally penalises retrieval systems that waste slots on repeated documents.
    context_precision = overlap / len(ranked) if ranked else 0.0
    reciprocal_rank = 0.0
    for idx, item in enumerate(ranked, start=1):
        if item in gold_set:
            reciprocal_rank = 1.0 / idx
            break

    return {
        f"hit_at_{k}": hit,
        f"recall_at_{k}": recall,
        f"mrr_at_{k}": reciprocal_rank,
        f"context_precision_at_{k}": context_precision,
        f"ndcg_at_{k}": _ndcg_at_k(ranked, gold_set, k),
        f"duplicate_count_at_{k}": 0,
        f"duplicate_rate_at_{k}": 0.0,
        f"deduped_hit_at_{k}": hit,
        f"deduped_recall_at_{k}": recall,
        f"deduped_mrr_at_{k}": reciprocal_rank,
        f"deduped_ndcg_at_{k}": _ndcg_at_k(ranked, gold_set, k),
    }


def duplicate_context_rate(retrieved_ids: list[str]) -> float:
    ids = [normalize_source_id(str(item)) for item in retrieved_ids if item is not None]
    if not ids:
        return 0.0
    return (len(ids) - len(set(ids))) / len(ids)


def _ndcg_at_k(ranked: list[str], gold_set: set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    # Deduplicate ranked list before DCG to prevent duplicate gold hits inflating DCG > IDCG
    seen: set[str] = set()
    deduped: list[str] = []
    for item in ranked[:k]:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(deduped, start=1) if item in gold_set)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _dedupe_preserving_order(items) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _normalize_and_dedupe_documents(items: list[str]) -> list[str]:
    return _dedupe_preserving_order(
        [normalize_source_id(str(item)) for item in items if item is not None and str(item).strip()]
    )


def _normalize_documents(items: list[str]) -> list[str]:
    return [normalize_source_id(str(item)) for item in items if item is not None and str(item).strip()]


def normalize_source_id(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    name = Path(text.replace("\\", "/")).name
    lowered = name.lower()
    for marker, ext_len in (
        (".txt_chunk_", 4), (".txt::chunk_", 4), (".txt#chunk=", 4),
        (".md_chunk_", 3), (".md::chunk_", 3), (".md#chunk=", 3),
    ):
        idx = lowered.find(marker)
        if idx >= 0:
            name = name[: idx + ext_len]
            break
    return name.casefold()
