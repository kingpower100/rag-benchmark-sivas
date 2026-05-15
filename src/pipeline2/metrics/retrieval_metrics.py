from __future__ import annotations

import math
from src.pipeline1.retrieval.metadata import QueryMetadata, extract_query_metadata, metadata_matches


DEFAULT_RETRIEVAL_KS = [1, 3, 5]


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
    output: dict[str, float | None] = {
        "duplicate_context_rate": duplicate_context_rate(retrieved_ids),
        "raw_duplicate_rate": None if raw_retrieved_ids is None else duplicate_context_rate(raw_retrieved_ids),
    }
    for k in ks or DEFAULT_RETRIEVAL_KS:
        output.update(_metrics_at_k(retrieved_ids, gold_ids, k))
    return output


def _metrics_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> dict[str, float | None]:
    ranked = _dedupe_preserving_order(str(item) for item in retrieved_ids if item is not None)[:k]
    gold_set = {str(item) for item in gold_ids if item is not None}
    overlap_set = set(ranked) & gold_set
    overlap = len(overlap_set)

    hit = 1.0 if overlap > 0 else 0.0
    recall = None if not gold_set else overlap / len(gold_set)
    context_precision = overlap / k
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
    }


def duplicate_context_rate(retrieved_ids: list[str]) -> float:
    ids = [str(item) for item in retrieved_ids if item is not None]
    if not ids:
        return 0.0
    return (len(ids) - len(set(ids))) / len(ids)


def compute_metadata_match_metrics(
    question: str,
    retrieved_metadata: list[dict],
    query_metadata: dict | None = None,
) -> dict[str, float | None]:
    query = _query_metadata_from_payload(query_metadata) if query_metadata else extract_query_metadata(question, retrieved_metadata)
    company_values = []
    year_values = []
    metadata_values = []
    for metadata in retrieved_metadata:
        matches = metadata_matches(metadata or {}, query)
        if matches["company_match"] is not None:
            company_values.append(float(bool(matches["company_match"])))
        if matches["year_match"] is not None:
            year_values.append(float(bool(matches["year_match"])))
        if matches["metadata_match"] is not None:
            metadata_values.append(float(bool(matches["metadata_match"])))
    return {
        "metadata_match_rate": None if not metadata_values else sum(metadata_values) / len(metadata_values),
        "company_match_rate": None if not company_values else sum(company_values) / len(company_values),
        "year_match_rate": None if not year_values else sum(year_values) / len(year_values),
    }


def _query_metadata_from_payload(payload: dict) -> QueryMetadata:
    return QueryMetadata(
        company_names=frozenset(payload.get("company_names") or []),
        company_symbols=frozenset(payload.get("company_symbols") or []),
        years=frozenset(int(value) for value in (payload.get("years") or [])),
        report_periods=frozenset(payload.get("report_periods") or []),
        file_names=frozenset(payload.get("file_names") or []),
        source_datasets=frozenset(payload.get("source_datasets") or []),
    )


def _ndcg_at_k(ranked: list[str], gold_set: set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(ranked[:k], start=1) if item in gold_set)
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
