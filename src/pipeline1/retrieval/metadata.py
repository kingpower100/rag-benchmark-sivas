from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from src.pipeline1.metadata import normalize_text, safe_int


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_QUARTER_RE = re.compile(r"\bq([1-4])\b|\b(first|second|third|fourth)\s+quarter\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryMetadata:
    company_names: frozenset[str] = frozenset()
    company_symbols: frozenset[str] = frozenset()
    years: frozenset[int] = frozenset()
    report_periods: frozenset[str] = frozenset()
    file_names: frozenset[str] = frozenset()
    source_datasets: frozenset[str] = frozenset()

    @property
    def has_any_signal(self) -> bool:
        return any(
            (
                self.company_names,
                self.company_symbols,
                self.years,
                self.report_periods,
                self.file_names,
                self.source_datasets,
            )
        )


def extract_query_metadata(question: str, known_metadata: Iterable[dict[str, Any]] = ()) -> QueryMetadata:
    question_norm = normalize_text(question) or ""
    years = frozenset(int(match.group()) for match in _YEAR_RE.finditer(question_norm))
    periods = set()
    for match in _QUARTER_RE.finditer(question_norm):
        q_digit, q_word = match.groups()
        if q_digit:
            periods.add(f"q{q_digit}")
        elif q_word:
            periods.add({"first": "q1", "second": "q2", "third": "q3", "fourth": "q4"}[q_word.casefold()])

    candidates = list(known_metadata)
    return QueryMetadata(
        company_names=_mentioned_values(question_norm, candidates, "company_name"),
        company_symbols=_mentioned_values(question_norm, candidates, "company_symbol"),
        years=years,
        report_periods=frozenset(periods),
        file_names=_mentioned_values(question_norm, candidates, "file_name"),
        source_datasets=_mentioned_values(question_norm, candidates, "source_dataset"),
    )


def metadata_boost(
    metadata: dict[str, Any],
    query: QueryMetadata,
    company_weight: float,
    year_weight: float,
    symbol_weight: float,
    file_name_weight: float = 0.0,
) -> float:
    boost = 0.0
    company = normalize_text(metadata.get("company_name"))
    symbol = normalize_text(metadata.get("company_symbol"))
    file_name = normalize_text(metadata.get("file_name"))
    year = safe_int(metadata.get("report_year"))
    if company and company in query.company_names:
        boost += company_weight
    if symbol and symbol in query.company_symbols:
        boost += symbol_weight
    if year is not None and year in query.years:
        boost += year_weight
    if file_name and file_name in query.file_names:
        boost += file_name_weight
    return boost


def metadata_matches(metadata: dict[str, Any], query: QueryMetadata) -> dict[str, bool | None]:
    company = normalize_text(metadata.get("company_name"))
    year = safe_int(metadata.get("report_year"))
    company_match = None if not query.company_names else bool(company and company in query.company_names)
    year_match = None if not query.years else bool(year is not None and year in query.years)
    checks = [value for value in (company_match, year_match) if value is not None]
    return {
        "company_match": company_match,
        "year_match": year_match,
        "metadata_match": None if not checks else all(checks),
    }


def filter_candidates_by_metadata(items: list, query: QueryMetadata, strict: bool) -> list:
    if not query.has_any_signal:
        return items
    filtered = [item for item in items if _item_matches_filter(item.metadata, query, strict)]
    return filtered or items


def _mentioned_values(question_norm: str, metadata_rows: list[dict[str, Any]], field: str) -> frozenset[str]:
    values = set()
    for row in metadata_rows:
        value = normalize_text(row.get(field))
        if value and value in question_norm:
            values.add(value)
    return frozenset(values)


def _item_matches_filter(metadata: dict[str, Any], query: QueryMetadata, strict: bool) -> bool:
    checks = []
    company = normalize_text(metadata.get("company_name"))
    symbol = normalize_text(metadata.get("company_symbol"))
    dataset = normalize_text(metadata.get("source_dataset"))
    year = safe_int(metadata.get("report_year"))
    if query.company_names:
        checks.append(bool(company and company in query.company_names))
    if query.company_symbols:
        checks.append(bool(symbol and symbol in query.company_symbols))
    if query.years:
        checks.append(bool(year is not None and year in query.years))
    if query.source_datasets:
        checks.append(bool(dataset and dataset in query.source_datasets))
    if not checks:
        return True
    return all(checks) if strict else any(checks)
