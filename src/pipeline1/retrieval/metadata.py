from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from src.pipeline1.metadata import normalize_text, safe_int


_YEAR_RE = re.compile(r"\b(?:19[3-9]\d|20[0-2]\d|2030)\b")
_QUARTER_RE = re.compile(r"\bq([1-4])\b|\b(first|second|third|fourth)\s+quarter\b", re.IGNORECASE)
_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_NAME_RE = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\b", re.IGNORECASE)
_MONTH_NUMBER_RE = re.compile(r"\b(?:month|mo\.?)\s*(?:no\.?\s*)?(?P<month>0?[1-9]|1[0-2])\b", re.IGNORECASE)
_FISCAL_YEAR_RE = re.compile(r"\b(?:fiscal\s+year|fy)\s*(?P<year>(?:19[3-9]\d|20[0-2]\d|2030))\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryMetadata:
    company_names: frozenset[str] = frozenset()
    company_symbols: frozenset[str] = frozenset()
    years: frozenset[int] = frozenset()
    months: frozenset[int] = frozenset()
    year_months: frozenset[str] = frozenset()
    fiscal_years: frozenset[int] = frozenset()
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
                self.months,
                self.year_months,
                self.fiscal_years,
                self.report_periods,
                self.file_names,
                self.source_datasets,
            )
        )


def extract_query_metadata(question: str, known_metadata: Iterable[dict[str, Any]] = ()) -> QueryMetadata:
    question_norm = normalize_text(question) or ""
    years = frozenset(int(match.group()) for match in _YEAR_RE.finditer(question_norm))
    fiscal_years = frozenset(int(match.group("year")) for match in _FISCAL_YEAR_RE.finditer(question_norm))
    months = _extract_months(question_norm)
    year_months = frozenset(
        f"{year}_{month:02d}"
        for year in years
        for month in months
        if _month_near_year(question_norm, month, year)
    )
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
        months=frozenset(months),
        year_months=year_months,
        fiscal_years=fiscal_years,
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
    month_weight: float = 0.0,
    year_month_weight: float = 0.0,
    wrong_year_penalty: float = 0.0,
    file_name_weight: float = 0.0,
) -> float:
    return sum(metadata_boost_components(
        metadata,
        query,
        company_weight=company_weight,
        year_weight=year_weight,
        month_weight=month_weight,
        year_month_weight=year_month_weight,
        wrong_year_penalty=wrong_year_penalty,
        symbol_weight=symbol_weight,
        file_name_weight=file_name_weight,
    ).values())


def metadata_boost_components(
    metadata: dict[str, Any],
    query: QueryMetadata,
    company_weight: float,
    year_weight: float,
    month_weight: float = 0.0,
    year_month_weight: float = 0.0,
    wrong_year_penalty: float = 0.0,
    symbol_weight: float = 0.2,
    file_name_weight: float = 0.0,
) -> dict[str, float]:
    components: dict[str, float] = {}
    boost = 0.0
    company = normalize_text(metadata.get("company_name"))
    symbol = normalize_text(metadata.get("company_symbol"))
    file_name = normalize_text(metadata.get("file_name"))
    year = safe_int(metadata.get("report_year"))
    month = safe_int(metadata.get("month"))
    year_month = str(metadata.get("year_month") or "")
    year_signals = query.years | query.fiscal_years
    if company and company in query.company_names:
        components["company"] = company_weight
    if symbol and symbol in query.company_symbols:
        components["symbol"] = symbol_weight
    if year is not None and year in year_signals:
        components["year"] = year_weight
    elif year is not None and year_signals and wrong_year_penalty:
        components["wrong_year_penalty"] = wrong_year_penalty
    if month is not None and month in query.months:
        components["month"] = month_weight
    if year_month and year_month in query.year_months:
        components["year_month"] = year_month_weight
    if file_name and file_name in query.file_names:
        components["file_name"] = file_name_weight
    return {key: value for key, value in components.items() if value}


def metadata_matches(metadata: dict[str, Any], query: QueryMetadata) -> dict[str, bool | None]:
    company = normalize_text(metadata.get("company_name"))
    year = safe_int(metadata.get("report_year"))
    month = safe_int(metadata.get("month"))
    year_month = metadata.get("year_month")
    year_signals = query.years | query.fiscal_years
    company_match = None if not query.company_names else bool(company and company in query.company_names)
    year_match = None if not year_signals else bool(year is not None and year in year_signals)
    month_match = None if not query.months else bool(month is not None and month in query.months)
    year_month_match = None if not query.year_months else bool(year_month and year_month in query.year_months)
    checks = [value for value in (company_match, year_match, month_match, year_month_match) if value is not None]
    return {
        "company_match": company_match,
        "year_match": year_match,
        "month_match": month_match,
        "year_month_match": year_month_match,
        "metadata_match": None if not checks else all(checks),
    }


def filter_candidates_by_metadata(
    items: list,
    query: QueryMetadata,
    strict: bool,
    strict_year_match: bool = False,
    strict_year_month_match: bool = False,
) -> list:
    return filter_candidates_by_metadata_with_diagnostics(
        items,
        query,
        strict,
        strict_year_match,
        strict_year_month_match,
    )[0]


def filter_candidates_by_metadata_with_diagnostics(
    items: list,
    query: QueryMetadata,
    strict: bool,
    strict_year_match: bool = False,
    strict_year_month_match: bool = False,
) -> tuple[list, dict[str, Any]]:
    diagnostics = {
        "candidates_before_filter": len(items),
        "candidates_after_filter": len(items),
        "filter_fallback": False,
    }
    if not query.has_any_signal:
        return items, diagnostics
    filtered = [
        item for item in items
        if _item_matches_filter(item.metadata, query, strict, strict_year_match, strict_year_month_match)
    ]
    if not filtered:
        diagnostics["filter_fallback"] = True
        return items, diagnostics
    diagnostics["candidates_after_filter"] = len(filtered)
    return filtered, diagnostics


def _mentioned_values(question_norm: str, metadata_rows: list[dict[str, Any]], field: str) -> frozenset[str]:
    values = set()
    for row in metadata_rows:
        value = normalize_text(row.get(field))
        if value and value in question_norm:
            values.add(value)
    return frozenset(values)


def _item_matches_filter(
    metadata: dict[str, Any],
    query: QueryMetadata,
    strict: bool,
    strict_year_match: bool = False,
    strict_year_month_match: bool = False,
) -> bool:
    checks = []
    company = normalize_text(metadata.get("company_name"))
    symbol = normalize_text(metadata.get("company_symbol"))
    dataset = normalize_text(metadata.get("source_dataset"))
    year = safe_int(metadata.get("report_year"))
    year_month = metadata.get("year_month")
    year_signals = query.years | query.fiscal_years
    if strict_year_month_match and query.year_months:
        return bool(year_month and year_month in query.year_months)
    if strict_year_match and year_signals:
        return bool(year is not None and year in year_signals)
    if query.company_names:
        checks.append(bool(company and company in query.company_names))
    if query.company_symbols:
        checks.append(bool(symbol and symbol in query.company_symbols))
    if year_signals:
        checks.append(bool(year is not None and year in year_signals))
    if query.source_datasets:
        checks.append(bool(dataset and dataset in query.source_datasets))
    if not checks:
        return True
    return all(checks) if strict else any(checks)


def _extract_months(question_norm: str) -> set[int]:
    months = {_MONTHS[match.group(1).casefold().rstrip(".")] for match in _MONTH_NAME_RE.finditer(question_norm)}
    months.update(int(match.group("month")) for match in _MONTH_NUMBER_RE.finditer(question_norm))
    return {month for month in months if 1 <= month <= 12}


def _month_near_year(question_norm: str, month: int, year: int) -> bool:
    month_tokens = [name for name, value in _MONTHS.items() if value == month]
    year_text = str(year)
    for token in month_tokens:
        pattern = rf"\b{re.escape(token)}\.?\b(?:\W+\w+){{0,4}}\W+{year_text}\b|\b{year_text}\b(?:\W+\w+){{0,4}}\W+\b{re.escape(token)}\.?\b"
        if re.search(pattern, question_norm, flags=re.IGNORECASE):
            return True
    return len(_YEAR_RE.findall(question_norm)) == 1 and len(_extract_months(question_norm)) == 1
