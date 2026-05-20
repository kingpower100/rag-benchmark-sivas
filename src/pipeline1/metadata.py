from __future__ import annotations

import re
from typing import Any


TREASURY_METADATA_SCHEMA_VERSION = "treasury_v1"
_TREASURY_FILENAME_RE = re.compile(r"^(treasury_bulletin_(?P<year>\d{4})_(?P<month>\d{2}))\.txt$", re.IGNORECASE)

CANONICAL_METADATA_FIELDS = (
    "company_name",
    "company_symbol",
    "report_year",
    "report_period",
    "page_number",
    "sector",
    "industry",
    "file_name",
    "source_dataset",
    "original_context_id",
)


def parse_treasury_filename(filename: str) -> dict[str, Any]:
    """Extract stable Treasury Bulletin metadata from a source filename."""
    name = str(filename)
    stem = name.rsplit(".", 1)[0] if "." in name else name
    metadata: dict[str, Any] = {
        "source_file": name,
        "file_name": name,
        "source_id": stem,
        "year": None,
        "month": None,
        "report_year": None,
        "source_dataset": "officeqa",
        "metadata_schema_version": TREASURY_METADATA_SCHEMA_VERSION,
    }
    match = _TREASURY_FILENAME_RE.fullmatch(name)
    if not match:
        return metadata
    year = int(match.group("year"))
    month = match.group("month")
    metadata.update(
        {
            "source_id": match.group(1),
            "year": year,
            "month": month,
            "report_year": year,
        }
    )
    return metadata


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text.casefold() if text else None


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = normalize_optional_string(value)
    if text is None:
        return None
    match = re.fullmatch(r"[+-]?\d+", text)
    return int(text) if match else None


def normalize_metadata(raw: dict[str, Any] | None, original_context_id: str | None = None) -> dict[str, Any]:
    source = dict(raw or {})
    normalized = dict(source)
    for field in ("company_name", "company_symbol", "report_period", "sector", "industry", "file_name", "source_dataset"):
        normalized[field] = normalize_optional_string(source.get(field))
    normalized["report_year"] = safe_int(source.get("report_year"))
    normalized["page_number"] = safe_int(source.get("page_number"))
    normalized["original_context_id"] = normalize_optional_string(
        source.get("original_context_id", original_context_id)
    )
    return normalized


def canonical_chunk_metadata(raw: dict[str, Any] | None, original_context_id: str | None = None) -> dict[str, Any]:
    normalized = normalize_metadata(raw, original_context_id)
    return {field: normalized.get(field) for field in CANONICAL_METADATA_FIELDS}
