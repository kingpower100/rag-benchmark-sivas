from __future__ import annotations

import re
from typing import Any


SIVAS_METADATA_SCHEMA_VERSION = "sivas_v1"
METADATA_SCHEMA_VERSION = SIVAS_METADATA_SCHEMA_VERSION

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
    "year_month",
)

SIVAS_METADATA_FIELDS = (
    "doc_id",
    "doc_key",
    "doc_name",
    "kategorie",
    "wissensart",
    "titel",
    "quellpfad",
    "sprache",
)


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
    normalized["year_month"] = normalize_optional_string(source.get("year_month"))
    normalized["page_number"] = safe_int(source.get("page_number"))
    normalized["original_context_id"] = normalize_optional_string(
        source.get("original_context_id", original_context_id)
    )
    return normalized


def canonical_chunk_metadata(raw: dict[str, Any] | None, original_context_id: str | None = None) -> dict[str, Any]:
    normalized = normalize_metadata(raw, original_context_id)
    return {field: normalized.get(field) for field in CANONICAL_METADATA_FIELDS}


def chunk_metadata(
    doc_metadata: dict[str, Any] | None,
    document_id: str,
    original_context_id: str | None,
    chunk_id: str,
    chunk_strategy: str,
    chunk_unit: str,
    **extra: Any,
) -> dict[str, Any]:
    metadata = {
        **dict(doc_metadata or {}),
        **canonical_chunk_metadata(doc_metadata, original_context_id),
    }
    for field in SIVAS_METADATA_FIELDS:
        if field in (doc_metadata or {}):
            metadata[field] = (doc_metadata or {}).get(field)
    metadata.update(
        {
            "doc_id": (doc_metadata or {}).get("doc_id", document_id),
            "doc_key": (doc_metadata or {}).get("doc_key"),
            "original_context_id": original_context_id,
            "source_file": (doc_metadata or {}).get("source_file") or (doc_metadata or {}).get("file_name"),
            "source_id": (doc_metadata or {}).get("source_id"),
            "year": (doc_metadata or {}).get("year"),
            "month": (doc_metadata or {}).get("month"),
            "chunk_id": chunk_id,
            "chunk_unit": chunk_unit,
            "chunk_strategy": chunk_strategy,
        }
    )
    metadata.update(extra)
    return metadata
