from __future__ import annotations

import json
from typing import Any


def parse_orchestration_response(raw: str, original_question: str, categories: list[str]) -> dict[str, str | bool | None]:
    payload = _loads_json_object(raw)
    answer_fields = {"answer", "generated_answer", "final_answer", "response"}
    unsafe = answer_fields & set(payload)
    if unsafe:
        raise ValueError(f"Orchestration response must not contain answer fields: {', '.join(sorted(unsafe))}")
    cleaned_question = str(payload.get("cleaned_question") or original_question).strip() or original_question
    raw_category = str(payload.get("detected_category") or "").strip()
    category, category_validated, category_validation_reason = validate_category(raw_category, categories)
    return {
        "cleaned_question": cleaned_question,
        "detected_category": category or None,
        "category_validated": category_validated,
        "category_validation_reason": category_validation_reason,
    }


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Orchestration model did not return a JSON object.")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Orchestration model response must be a JSON object.")
    return value


def validate_category(value: str | None, categories: list[str]) -> tuple[str, bool, str | None]:
    raw_value = str(value or "").strip()
    if not raw_value:
        return "", False, "detected_category is empty or missing"
    if not categories:
        return raw_value, False, "KB category list is empty"
    by_exact = {category: category for category in categories}
    if raw_value in by_exact:
        return by_exact[raw_value], True, None
    normalized_value = _normalize_category(raw_value)
    by_normalized = {_normalize_category(category): category for category in categories}
    if normalized_value in by_normalized:
        return by_normalized[normalized_value], True, None
    return raw_value, False, "detected_category not found in KB category list"


def _normalize_category(value: str) -> str:
    return " ".join(str(value or "").strip().split()).casefold()
