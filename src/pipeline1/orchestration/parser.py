from __future__ import annotations

import json
from difflib import get_close_matches
from typing import Any


def parse_orchestration_response(raw: str, original_question: str, categories: list[str]) -> dict[str, str | float]:
    payload = _loads_json_object(raw)
    answer_fields = {"answer", "generated_answer", "final_answer", "response"}
    unsafe = answer_fields & set(payload)
    if unsafe:
        raise ValueError(f"Orchestration response must not contain answer fields: {', '.join(sorted(unsafe))}")
    cleaned_question = str(payload.get("cleaned_question") or original_question).strip() or original_question
    category = _resolve_category(str(payload.get("detected_category") or "").strip(), categories)
    confidence = _clamp_float(payload.get("category_confidence"), 0.0, 1.0)
    if category == "":
        confidence = 0.0
    return {
        "cleaned_question": cleaned_question,
        "detected_category": category,
        "category_confidence": confidence,
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


def _resolve_category(value: str, categories: list[str]) -> str:
    if not categories:
        return ""
    by_casefold = {category.casefold(): category for category in categories}
    if value.casefold() in by_casefold:
        return by_casefold[value.casefold()]
    matches = get_close_matches(value.casefold(), list(by_casefold), n=1, cutoff=0.82)
    return by_casefold[matches[0]] if matches else ""


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, number))
