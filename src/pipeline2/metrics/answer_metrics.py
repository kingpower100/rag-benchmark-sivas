from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


_NUMBER_RE = re.compile(r"[-+]?\(?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_YES_NO = {"yes": Decimal("1"), "no": Decimal("0")}
_SCALE_WORDS = {
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "mm": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
}


def resolve_ground_truth_answer(row: dict[str, Any], qa_by_id: dict[str, dict[str, Any]]) -> str:
    qid = str(row.get("question_id", ""))
    qa = qa_by_id.get(qid, {})
    for key in ("ground_truth_answer", "answer", "gold_answer", "expected_answer", "program_answer", "original_answer"):
        if key in qa and qa[key] is not None:
            return str(qa[key])
    return ""


def compute_answer_metrics(generated_answer: str, ground_truth_answer: str) -> dict[str, Any]:
    generated = generated_answer or ""
    truth = ground_truth_answer or ""
    match = _compare_answers(generated, truth)
    return {
        "numeric_accuracy": match["numeric_accuracy"],
        "number_match": match["numeric_accuracy"],
        "exact_match_debug": 1.0 if generated.strip().lower() == truth.strip().lower() and truth.strip() else 0.0,
        "normalized_generated_answer": match["normalized_generated_answer"],
        "normalized_gold_answer": match["normalized_gold_answer"],
        "generated_number": _decimal_to_float(match["generated_number"]),
        "gold_number": _decimal_to_float(match["gold_number"]),
        "absolute_error": _decimal_to_float(match["absolute_error"]),
        "relative_error": _decimal_to_float(match["relative_error"]),
        "answer_match_status": match["answer_match_status"],
        "answer_coverage_rate": 1.0 if generated.strip() else 0.0,
    }


def compute_numeric_accuracy(generated_answer: str, ground_truth_answer: str) -> float | None:
    return _compare_answers(generated_answer, ground_truth_answer)["numeric_accuracy"]


def compute_number_match(generated_answer: str, ground_truth_answer: str) -> float | None:
    return compute_numeric_accuracy(generated_answer, ground_truth_answer)


def _extract_numbers(text: str) -> list[Decimal]:
    return [item["value"] for item in _extract_number_records(text)]


def _extract_number_records(text: str) -> list[dict[str, Any]]:
    values = []
    source = text or ""
    for match in _NUMBER_RE.finditer(source):
        raw = match.group(0).strip()
        negative = raw.startswith("(") and raw.endswith(")")
        raw = raw.strip("()").replace("$", "").replace(",", "")
        percent = raw.endswith("%")
        raw = raw.rstrip("%")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        if negative:
            value = -value
        if percent:
            value = value / Decimal("100")
        scale = _scale_after(source, match.end())
        values.append({"value": value * scale, "percent": percent, "scale": scale, "raw": match.group(0)})
    return values


def _close(left: Decimal, right: Decimal) -> bool:
    if left == right:
        return True
    tolerance = max(abs(right) * Decimal("0.001"), Decimal("0.0001"))
    return abs(left - right) <= tolerance


def _compare_answers(generated_answer: str, ground_truth_answer: str) -> dict[str, Any]:
    truth_value = _best_value(ground_truth_answer)
    generated_value = _best_value(generated_answer)
    normalized_truth = _normalized_text(ground_truth_answer)
    normalized_generated = _normalized_text(generated_answer)
    if truth_value is None:
        return {
            "numeric_accuracy": None,
            "normalized_generated_answer": normalized_generated,
            "normalized_gold_answer": normalized_truth,
            "generated_number": generated_value,
            "gold_number": None,
            "absolute_error": None,
            "relative_error": None,
            "answer_match_status": "no_numeric_gold",
        }
    if generated_value is None:
        return {
            "numeric_accuracy": 0.0,
            "normalized_generated_answer": normalized_generated,
            "normalized_gold_answer": normalized_truth,
            "generated_number": None,
            "gold_number": truth_value,
            "absolute_error": None,
            "relative_error": None,
            "answer_match_status": "missing_generated_number",
        }
    absolute_error = abs(generated_value - truth_value)
    relative_error = absolute_error / abs(truth_value) if truth_value != 0 else None
    matched = _close(generated_value, truth_value)
    return {
        "numeric_accuracy": 1.0 if matched else 0.0,
        "normalized_generated_answer": normalized_generated,
        "normalized_gold_answer": normalized_truth,
        "generated_number": generated_value,
        "gold_number": truth_value,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "answer_match_status": "match" if matched else "mismatch",
    }


def _best_value(text: str) -> Decimal | None:
    normalized = _normalized_text(text)
    if normalized in _YES_NO:
        return _YES_NO[normalized]
    records = _extract_number_records(text)
    if not records:
        return None
    return records[0]["value"]


def _normalized_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _scale_after(text: str, end: int) -> Decimal:
    tail = text[end : end + 24].strip().lower()
    match = re.match(r"^[\s\-]*(thousand|million|billion|bn|mm|k)\b", tail)
    if not match:
        return Decimal("1")
    return _SCALE_WORDS[match.group(1)]


def _decimal_to_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)
