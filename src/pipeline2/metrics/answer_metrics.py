from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


_NUMBER_RE = re.compile(r"[-+]?\(?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_YES_NO = {
    "yes": Decimal("1"),
    "y": Decimal("1"),
    "true": Decimal("1"),
    "no": Decimal("0"),
    "n": Decimal("0"),
    "false": Decimal("0"),
}
_SCALE_WORDS = {
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "mm": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
}
DEFAULT_ABSTENTION_PATTERNS = (
    "unknown",
    "not found",
    "n/a",
    "na",
    "cannot determine",
    "can't determine",
    "insufficient information",
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RELEVANCY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def resolve_ground_truth_answer(row: dict[str, Any], qa_by_id: dict[str, dict[str, Any]]) -> str:
    qid = str(row.get("question_id", ""))
    qa = qa_by_id.get(qid, {})
    for key in ("ground_truth_answer", "answer", "gold_answer", "expected_answer", "program_answer", "original_answer"):
        if key in qa and qa[key] is not None:
            return str(qa[key])
    return ""


def compute_answer_metrics(
    generated_answer: str,
    ground_truth_answer: str,
    question: str = "",
    abstention_patterns: Iterable[str] | None = None,
) -> dict[str, Any]:
    generated = generated_answer or ""
    truth = ground_truth_answer or ""
    match = _compare_answers(generated, truth)
    non_empty = 1.0 if generated.strip() else 0.0
    abstained = 1.0 if is_abstention(generated, abstention_patterns) else 0.0
    parse_success = 1.0 if _best_value(generated) is not None else 0.0
    literal_exact_match = 1.0 if _literal_exact_text(generated) == _literal_exact_text(truth) and truth.strip() else 0.0
    canonical_exact_match = 1.0 if _normalized_exact_text(generated) == _normalized_exact_text(truth) and truth.strip() else 0.0
    return {
        "numeric_accuracy": match["numeric_accuracy"],
        "strict_numeric_accuracy": match["strict_numeric_accuracy"],
        "tolerant_numeric_accuracy": match["tolerant_numeric_accuracy"],
        "number_match": match["numeric_accuracy"],
        "exact_match": literal_exact_match,
        "literal_exact_match": literal_exact_match,
        "canonical_exact_match": canonical_exact_match,
        "exact_match_debug": canonical_exact_match,
        "normalized_generated_answer": match["normalized_generated_answer"],
        "normalized_gold_answer": match["normalized_gold_answer"],
        "generated_number": _decimal_to_float(match["generated_number"]),
        "gold_number": _decimal_to_float(match["gold_number"]),
        "absolute_error": _decimal_to_float(match["absolute_error"]),
        "relative_error": _decimal_to_float(match["relative_error"]),
        "numeric_parse_success": parse_success,
        "answer_match_status": match["answer_match_status"],
        "non_empty_answer_rate": non_empty,
        "answer_coverage_rate": non_empty,  # Backward-compatible alias; canonical name is non_empty_answer_rate.
        "abstention_rate": abstained,
        "answer_relevancy_score": answer_relevancy_score(question, generated),
    }


def compute_numeric_accuracy(generated_answer: str, ground_truth_answer: str) -> float | None:
    return _compare_answers(generated_answer, ground_truth_answer)["numeric_accuracy"]


def compute_number_match(generated_answer: str, ground_truth_answer: str) -> float | None:
    return compute_numeric_accuracy(generated_answer, ground_truth_answer)


def is_abstention(text: str, patterns: Iterable[str] | None = None) -> bool:
    normalized = _normalized_text(text)
    if not normalized:
        return True
    canonical = normalized.strip(" .!?:;")
    configured = tuple(patterns or DEFAULT_ABSTENTION_PATTERNS)
    return any(canonical == _normalized_text(pattern).strip(" .!?:;") for pattern in configured)


def answer_relevancy_score(question: str, generated_answer: str) -> float:
    """Deterministic lexical-overlap baseline, not a semantic correctness metric."""
    question_tokens = _content_tokens(question)
    answer_tokens = _content_tokens(generated_answer)
    if not question_tokens or not answer_tokens:
        return 0.0
    return len(question_tokens & answer_tokens) / len(answer_tokens)


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
            "strict_numeric_accuracy": None,
            "tolerant_numeric_accuracy": None,
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
            "strict_numeric_accuracy": 0.0,
            "tolerant_numeric_accuracy": 0.0,
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
    strict_matched = generated_value == truth_value
    tolerant_matched = _close(generated_value, truth_value)
    return {
        "numeric_accuracy": 1.0 if strict_matched else 0.0,
        "strict_numeric_accuracy": 1.0 if strict_matched else 0.0,
        "tolerant_numeric_accuracy": 1.0 if tolerant_matched else 0.0,
        "normalized_generated_answer": normalized_generated,
        "normalized_gold_answer": normalized_truth,
        "generated_number": generated_value,
        "gold_number": truth_value,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "answer_match_status": "match" if strict_matched else "mismatch",
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


def _literal_exact_text(text: str) -> str:
    return _normalized_text(text)


def _normalized_exact_text(text: str) -> str:
    normalized = _normalized_text(text)
    if normalized in _YES_NO:
        return str(_YES_NO[normalized])
    value = _best_value(text)
    if value is not None and len(_extract_number_records(text)) == 1:
        return _decimal_to_canonical_text(value)
    return re.sub(r"[\s,$%]", "", normalized)


def _decimal_to_canonical_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


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


def _content_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").lower()) if token not in _RELEVANCY_STOPWORDS}
