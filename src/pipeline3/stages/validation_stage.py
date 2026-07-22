from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pipeline3.validation")


class ValidationError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def validate_inputs(
    rag_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    questions_rows: list[dict[str, Any]],
    *,
    official_mode: bool = False,
    pipeline1_manifests: list[dict[str, Any]] | None = None,
) -> ValidationReport:
    """Stage 2: Data integrity validation."""
    errors: list[str] = []
    warnings: list[str] = []

    if not rag_rows:
        errors.append("Pipeline 1 results file contains zero rows.")
    if not qa_rows:
        errors.append("QA ground truth file contains zero rows.")
    if not questions_rows:
        errors.append("Questions file contains zero rows.")

    qa_by_id = _index_by_id(qa_rows)
    rag_ids = [_resolve_id(row) for row in rag_rows]
    rag_ids_clean = [qid for qid in rag_ids if qid]

    missing_qids = [qid for qid in rag_ids if not qid]
    if missing_qids:
        errors.append(
            f"Pipeline 1 result rows missing question_id: {len(missing_qids)} rows"
        )

    dup_keys = [key for key, count in Counter(rag_ids_clean).items() if count > 1]
    if dup_keys:
        errors.append(
            f"Duplicate question_ids in Pipeline 1 results: {sorted(dup_keys)[:10]}"
        )

    missing_qa = [qid for qid in rag_ids_clean if qid not in qa_by_id]
    if missing_qa:
        errors.append(
            f"QA ground truth missing for {len(missing_qa)} Pipeline 1 question_ids: "
            f"{sorted(missing_qa)[:10]}"
        )

    missing_answers = [
        _resolve_id(row)
        for row in rag_rows
        if not str(row.get("generated_answer", "")).strip()
    ]
    if missing_answers:
        message = f"Missing generated_answer for {len(missing_answers)} rows: {missing_answers[:5]}"
        if official_mode:
            errors.append(message)
        else:
            warnings.append(message)

    missing_contexts = [
        _resolve_id(row) for row in rag_rows if not _extract_context_texts(row)
    ]
    if missing_contexts:
        message = f"Missing retrieved contexts for {len(missing_contexts)} rows: {missing_contexts[:5]}"
        if official_mode:
            errors.append(message)
        else:
            warnings.append(message)

    missing_retrieved_contexts = [
        _resolve_id(row) for row in rag_rows if not _extract_retrieved_context_texts(row)
    ]
    if missing_retrieved_contexts:
        message = (
            f"Missing retrieved contexts for {len(missing_retrieved_contexts)} rows: "
            f"{missing_retrieved_contexts[:5]}"
        )
        if official_mode:
            errors.append(message)
        elif not missing_contexts:
            warnings.append(message)

    missing_generation_contexts = [
        _resolve_id(row) for row in rag_rows if not _extract_generation_context_texts(row)
    ]
    if missing_generation_contexts:
        message = (
            f"Missing generation contexts for {len(missing_generation_contexts)} rows: "
            f"{missing_generation_contexts[:5]}"
        )
        if official_mode:
            errors.append(message)
        else:
            warnings.append(message)

    missing_questions = [
        _resolve_id(row) for row in rag_rows if not _resolve_question_text(row)
    ]
    if missing_questions:
        message = f"Missing question text for {len(missing_questions)} rows: {missing_questions[:5]}"
        if official_mode:
            errors.append(message)
        else:
            warnings.append(message)

    failed_pipeline1_manifests = _failed_pipeline1_manifests(pipeline1_manifests or [])
    if failed_pipeline1_manifests:
        errors.append(
            "Official Pipeline 3 rejects non-PASS Pipeline 1 manifest(s): "
            f"{failed_pipeline1_manifests}"
        )

    empty_answers = [
        qid for qid, row in qa_by_id.items() if not _resolve_qa_answer(row)
    ]
    if empty_answers:
        warnings.append(
            f"QA rows with empty answers: {len(empty_answers)}: "
            f"{sorted(empty_answers)[:5]}"
        )

    failed_row_ids = {
        qid
        for qid in missing_qids
        + missing_qa
        + missing_answers
        + missing_contexts
        + missing_retrieved_contexts
        + missing_generation_contexts
        + missing_questions
        if qid
    }
    failed_rows = len(failed_row_ids) + len(missing_qids)
    expected_rows = len(questions_rows)
    valid_rows = max(0, len(rag_rows) - failed_rows)
    run_status = "PASS" if failed_rows == 0 and len(errors) == 0 else "FAIL"
    passed = len(errors) == 0
    report = ValidationReport(
        passed=passed,
        errors=errors,
        warnings=warnings,
        stats={
            "rag_rows": len(rag_rows),
            "qa_rows": len(qa_rows),
            "questions_rows": len(questions_rows),
            "rag_ids_with_qa": len(rag_ids_clean) - len(missing_qa),
            "missing_answers": len(missing_answers),
            "missing_contexts": len(missing_contexts),
            "missing_retrieved_contexts": len(missing_retrieved_contexts),
            "missing_generation_contexts": len(missing_generation_contexts),
            "missing_questions": len(missing_questions),
            "expected_rows": expected_rows,
            "valid_rows": valid_rows,
            "failed_rows": failed_rows,
            "run_status": run_status,
        },
    )

    if not passed:
        raise ValidationError(
            f"Validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    for w in warnings:
        logger.warning("Validation warning: %s", w)

    return report


def build_qa_index(qa_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _index_by_id(qa_rows)


def _index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = _resolve_id(row)
        if qid:
            indexed[qid] = row
    return indexed


def _resolve_id(row: dict[str, Any]) -> str:
    for key in ("question_id", "uid", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_context_texts(row: dict[str, Any]) -> list[str]:
    # C03 and later: prefer generation_context_texts (expanded parent sections).
    gen_texts = row.get("generation_context_texts")
    if isinstance(gen_texts, list) and any(str(t).strip() for t in gen_texts):
        return [str(t) for t in gen_texts if str(t).strip()]
    # Fall back to child retrieval contexts (C00 and older runs).
    texts = row.get("retrieved_context_texts") or row.get("retrieved_chunk_texts")
    if isinstance(texts, list) and any(str(t).strip() for t in texts):
        return [str(t) for t in texts if str(t).strip()]
    chunks = row.get("retrieved_chunks")
    if isinstance(chunks, list):
        result = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("chunk_text") or chunk.get("text") or ""
                if text.strip():
                    result.append(str(text))
            elif isinstance(chunk, str) and chunk.strip():
                result.append(chunk)
        return result
    return []


def _extract_generation_context_texts(row: dict[str, Any]) -> list[str]:
    gen_texts = row.get("generation_context_texts")
    if isinstance(gen_texts, list) and any(str(t).strip() for t in gen_texts):
        return [str(t) for t in gen_texts if str(t).strip()]
    return []


def _extract_retrieved_context_texts(row: dict[str, Any]) -> list[str]:
    texts = row.get("retrieved_context_texts") or row.get("retrieved_chunk_texts")
    if isinstance(texts, list) and any(str(t).strip() for t in texts):
        return [str(t) for t in texts if str(t).strip()]
    chunks = row.get("retrieved_chunks")
    if isinstance(chunks, list):
        result = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("chunk_text") or chunk.get("text") or ""
                if str(text).strip():
                    result.append(str(text))
            elif isinstance(chunk, str) and chunk.strip():
                result.append(chunk)
        return result
    return []


def _resolve_question_text(row: dict[str, Any]) -> str:
    for key in ("question", "frage", "query", "question_text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _failed_pipeline1_manifests(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifests):
        run_stats = manifest.get("run_stats") if isinstance(manifest.get("run_stats"), dict) else {}
        run_status = manifest.get("run_status") or run_stats.get("run_status")
        failed_raw = manifest.get("failed_questions", run_stats.get("failed_questions", 0))
        try:
            failed_questions = int(failed_raw)
        except (TypeError, ValueError):
            failed_questions = -1
        if run_status != "PASS" or failed_questions != 0:
            failed.append(
                {
                    "index": index,
                    "run_id": manifest.get("run_id"),
                    "run_status": run_status,
                    "failed_questions": failed_raw,
                }
            )
    return failed


def _resolve_qa_answer(row: dict[str, Any]) -> str:
    for key in (
        "ground_truth_answer",
        "answer",
        "gold_answer",
        "expected_answer",
        "program_answer",
        "original_answer",
        "referenzantwort",
    ):
        if key in row and row[key] is not None and str(row[key]).strip():
            return str(row[key])
    return ""
