from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_pipeline1_manifest(results_path: Path) -> dict[str, Any]:
    manifest_path = results_path.parent / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path = results_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Official evaluation requires a Pipeline 1 manifest next to {results_path}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        raise ValueError(f"Pipeline 1 manifest is malformed JSON: {manifest_path}") from ex


def validate_pipeline1_source(
    *,
    results_path: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    source_validation: Any,
    pipeline_name: str,
) -> None:
    if source_validation is None:
        return
    expected_experiment_id = getattr(source_validation, "expected_experiment_id", None)
    expected_retriever_type = getattr(source_validation, "expected_retriever_type", None)
    expected_orchestration_enabled = getattr(source_validation, "expected_orchestration_enabled", None)
    require_hybrid = bool(getattr(source_validation, "require_hybrid_diagnostics", False))
    require_reranker = bool(getattr(source_validation, "require_reranker_diagnostics", False))
    require_routing = bool(getattr(source_validation, "require_routing_diagnostics", False))
    require_reconciliation = bool(getattr(source_validation, "require_routing_reconciliation", False))

    _validate_manifest_pass(manifest, results_path, pipeline_name)
    if not rows:
        raise ValueError(f"{pipeline_name} source validation failed: Pipeline 1 results contain zero rows.")
    if expected_experiment_id:
        _require_equal(_manifest_experiment_id(manifest), expected_experiment_id, "manifest experiment_id", pipeline_name)
        _require_all_rows(rows, "experiment_id", expected_experiment_id, pipeline_name)
    if expected_retriever_type:
        _require_equal(_manifest_retriever_type(manifest), expected_retriever_type, "manifest retriever_type", pipeline_name)
        for row in rows:
            qid = _qid(row)
            diagnostics = _diagnostics(row)
            row_type = row.get("retriever_type") or diagnostics.get("retriever_type")
            if row_type != expected_retriever_type:
                raise ValueError(
                    f"{pipeline_name} source validation failed for question_id={qid}: "
                    f"retriever_type={row_type!r}, expected {expected_retriever_type!r}."
                )
            mode = row.get("retrieval_mode") or diagnostics.get("retrieval_mode")
            if mode == "adaptive_category_aware_dense" and expected_retriever_type == "adaptive_category_aware_hybrid_rrf":
                raise ValueError(
                    f"{pipeline_name} source validation failed for question_id={qid}: "
                    "R01 hybrid row is mislabeled as adaptive_category_aware_dense."
                )
    if expected_orchestration_enabled is not None:
        observed = _manifest_orchestration_enabled(manifest)
        _require_equal(observed, expected_orchestration_enabled, "manifest orchestration_enabled", pipeline_name)

    expected_questions = _manifest_expected_questions(manifest)
    if expected_questions is not None and expected_questions != len(rows):
        raise ValueError(
            f"{pipeline_name} source validation failed: Pipeline 1 row count {len(rows)} "
            f"does not match manifest expected_questions={expected_questions}."
        )

    top_k = _manifest_top_k(manifest)
    route_counts = {"category": 0, "global": 0, "global_fallback": 0}
    rows_with_routing = 0
    dense_total = bm25_total = fused_total = reranked_total = 0
    for row in rows:
        qid = _qid(row)
        diagnostics = _diagnostics(row)
        if not diagnostics:
            raise ValueError(f"{pipeline_name} source validation failed for question_id={qid}: missing retrieval_diagnostics.")
        if require_hybrid:
            dense_total += _require_nonnegative_int(diagnostics, ("dense_candidate_count", "es_hybrid_dense_candidates"), qid, "dense candidate count", pipeline_name)
            bm25_total += _require_nonnegative_int(diagnostics, ("bm25_candidate_count", "es_hybrid_bm25_candidates"), qid, "BM25 candidate count", pipeline_name)
            fused_total += _require_nonnegative_int(diagnostics, ("fused_candidate_count", "es_hybrid_fused_candidates"), qid, "fused candidate count", pipeline_name)
        if require_reranker:
            if row.get("reranker_applied") is not True and diagnostics.get("reranker_applied") is not True:
                raise ValueError(f"{pipeline_name} source validation failed for question_id={qid}: reranker evidence is missing.")
            reranked_total += _require_nonnegative_int(diagnostics, ("reranked_candidate_count", "reranker_output_count"), qid, "reranked candidate count", pipeline_name)
        final_count = _require_nonnegative_int(diagnostics, ("final_context_count", "final_result_count"), qid, "final context count", pipeline_name)
        row_top_k = int(diagnostics.get("top_k") or top_k or row.get("retrieval_k") or 0)
        if row_top_k and final_count > row_top_k:
            raise ValueError(
                f"{pipeline_name} source validation failed for question_id={qid}: "
                f"final_context_count={final_count} exceeds top_k={row_top_k}."
            )
        scope = str(diagnostics.get("retrieval_scope") or row.get("retrieval_scope") or "")
        if expected_retriever_type == "elasticsearch_hybrid_rrf":
            if scope and scope != "global":
                raise ValueError(f"{pipeline_name} source validation failed for question_id={qid}: R00 scope is {scope!r}, expected 'global'.")
            if diagnostics.get("routing_decision") in {"accepted", "rejected"}:
                raise ValueError(f"{pipeline_name} source validation failed for question_id={qid}: R00 contains adaptive routing diagnostics.")
        if require_routing:
            decision = diagnostics.get("routing_decision")
            final_mode = str(diagnostics.get("final_retrieval_mode") or diagnostics.get("retrieval_scope") or "")
            if decision not in {"accepted", "rejected"} or final_mode not in {"category", "global"}:
                raise ValueError(f"{pipeline_name} source validation failed for question_id={qid}: missing explicit routing outcome.")
            rows_with_routing += 1
            fallback_used = diagnostics.get("fallback_used") is True or diagnostics.get("category_fallback_used") is True
            if final_mode == "category" and not fallback_used:
                route_counts["category"] += 1
                if diagnostics.get("category_filter_applied_dense") is not True or diagnostics.get("category_filter_applied_bm25") is not True:
                    raise ValueError(
                        f"{pipeline_name} source validation failed for question_id={qid}: "
                        "category route lacks dense/BM25 category-filter evidence."
                    )
            elif fallback_used:
                route_counts["global_fallback"] += 1
            else:
                route_counts["global"] += 1

    if require_hybrid and (dense_total <= 0 or bm25_total <= 0 or fused_total <= 0):
        raise ValueError(f"{pipeline_name} source validation failed: aggregate Hybrid RRF evidence is empty.")
    if require_reranker and reranked_total <= 0:
        raise ValueError(f"{pipeline_name} source validation failed: aggregate reranking evidence is empty.")
    if require_routing:
        successful_rows = _manifest_successful_questions(manifest, len(rows))
        if rows_with_routing != successful_rows:
            raise ValueError(
                f"{pipeline_name} source validation failed: rows_with_routing_diagnostics={rows_with_routing} "
                f"does not match successful_rows={successful_rows}."
            )
    if require_reconciliation:
        _validate_routing_reconciliation(manifest, route_counts, rows_with_routing, pipeline_name)


def _validate_manifest_pass(manifest: dict[str, Any], results_path: Path, pipeline_name: str) -> None:
    run_stats = manifest.get("run_stats") if isinstance(manifest.get("run_stats"), dict) else {}
    run_status = manifest.get("run_status") or run_stats.get("run_status")
    failed_raw = manifest.get("failed_questions", run_stats.get("failed_questions"))
    try:
        failed_questions = int(failed_raw)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"Pipeline 1 manifest missing numeric failed_questions: {results_path.parent}") from ex
    if run_status != "PASS" or failed_questions != 0:
        raise RuntimeError(
            f"{pipeline_name} rejects non-PASS Pipeline 1 output: "
            f"manifest={results_path.parent / 'run_manifest.json'}, run_status={run_status}, failed_questions={failed_questions}"
        )


def _validate_routing_reconciliation(
    manifest: dict[str, Any],
    route_counts: dict[str, int],
    rows_with_routing: int,
    pipeline_name: str,
) -> None:
    routing = manifest.get("category_routing_validation")
    if not isinstance(routing, dict):
        raise ValueError(f"{pipeline_name} source validation failed: missing manifest category_routing_validation.")
    manifest_category = int(routing.get("category_route_count", -1))
    manifest_global = int(routing.get("global_route_count", -1))
    manifest_fallback = int(routing.get("fallback_count", routing.get("fallback_route_count", -1)))
    row_category = route_counts["category"]
    row_global_total = route_counts["global"] + route_counts["global_fallback"]
    if manifest_category != row_category or manifest_global != row_global_total or manifest_fallback != route_counts["global_fallback"]:
        raise ValueError(
            f"{pipeline_name} source validation failed: routing manifest counts do not match rows "
            f"(manifest category/global/fallback={manifest_category}/{manifest_global}/{manifest_fallback}, "
            f"rows={row_category}/{row_global_total}/{route_counts['global_fallback']})."
        )
    if row_category + row_global_total != rows_with_routing:
        raise ValueError(
            f"{pipeline_name} source validation failed: category_route_count + global_route_count "
            f"!= successful routed rows ({row_category} + {row_global_total} != {rows_with_routing})."
        )


def _manifest_experiment_id(manifest: dict[str, Any]) -> str | None:
    return manifest.get("run_id") or manifest.get("experiment_id") or manifest.get("config", {}).get("experiment", {}).get("experiment_id")


def _manifest_retriever_type(manifest: dict[str, Any]) -> str | None:
    return (
        manifest.get("models", {}).get("retriever_type")
        or manifest.get("resolved_config", {}).get("retrieval", {}).get("retriever_type")
        or manifest.get("config", {}).get("retrieval", {}).get("retriever_type")
    )


def _manifest_orchestration_enabled(manifest: dict[str, Any]) -> bool | None:
    if "orchestration_enabled" in manifest:
        return bool(manifest["orchestration_enabled"])
    models = manifest.get("models") if isinstance(manifest.get("models"), dict) else {}
    if "orchestration_enabled" in models:
        return bool(models["orchestration_enabled"])
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    orchestration = config.get("orchestration") if isinstance(config.get("orchestration"), dict) else {}
    if "enabled" in orchestration:
        return bool(orchestration["enabled"])
    return None


def _manifest_expected_questions(manifest: dict[str, Any]) -> int | None:
    value = manifest.get("expected_questions", manifest.get("run_stats", {}).get("expected_questions"))
    return None if value is None else int(value)


def _manifest_successful_questions(manifest: dict[str, Any], default: int) -> int:
    value = manifest.get("successful_questions", manifest.get("run_stats", {}).get("successful_questions", default))
    return int(value)


def _manifest_top_k(manifest: dict[str, Any]) -> int | None:
    value = manifest.get("config", {}).get("retrieval", {}).get("top_k")
    return None if value is None else int(value)


def _require_equal(observed: Any, expected: Any, label: str, pipeline_name: str) -> None:
    if observed != expected:
        raise ValueError(
            f"{pipeline_name} source validation failed: {label}={observed!r}, expected {expected!r}."
        )


def _require_all_rows(rows: list[dict[str, Any]], field: str, expected: Any, pipeline_name: str) -> None:
    mismatches = [_qid(row) for row in rows if row.get(field) != expected]
    if mismatches:
        raise ValueError(
            f"{pipeline_name} source validation failed: {field} mismatch for question_ids={mismatches[:10]}."
        )


def _require_nonnegative_int(
    diagnostics: dict[str, Any],
    keys: tuple[str, ...],
    qid: str,
    label: str,
    pipeline_name: str,
) -> int:
    for key in keys:
        if key in diagnostics and diagnostics[key] is not None:
            value = int(diagnostics[key])
            if value < 0:
                raise ValueError(f"{pipeline_name} source validation failed for question_id={qid}: {label} is negative.")
            return value
    raise ValueError(f"{pipeline_name} source validation failed for question_id={qid}: missing {label}.")


def _diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = row.get("retrieval_diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _qid(row: dict[str, Any]) -> str:
    return str(row.get("question_id") or row.get("uid") or "<unknown>")
