from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

from src.pipeline2.aggregation.summarizer import build_leaderboard, summarize_by_experiment
from src.pipeline2.io.jsonl import read_jsonl, write_jsonl
from src.pipeline2.io.tabular import write_csv
from src.pipeline2.metrics.answer_metrics import compute_answer_metrics, resolve_ground_truth_answer
from src.pipeline2.metrics.efficiency_metrics import compute_efficiency_metrics
from src.pipeline2.metrics.retrieval_metrics import compute_metadata_match_metrics, compute_retrieval_metrics_for_ks
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline1.utils.hashing import file_sha256
from tqdm.auto import tqdm


class EvaluationOrchestrator:
    def run(self, config_path: str) -> Path:
        start_time = time.time()
        cfg = EvalConfig.from_yaml(config_path)
        project_root = Path(__file__).resolve().parents[2]
        run_dir = project_root / cfg.evaluation.output_dir / cfg.evaluation.eval_run_id
        if run_dir.exists() and not cfg.runtime.overwrite:
            raise FileExistsError(f"Evaluation run already exists and overwrite=false: {run_dir}")
        if run_dir.exists() and cfg.runtime.overwrite:
            for name in (
                "per_question.jsonl",
                "per_question.csv",
                "summary_by_experiment.csv",
                "summary_by_difficulty.csv",
                "summary_by_difficulty.json",
                "leaderboard.csv",
                "eval_manifest.json",
            ):
                path = run_dir / name
                if path.exists():
                    path.unlink()
        run_dir.mkdir(parents=True, exist_ok=True)

        print("[1/6] Loading Pipeline 1 outputs")
        rag_rows = []
        resolved_rag_paths = []
        for rag_path in cfg.inputs.rag_outputs:
            resolved = _resolve(project_root, rag_path)
            resolved_rag_paths.append(resolved)
            rag_rows.extend(read_jsonl(resolved))
        print("[2/6] Loading QA gold answers")
        qa_path = _resolve(project_root, cfg.inputs.qa_path)
        qa_rows = read_jsonl(qa_path)
        qa_by_id = _index_by_id(qa_rows, require_answer=not cfg.evaluation.retrieval_only)
        _validate_pipeline1_questions_have_qa(rag_rows, qa_by_id)
        if cfg.debug.enable_officeqa_smoke_check:
            _run_officeqa_smoke_validation(qa_by_id)
        print("[3/6] Loading gold contexts")
        gold_path = _resolve(project_root, cfg.inputs.gold_contexts_path)
        gold_rows = read_jsonl(gold_path) if gold_path.exists() else []
        gold_by_id = _merge_gold_with_qa_fallback(_gold_by_question(gold_rows), qa_by_id)

        print("[4/6] Computing automatic metrics")
        per_question = self._evaluate_rows(rag_rows, qa_by_id, gold_by_id, cfg)
        print("[5/6] Aggregating summaries")
        summary = summarize_by_experiment(per_question)
        run_validity = _run_validity_by_experiment(per_question, cfg.evaluation.max_generation_failure_rate)
        _attach_run_validity(summary, run_validity)
        if cfg.evaluation.strict_failure_threshold:
            _raise_on_failure_threshold(run_validity, cfg.evaluation.max_generation_failure_rate)
        difficulty_summary = summarize_by_difficulty(per_question)
        leaderboard = build_leaderboard(summary, cfg.leaderboard.sort_metric, cfg.leaderboard.sort_ascending)
        ks = _metric_ks(cfg)
        per_fields = _per_question_fields(ks)
        summary_fields = _summary_fields(ks)
        leaderboard_fields = ["rank", "sort_metric", *summary_fields]
        print("[6/6] Writing evaluation outputs")
        write_jsonl(run_dir / "per_question.jsonl", per_question)
        if cfg.runtime.save_csv:
            write_csv(run_dir / "per_question.csv", per_question, per_fields)
            write_csv(run_dir / "summary_by_experiment.csv", summary, summary_fields)
            write_csv(run_dir / "summary_by_difficulty.csv", difficulty_summary, _difficulty_summary_fields(ks))
            write_csv(run_dir / "leaderboard.csv", leaderboard, leaderboard_fields)
        (run_dir / "summary_by_difficulty.json").write_text(
            json.dumps(difficulty_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "eval_manifest.json").write_text(
            json.dumps(
                _eval_manifest(
                    config_path,
                    cfg,
                    resolved_rag_paths,
                    qa_path,
                    gold_path,
                    rag_rows,
                    qa_rows,
                    gold_rows,
                    per_question,
                    leaderboard,
                    difficulty_summary,
                    run_validity,
                    start_time,
                    time.time(),
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return run_dir

    def _evaluate_rows(
        self,
        rag_rows: list[dict[str, Any]],
        qa_by_id: dict[str, dict[str, Any]],
        gold_by_id: dict[str, list[str]],
        cfg: EvalConfig,
    ) -> list[dict[str, Any]]:
        ks = _metric_ks(cfg)
        missing_gold_ids = [
            str(row.get("question_id", ""))
            for row in rag_rows
            if not gold_by_id.get(str(row.get("question_id", "")))
        ]
        if missing_gold_ids:
            sample = ", ".join(missing_gold_ids[:20])
            suffix = "" if len(missing_gold_ids) <= 20 else f", ... ({len(missing_gold_ids)} total)"
            raise ValueError(
                "Pipeline 2 requires ground_truth_contexts.jsonl entries for every evaluated Pipeline 1 "
                f"question_id. Missing {len(missing_gold_ids)} question(s): {sample}{suffix}"
            )
        evaluated = []
        for row in tqdm(rag_rows, desc="Computing metrics", unit="question"):
            errors = []
            qid = str(row.get("question_id", ""))
            pipeline1_error = row.get("error") or row.get("pipeline1_error")
            generation_failed = False if cfg.evaluation.retrieval_only else _is_generation_failure(row)
            pipeline_success = 0.0 if generation_failed else 1.0
            retrieved_ids = row.get("retrieved_original_context_ids")
            id_alignment_ok = True
            if "retrieved_original_context_ids" not in row:
                retrieved_ids = []
                id_alignment_ok = False
                message = (
                    f"Pipeline 1 result for question_id={qid!r} is missing "
                    "retrieved_original_context_ids; retrieval metrics will be scored as empty."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                errors.append(message)
            if not isinstance(retrieved_ids, list):
                retrieved_ids = []
                id_alignment_ok = False
                errors.append("retrieved_original_context_ids must be a list")
            gold_ids = gold_by_id.get(qid, [])
            if not gold_ids:
                id_alignment_ok = False
            retrieval_eval_ids = _configured_retrieval_eval_ids(row, cfg.evaluation.retrieval_eval_field)
            qa_row = qa_by_id.get(qid, {})
            ground_truth = "" if cfg.evaluation.retrieval_only else resolve_ground_truth_answer(row, qa_by_id)
            raw_retrieved_ids = row.get("raw_retrieved_original_context_ids")
            if raw_retrieved_ids is not None and not isinstance(raw_retrieved_ids, list):
                raw_retrieved_ids = []
                errors.append("raw_retrieved_original_context_ids must be a list")
            raw_retrieval_eval_ids = _configured_raw_retrieval_eval_ids(row, cfg.evaluation.retrieval_eval_field)
            retrieved_metadata = row.get("retrieved_chunk_metadata") or []
            if not isinstance(retrieved_metadata, list):
                retrieved_metadata = []
                errors.append("retrieved_chunk_metadata must be a list")
            if cfg.evaluation.retrieval_only:
                answer_metrics = _null_answer_metrics()
            else:
                answer_metrics = compute_answer_metrics(
                    str(row.get("generated_answer", "")),
                    ground_truth,
                    question=str(row.get("question", "")),
                    abstention_patterns=cfg.answer_quality.abstention_patterns,
                )
                if not cfg.answer_quality.enable_numeric_accuracy:
                    answer_metrics["numeric_accuracy"] = None
                    answer_metrics["strict_numeric_accuracy"] = None
                    answer_metrics["tolerant_numeric_accuracy"] = None
            if generation_failed and not cfg.evaluation.retrieval_only:
                failure_status = "pipeline1_error" if pipeline1_error else "generation_failure"
                answer_metrics.update(
                    {
                        "numeric_accuracy": 0.0,
                        "strict_numeric_accuracy": 0.0,
                        "tolerant_numeric_accuracy": 0.0,
                        "exact_match": 0.0,
                        "literal_exact_match": 0.0,
                        "canonical_exact_match": 0.0,
                        "numeric_parse_success": 0.0,
                        "non_empty_answer_rate": 0.0,
                        "answer_coverage_rate": 0.0,
                        "abstention_rate": 0.0,
                        "answer_relevancy_score": 0.0,
                        "normalized_generated_answer": "",
                        "generated_number": None,
                        "absolute_error": None,
                        "relative_error": None,
                        "answer_match_status": failure_status,
                    }
                )
            output = {
                "question_id": qid,
                "uid": qid,
                "experiment_id": str(row.get("experiment_id", "")),
                "generated_answer": row.get("generated_answer", ""),
                "ground_truth_answer": ground_truth,
                "difficulty": str(qa_row.get("difficulty", "unknown") or "unknown"),
                "retrieved_original_context_ids": retrieved_ids,
                "raw_retrieved_original_context_ids": raw_retrieved_ids,
                "retrieval_eval_ids": retrieval_eval_ids,
                "raw_retrieval_eval_ids": raw_retrieval_eval_ids,
                "gold_context_ids": gold_ids,
                "id_alignment_ok": id_alignment_ok,
                **compute_retrieval_metrics_for_ks(retrieval_eval_ids, gold_ids, ks, raw_retrieval_eval_ids),
                **compute_metadata_match_metrics(
                    str(row.get("question", "")),
                    retrieved_metadata,
                    row.get("query_metadata") if isinstance(row.get("query_metadata"), dict) else None,
                ),
                "numeric_accuracy": answer_metrics["numeric_accuracy"],
                "strict_numeric_accuracy": answer_metrics["strict_numeric_accuracy"],
                "tolerant_numeric_accuracy": answer_metrics["tolerant_numeric_accuracy"],
                "exact_match": answer_metrics["exact_match"],
                "literal_exact_match": answer_metrics["literal_exact_match"],
                "canonical_exact_match": answer_metrics["canonical_exact_match"],
                "numeric_parse_success": answer_metrics["numeric_parse_success"],
                "non_empty_answer_rate": answer_metrics["non_empty_answer_rate"],
                "answer_coverage_rate": answer_metrics["answer_coverage_rate"],
                "abstention_rate": answer_metrics["abstention_rate"],
                "answer_relevancy_score": answer_metrics["answer_relevancy_score"],
                "normalized_generated_answer": answer_metrics["normalized_generated_answer"],
                "normalized_gold_answer": answer_metrics["normalized_gold_answer"],
                "generated_number": answer_metrics["generated_number"],
                "gold_number": answer_metrics["gold_number"],
                "absolute_error": answer_metrics["absolute_error"],
                "relative_error": answer_metrics["relative_error"],
                "answer_match_status": answer_metrics["answer_match_status"],
                "hallucination_rate": None,
                **compute_efficiency_metrics(row),
                "pipeline_success": pipeline_success,
                "generation_failed": generation_failed,
                "pipeline1_error": pipeline1_error,
                "evaluation_errors": errors,
            }
            evaluated.append(output)
        return evaluated


def _index_by_id(rows: list[dict[str, Any]], require_answer: bool = True) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    counts = {"uid": 0, "id": 0, "question_id": 0}
    duplicate_ids: set[str] = set()
    missing_rows: list[int] = []
    empty_answer_ids: list[str] = []

    for line_number, row in enumerate(rows, start=1):
        key_name, raw_id = _resolve_qa_row_id(row)
        if raw_id is None or str(raw_id).strip() == "":
            missing_rows.append(line_number)
            continue
        qid = str(raw_id)
        counts[key_name] += 1
        if qid in indexed:
            duplicate_ids.add(qid)
        indexed[qid] = row
        if require_answer and not _has_non_empty_answer(row):
            empty_answer_ids.append(qid)

    if missing_rows:
        sample = ", ".join(str(item) for item in missing_rows[:20])
        suffix = "" if len(missing_rows) <= 20 else f", ... ({len(missing_rows)} total)"
        raise ValueError(f"QA rows are missing uid/id/question_id on line(s): {sample}{suffix}")
    if duplicate_ids:
        sample = ", ".join(sorted(duplicate_ids)[:20])
        suffix = "" if len(duplicate_ids) <= 20 else f", ... ({len(duplicate_ids)} total)"
        raise ValueError(f"QA rows contain duplicate resolved IDs: {sample}{suffix}")
    if empty_answer_ids:
        sample = ", ".join(empty_answer_ids[:20])
        suffix = "" if len(empty_answer_ids) <= 20 else f", ... ({len(empty_answer_ids)} total)"
        raise ValueError(f"QA rows have empty answer fields for {len(empty_answer_ids)} ID(s): {sample}{suffix}")

    print(
        "QA validation: "
        f"total_rows={len(rows)} unique_ids={len(indexed)} "
        f"indexed_by_uid={counts['uid']} indexed_by_id={counts['id']} "
        f"indexed_by_question_id={counts['question_id']}"
    )
    return indexed


def _resolve_qa_row_id(row: dict[str, Any]) -> tuple[str, Any]:
    for key in ("uid", "id", "question_id"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return key, value
    return "missing", None


def _has_non_empty_answer(row: dict[str, Any]) -> bool:
    for key in ("ground_truth_answer", "answer", "gold_answer", "expected_answer", "program_answer", "original_answer"):
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return True
    return False


def _validate_pipeline1_questions_have_qa(rag_rows: list[dict[str, Any]], qa_by_id: dict[str, dict[str, Any]]) -> None:
    missing = [
        str(row.get("question_id", ""))
        for row in rag_rows
        if str(row.get("question_id", "")) not in qa_by_id
    ]
    if missing:
        sample = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(f"QA file is missing answers for {len(missing)} Pipeline 1 question_id(s): {sample}{suffix}")


def _run_officeqa_smoke_validation(qa_by_id: dict[str, dict[str, Any]]) -> None:
    if "UID0002" not in qa_by_id:
        return
    probe_row = {"question_id": "UID0002", "generated_answer": "507"}
    gold_answer = resolve_ground_truth_answer(probe_row, qa_by_id)
    if not gold_answer.strip():
        raise ValueError("OfficeQA smoke validation failed: UID0002 resolved to an empty gold answer.")
    metrics = compute_answer_metrics(probe_row["generated_answer"], gold_answer)
    if metrics["numeric_accuracy"] != 1.0:
        raise ValueError(
            "OfficeQA smoke validation failed: generated_answer='507' did not numerically match "
            f"UID0002 gold answer={gold_answer!r}."
        )
    print(f"OfficeQA smoke validation: UID0002 gold_answer={gold_answer!r} numeric_accuracy=1.0")


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _gold_by_question(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row.get("id") or row.get("question_id"))
        ids = row.get("context_id") or []
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list):
            ids = []
        output.setdefault(qid, [])
        output[qid].extend(str(item) for item in ids if item is not None)
    return output


def _merge_gold_with_qa_fallback(gold_by_id: dict[str, list[str]], qa_by_id: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    merged = {qid: list(ids) for qid, ids in gold_by_id.items()}
    for qid, row in qa_by_id.items():
        if merged.get(qid):
            continue
        source_files = row.get("source_files") or []
        if isinstance(source_files, str):
            source_files = [source_files]
        elif not isinstance(source_files, list):
            source_files = []
        merged[qid] = [str(item) for item in source_files if item is not None and str(item).strip()]
    return merged


def _null_answer_metrics() -> dict[str, Any]:
    return {
        "numeric_accuracy": None,
        "strict_numeric_accuracy": None,
        "tolerant_numeric_accuracy": None,
        "exact_match": None,
        "literal_exact_match": None,
        "canonical_exact_match": None,
        "numeric_parse_success": None,
        "non_empty_answer_rate": None,
        "answer_coverage_rate": None,
        "abstention_rate": None,
        "answer_relevancy_score": None,
        "normalized_generated_answer": "",
        "normalized_gold_answer": "",
        "generated_number": None,
        "gold_number": None,
        "absolute_error": None,
        "relative_error": None,
        "answer_match_status": "skipped_retrieval_only",
    }


def _configured_retrieval_eval_ids(row: dict[str, Any], field: str) -> list[str]:
    return _required_list_field(row, field)


def _configured_raw_retrieval_eval_ids(row: dict[str, Any], field: str) -> list[str] | None:
    raw_field = {
        "retrieved_file_names": "raw_retrieved_file_names",
        "retrieved_files": "raw_retrieved_file_names",
        "retrieved_document_ids": "raw_retrieved_document_ids",
        "retrieved_original_context_ids": "raw_retrieved_original_context_ids",
    }[field]
    if raw_field not in row:
        return None
    return _required_list_field(row, raw_field)


def _required_list_field(row: dict[str, Any], field: str) -> list[str]:
    if field not in row:
        qid = row.get("question_id") or row.get("uid") or "<unknown>"
        raise ValueError(f"Configured retrieval_eval_field={field!r} is missing for question_id={qid!r}.")
    value = row.get(field)
    if not isinstance(value, list):
        qid = row.get("question_id") or row.get("uid") or "<unknown>"
        raise ValueError(f"Configured retrieval_eval_field={field!r} must be a list for question_id={qid!r}.")
    return [str(item) for item in value if item is not None and str(item).strip()]


def _list_field(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item).strip()]


def _is_generation_failure(row: dict[str, Any]) -> bool:
    if row.get("error") or row.get("pipeline1_error"):
        return True
    if "success" in row and row.get("success") is False:
        return True
    if "pipeline_success" in row and row.get("pipeline_success") is False:
        return True
    if row.get("generated_answer") is None or str(row.get("generated_answer", "")).strip() == "":
        return True
    return False


def _run_validity_by_experiment(rows: list[dict[str, Any]], max_failure_rate: float) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("experiment_id", "")), []).append(row)
    validity: dict[str, dict[str, Any]] = {}
    for experiment_id, group in groups.items():
        total = len(group)
        failures = sum(1 for row in group if row.get("generation_failed"))
        failure_rate = failures / total if total else 0.0
        valid = failure_rate <= max_failure_rate
        validity[experiment_id] = {
            "total_questions": total,
            "generation_failure_count": failures,
            "generation_failure_rate": failure_rate,
            "pipeline_success_rate": 1.0 - failure_rate if total else None,
            "max_generation_failure_rate": max_failure_rate,
            "run_valid": valid,
            "failure_threshold_exceeded": not valid,
        }
    return validity


def _attach_run_validity(summary_rows: list[dict[str, Any]], validity: dict[str, dict[str, Any]]) -> None:
    for row in summary_rows:
        row.update(validity.get(str(row.get("experiment_id", "")), {}))


def _raise_on_failure_threshold(validity: dict[str, dict[str, Any]], max_failure_rate: float) -> None:
    invalid = [
        f"{experiment_id}={stats['generation_failure_rate']:.3f}"
        for experiment_id, stats in validity.items()
        if not stats.get("run_valid", True)
    ]
    if invalid:
        raise RuntimeError(
            "Generation failure rate exceeded max_generation_failure_rate="
            f"{max_failure_rate}: {', '.join(invalid)}"
        )


def _metric_ks(cfg: EvalConfig) -> list[int]:
    return sorted({int(k) for k in (cfg.retrieval.ks or [cfg.retrieval.k]) if int(k) > 0})


def _per_question_fields(ks: list[int]) -> list[str]:
    metric_fields = []
    for k in ks:
        metric_fields.extend([f"hit_at_{k}", f"recall_at_{k}", f"mrr_at_{k}", f"context_precision_at_{k}", f"ndcg_at_{k}"])
    return [
        "uid",
        "question_id",
        "experiment_id",
        "difficulty",
        "generated_answer",
        "ground_truth_answer",
        "retrieved_original_context_ids",
        "raw_retrieved_original_context_ids",
        "retrieval_eval_ids",
        "raw_retrieval_eval_ids",
        "gold_context_ids",
        "id_alignment_ok",
        *metric_fields,
        "duplicate_context_rate",
        "raw_duplicate_rate",
        "metadata_match_rate",
        "company_match_rate",
        "year_match_rate",
        "numeric_accuracy",
        "strict_numeric_accuracy",
        "tolerant_numeric_accuracy",
        "exact_match",
        "literal_exact_match",
        "canonical_exact_match",
        "numeric_parse_success",
        "non_empty_answer_rate",
        "answer_coverage_rate",
        "abstention_rate",
        "answer_relevancy_score",
        "normalized_generated_answer",
        "normalized_gold_answer",
        "generated_number",
        "gold_number",
        "absolute_error",
        "relative_error",
        "answer_match_status",
        "retrieval_time_ms",
        "generation_time_ms",
        "total_latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost",
        "hallucination_rate",
        "pipeline_success",
        "generation_failed",
        "pipeline1_error",
        "evaluation_errors",
    ]


def summarize_by_difficulty(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for row in rows:
        difficulty = str(row.get("difficulty") or "unknown")
        groups.setdefault(difficulty, []).append(row)
    output = []
    for difficulty in ["all", *sorted(key for key in groups if key != "all")]:
        group = groups[difficulty]
        summary: dict[str, Any] = {"difficulty": difficulty, "n_questions": len(group)}
        metric_cols = sorted(
            {
                key
                for row in group
                for key in row
                if key.startswith(("hit_at_", "recall_at_", "mrr_at_", "context_precision_at_", "ndcg_at_"))
            }
        )
        for col in metric_cols:
            summary[f"mean_{col}"] = _mean([row.get(col) for row in group if row.get(col) is not None])
        for col in (
            "exact_match",
            "literal_exact_match",
            "canonical_exact_match",
            "numeric_accuracy",
            "strict_numeric_accuracy",
            "tolerant_numeric_accuracy",
            "hallucination_rate",
            "total_latency_ms",
            "total_tokens",
            "generation_failed",
        ):
            summary[f"mean_{col}"] = _mean([row.get(col) for row in group if row.get(col) is not None])
        output.append(summary)
    return output


def _difficulty_summary_fields(ks: list[int]) -> list[str]:
    metric_fields = []
    for k in ks:
        metric_fields.extend(
            [
                f"mean_hit_at_{k}",
                f"mean_recall_at_{k}",
                f"mean_mrr_at_{k}",
                f"mean_context_precision_at_{k}",
                f"mean_ndcg_at_{k}",
            ]
        )
    return [
        "difficulty",
        "n_questions",
        *metric_fields,
        "mean_exact_match",
        "mean_literal_exact_match",
        "mean_canonical_exact_match",
        "mean_numeric_accuracy",
        "mean_strict_numeric_accuracy",
        "mean_tolerant_numeric_accuracy",
        "mean_hallucination_rate",
        "mean_total_latency_ms",
        "mean_total_tokens",
        "mean_generation_failed",
    ]


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _summary_fields(ks: list[int]) -> list[str]:
    metric_fields = []
    for k in ks:
        metric_fields.extend(
            [
                f"mean_hit_at_{k}",
                f"mean_recall_at_{k}",
                f"mean_mrr_at_{k}",
                f"mean_context_precision_at_{k}",
                f"mean_ndcg_at_{k}",
            ]
        )
    return [
        "experiment_id",
        "n_questions",
        "pipeline_success_rate",
        "eval_success_rate",
        *metric_fields,
        "mean_duplicate_context_rate",
        "mean_raw_duplicate_rate",
        "mean_metadata_match_rate",
        "mean_company_match_rate",
        "mean_year_match_rate",
        "mean_numeric_accuracy",
        "mean_strict_numeric_accuracy",
        "mean_tolerant_numeric_accuracy",
        "mean_exact_match",
        "mean_literal_exact_match",
        "mean_canonical_exact_match",
        "mean_relative_error",
        "median_relative_error",
        "numeric_parse_success_rate",
        "mean_non_empty_answer_rate",
        "mean_answer_coverage_rate",
        "mean_abstention_rate",
        "mean_answer_relevancy",
        "mean_retrieval_time_ms",
        "mean_generation_time_ms",
        "mean_total_latency_ms",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_total_tokens",
        "mean_estimated_cost",
        "total_questions",
        "generation_failure_count",
        "generation_failure_rate",
        "max_generation_failure_rate",
        "run_valid",
        "failure_threshold_exceeded",
    ]


def _eval_manifest(
    config_path: str,
    cfg: EvalConfig,
    rag_paths: list[Path],
    qa_path: Path,
    gold_path: Path,
    rag_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    per_question: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    difficulty_summary: list[dict[str, Any]],
    run_validity: dict[str, dict[str, Any]],
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    ks = _metric_ks(cfg)
    invalid_experiments = {
        experiment_id: stats
        for experiment_id, stats in run_validity.items()
        if not stats.get("run_valid", True)
    }
    return {
        "config_path": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "input_result_paths": [str(path) for path in rag_paths],
        "input_result_hashes": {str(path): file_sha256(path) for path in rag_paths},
        "qa_path": str(qa_path),
        "qa_hash": file_sha256(qa_path),
        "gold_contexts_path": str(gold_path),
        "gold_contexts_hash": file_sha256(gold_path) if gold_path.exists() else None,
        "row_counts": {
            "pipeline1_results": len(rag_rows),
            "qa_rows": len(qa_rows),
            "gold_context_rows": len(gold_rows),
            "evaluated_rows": len(per_question),
            "pipeline1_failed_rows": sum(1 for row in per_question if row.get("pipeline1_error")),
            "leaderboard_rows": len(leaderboard),
            "difficulty_summary_rows": len(difficulty_summary),
            "generation_failure_count": sum(int(stats["generation_failure_count"]) for stats in run_validity.values()),
        },
        "retrieval_only": cfg.evaluation.retrieval_only,
        "retrieval_eval_field": cfg.evaluation.retrieval_eval_field,
        "generation_failure_threshold": {
            "max_generation_failure_rate": cfg.evaluation.max_generation_failure_rate,
            "strict_failure_threshold": cfg.evaluation.strict_failure_threshold,
            "run_valid": not invalid_experiments,
            "invalid_experiments": invalid_experiments,
            "warning": (
                "Generation failure threshold exceeded; run_valid=false."
                if invalid_experiments
                else None
            ),
        },
        "leaderboard": {
            "sort_metric": cfg.leaderboard.sort_metric,
            "sort_ascending": cfg.leaderboard.sort_ascending,
        },
        "metrics_used": [
            *[name for k in ks for name in (f"hit_at_{k}", f"recall_at_{k}", f"mrr_at_{k}", f"context_precision_at_{k}")],
            *[f"ndcg_at_{k}" for k in ks],
            "duplicate_context_rate",
            "raw_duplicate_rate",
            "metadata_match_rate",
            "company_match_rate",
            "year_match_rate",
            "numeric_accuracy",
            "strict_numeric_accuracy",
            "tolerant_numeric_accuracy",
            "exact_match",
            "literal_exact_match",
            "canonical_exact_match",
            "relative_error",
            "numeric_parse_success",
            "non_empty_answer_rate",
            "answer_coverage_rate",
            "abstention_rate",
            "answer_relevancy_score",
            "retrieval_time_ms",
            "generation_time_ms",
            "total_latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost",
            "pipeline_success_rate",
            "eval_success_rate",
            "generation_failure_rate",
            "run_valid",
        ],
        "summary_behavior": (
            "mean retrieval and answer metrics use all evaluated rows; generation failures are retained, "
            "score zero for answer correctness, and can mark run_valid=false"
        ),
        "start_timestamp_utc": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_timestamp_utc": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
    }
