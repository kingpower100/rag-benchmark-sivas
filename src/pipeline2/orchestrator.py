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
            for name in ("per_question.jsonl", "per_question.csv", "summary_by_experiment.csv", "leaderboard.csv", "eval_manifest.json"):
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
        qa_by_id = _index_by_id(qa_rows)
        _validate_pipeline1_questions_have_qa(rag_rows, qa_by_id)
        _run_officeqa_smoke_validation(qa_by_id)
        print("[3/6] Loading gold contexts")
        gold_path = _resolve(project_root, cfg.inputs.gold_contexts_path)
        gold_rows = read_jsonl(gold_path)
        gold_by_id = _gold_by_question(gold_rows)

        print("[4/6] Computing automatic metrics")
        per_question = self._evaluate_rows(rag_rows, qa_by_id, gold_by_id, cfg)
        print("[5/6] Aggregating summaries")
        summary = summarize_by_experiment(per_question)
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
            write_csv(run_dir / "leaderboard.csv", leaderboard, leaderboard_fields)
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
            pipeline1_error = row.get("error")
            pipeline_success = 0.0 if pipeline1_error else 1.0
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
            retrieval_eval_ids = _resolve_retrieval_eval_ids(row, retrieved_ids, gold_ids)
            ground_truth = resolve_ground_truth_answer(row, qa_by_id)
            raw_retrieved_ids = row.get("raw_retrieved_original_context_ids")
            if raw_retrieved_ids is not None and not isinstance(raw_retrieved_ids, list):
                raw_retrieved_ids = []
                errors.append("raw_retrieved_original_context_ids must be a list")
            raw_retrieval_eval_ids = _resolve_raw_retrieval_eval_ids(row, raw_retrieved_ids, gold_ids)
            retrieved_metadata = row.get("retrieved_chunk_metadata") or []
            if not isinstance(retrieved_metadata, list):
                retrieved_metadata = []
                errors.append("retrieved_chunk_metadata must be a list")
            answer_metrics = compute_answer_metrics(
                str(row.get("generated_answer", "")),
                ground_truth,
                question=str(row.get("question", "")),
                abstention_patterns=cfg.answer_quality.abstention_patterns,
            )
            if not cfg.answer_quality.enable_numeric_accuracy:
                answer_metrics["numeric_accuracy"] = None
            if pipeline1_error:
                answer_metrics.update(
                    {
                        "numeric_accuracy": 0.0,
                        "exact_match": 0.0,
                        "numeric_parse_success": 0.0,
                        "non_empty_answer_rate": 0.0,
                        "answer_coverage_rate": 0.0,
                        "abstention_rate": 0.0,
                        "answer_relevancy_score": 0.0,
                        "normalized_generated_answer": "",
                        "generated_number": None,
                        "absolute_error": None,
                        "relative_error": None,
                        "answer_match_status": "pipeline1_error",
                    }
                )
            output = {
                "question_id": qid,
                "experiment_id": str(row.get("experiment_id", "")),
                "generated_answer": row.get("generated_answer", ""),
                "ground_truth_answer": ground_truth,
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
                "exact_match": answer_metrics["exact_match"],
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
                **compute_efficiency_metrics(row),
                "pipeline_success": pipeline_success,
                "pipeline1_error": pipeline1_error,
                "evaluation_errors": errors,
            }
            evaluated.append(output)
        return evaluated


def _index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
        if not _has_non_empty_answer(row):
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


def _resolve_retrieval_eval_ids(row: dict[str, Any], retrieved_ids: list[str], gold_ids: list[str]) -> list[str]:
    candidates = [
        retrieved_ids,
        _list_field(row, "retrieved_file_names"),
        _list_field(row, "retrieved_document_ids"),
    ]
    return _best_id_projection(candidates, gold_ids)


def _resolve_raw_retrieval_eval_ids(
    row: dict[str, Any],
    raw_retrieved_ids: list[str] | None,
    gold_ids: list[str],
) -> list[str] | None:
    if raw_retrieved_ids is None:
        return None
    candidates = [
        raw_retrieved_ids,
        _list_field(row, "raw_retrieved_file_names"),
        _list_field(row, "raw_retrieved_document_ids"),
    ]
    return _best_id_projection(candidates, gold_ids)


def _best_id_projection(candidates: list[list[str]], gold_ids: list[str]) -> list[str]:
    gold_set = {str(item) for item in gold_ids if item is not None}
    usable = [candidate for candidate in candidates if candidate]
    if not usable:
        return []
    if not gold_set:
        return usable[0]
    return max(
        usable,
        key=lambda candidate: (
            len({str(item) for item in candidate if item is not None} & gold_set),
            -usable.index(candidate),
        ),
    )


def _list_field(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item).strip()]


def _metric_ks(cfg: EvalConfig) -> list[int]:
    return sorted({int(k) for k in (cfg.retrieval.ks or [cfg.retrieval.k]) if int(k) > 0})


def _per_question_fields(ks: list[int]) -> list[str]:
    metric_fields = []
    for k in ks:
        metric_fields.extend([f"hit_at_{k}", f"recall_at_{k}", f"mrr_at_{k}", f"context_precision_at_{k}", f"ndcg_at_{k}"])
    return [
        "question_id",
        "experiment_id",
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
        "exact_match",
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
        "pipeline_success",
        "pipeline1_error",
        "evaluation_errors",
    ]


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
        "mean_exact_match",
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
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    ks = _metric_ks(cfg)
    return {
        "config_path": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "input_result_paths": [str(path) for path in rag_paths],
        "input_result_hashes": {str(path): file_sha256(path) for path in rag_paths},
        "qa_path": str(qa_path),
        "qa_hash": file_sha256(qa_path),
        "gold_contexts_path": str(gold_path),
        "gold_contexts_hash": file_sha256(gold_path),
        "row_counts": {
            "pipeline1_results": len(rag_rows),
            "qa_rows": len(qa_rows),
            "gold_context_rows": len(gold_rows),
            "evaluated_rows": len(per_question),
            "pipeline1_failed_rows": sum(1 for row in per_question if row.get("pipeline1_error")),
            "leaderboard_rows": len(leaderboard),
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
            "exact_match",
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
        ],
        "summary_behavior": "mean retrieval and answer metrics use all evaluated rows; pipeline1_error rows are retained and score zero for answer correctness",
        "start_timestamp_utc": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_timestamp_utc": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
    }
