from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from src.pipeline2.aggregation.summarizer import summarize_by_category, summarize_by_experiment
from src.pipeline2.io.jsonl import read_jsonl, write_jsonl
from src.pipeline2.io.tabular import write_csv
from src.pipeline2.metrics.answer_metrics import compute_answer_metrics, resolve_ground_truth_answer
from src.pipeline2.metrics.category_metrics import compute_category_metrics, compute_category_routing_report
from src.pipeline2.metrics.embedding_similarity import build_answer_embedder, compute_embedding_similarity
from src.pipeline2.metrics.efficiency_metrics import compute_efficiency_metrics
from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline1.utils.hashing import file_sha256
from tqdm.auto import tqdm

_SIVAS_CATEGORIES = ["Technik", "Vertrieb", "Materialwirtschaft", "Einkauf", "Service"]


class EvaluationOrchestrator:
    def run(self, config_path: str) -> Path:
        start_time = time.time()
        cfg = EvalConfig.from_yaml(config_path)
        project_root = Path(__file__).resolve().parents[2]
        run_dir = project_root / cfg.evaluation.output_dir / cfg.evaluation.eval_run_id
        print(f"Resolved eval output dir: {run_dir}")
        if run_dir.exists() and not cfg.runtime.overwrite:
            raise FileExistsError(f"Evaluation run already exists and overwrite=false: {run_dir}")
        if run_dir.exists() and cfg.runtime.overwrite:
            for name in (
                "per_question.jsonl",
                "per_question_metrics.jsonl",
                "per_question.csv",
                "summary_metrics.json",
                "summary_by_category.csv",
                "summary_by_category.json",
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
            if not resolved.exists():
                print("Real-run audit skipped: Pipeline 1 outputs not found on this machine.")
                audit = _skipped_real_run_audit(config_path, cfg, resolved_rag_paths, start_time)
                _write_audit_reports(run_dir, audit)
                return run_dir
            print(f"Pipeline 1 results path: {resolved}")
            rows = read_jsonl(resolved)
            print(f"Pipeline 1 results rows: {len(rows)}")
            rag_rows.extend(rows)
        print("[2/6] Loading SIVAS QA ground truth")
        questions_path = _resolve(project_root, cfg.inputs.questions_path)
        print(f"SIVAS questions file: {questions_path}")
        questions_rows = read_jsonl(questions_path)
        qa_path = _resolve(project_root, cfg.inputs.qa_path)
        print(f"SIVAS QA ground truth file: {qa_path}")
        qa_rows = read_jsonl(qa_path)
        strict_alignment = build_three_way_alignment_report(questions_rows, qa_rows, [], [])
        _validate_no_duplicate_pipeline1_question_ids(rag_rows)
        qa_by_id = _index_by_id(qa_rows, require_answer=not cfg.evaluation.retrieval_only)
        _validate_pipeline1_questions_have_qa(rag_rows, qa_by_id)
        print("[3/6] Loading SIVAS retrieval evidence")
        gold_path = _resolve(project_root, cfg.inputs.gold_contexts_path)
        print(f"SIVAS retrieval evidence file: {gold_path}")
        gold_rows = read_jsonl(gold_path) if gold_path.exists() else []
        strict_alignment = build_three_way_alignment_report(questions_rows, qa_rows, gold_rows, rag_rows)
        _validate_three_way_alignment(strict_alignment)
        gold_by_id = _gold_by_question(gold_rows)
        _validate_pipeline1_questions_have_gold_contexts(rag_rows, gold_by_id)
        input_diagnostics = build_eval_diagnostics(rag_rows, questions_rows, qa_rows, gold_rows, qa_by_id, gold_by_id, strict_alignment, cfg)
        _print_eval_diagnostics(input_diagnostics)
        _validate_eval_diagnostics(input_diagnostics, cfg)
        leakage_audit = build_leakage_audit(resolved_rag_paths)
        if leakage_audit.get("message"):
            print(leakage_audit["message"])
        _validate_leakage_audit(leakage_audit)

        print("[4/6] Computing automatic metrics")
        per_question = self._evaluate_rows(rag_rows, qa_by_id, gold_by_id, cfg)
        if not per_question:
            raise ValueError("Pipeline 2 evaluated zero rows.")
        reported_metric_comparison = compare_reported_vs_recomputed_metrics(rag_rows, per_question, _metric_ks(cfg))
        if reported_metric_comparison.get("message"):
            print(reported_metric_comparison["message"])
        print("[5/6] Aggregating summaries")
        summary = summarize_by_experiment(per_question)
        run_validity = _run_validity_by_experiment(per_question, cfg.evaluation.max_generation_failure_rate)
        _attach_run_validity(summary, run_validity)
        if cfg.evaluation.strict_failure_threshold:
            _raise_on_failure_threshold(run_validity, cfg.evaluation.max_generation_failure_rate)
        category_summary = summarize_by_category(per_question)
        category_routing_report = compute_category_routing_report(per_question, _SIVAS_CATEGORIES)
        ks = _metric_ks(cfg)
        per_fields = _per_question_fields(ks)
        summary_fields = _summary_fields(ks)
        validity_report = _benchmark_validity_report(per_question, input_diagnostics, strict_alignment, ks)
        print("[6/6] Writing evaluation outputs")
        write_jsonl(run_dir / "per_question.jsonl", per_question)
        write_jsonl(run_dir / "per_question_metrics.jsonl", per_question)
        (run_dir / "summary_metrics.json").write_text(
            json.dumps(
                {
                    "summary_by_experiment": summary,
                    "summary_by_category": category_summary,
                    "run_validity": run_validity,
                    "category_routing": category_routing_report,
                    "benchmark_validity": validity_report,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if cfg.runtime.save_csv:
            write_csv(run_dir / "per_question.csv", per_question, per_fields)
            write_csv(run_dir / "summary_by_category.csv", category_summary, _category_summary_fields(ks))
        (run_dir / "summary_by_category.json").write_text(
            json.dumps(category_summary, indent=2, ensure_ascii=False),
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
                    questions_rows,
                    qa_rows,
                    gold_rows,
                    per_question,
                    summary,
                    run_validity,
                    input_diagnostics,
                    strict_alignment,
                    leakage_audit,
                    reported_metric_comparison,
                    category_routing_report,
                    validity_report,
                    start_time,
                    time.time(),
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        audit_report = _eval_manifest(
            config_path,
            cfg,
            resolved_rag_paths,
            qa_path,
            gold_path,
            rag_rows,
            questions_rows,
            qa_rows,
            gold_rows,
            per_question,
            summary,
            run_validity,
            input_diagnostics,
            strict_alignment,
            leakage_audit,
            reported_metric_comparison,
            category_routing_report,
            validity_report,
            start_time,
            time.time(),
        )
        audit_report["fake_run_detection"] = build_fake_run_detection(
            cfg,
            resolved_rag_paths,
            rag_rows,
            questions_rows,
            per_question,
            reported_metric_comparison,
            leakage_audit,
            run_validity,
        )
        audit_report["linked_pipeline1_runs"] = _linked_pipeline1_runs(resolved_rag_paths)
        audit_report["input_artifact_hashes"] = _artifact_hashes(
            [questions_path, qa_path, gold_path, *resolved_rag_paths]
        )
        audit_report["output_artifact_hashes"] = _artifact_hashes(
            [
                run_dir / "per_question.jsonl",
                run_dir / "per_question_metrics.jsonl",
                run_dir / "summary_metrics.json",
                run_dir / "per_question.csv",
                run_dir / "summary_by_category.csv",
                run_dir / "summary_by_category.json",
                run_dir / "eval_manifest.json",
            ]
        )
        audit_report["final_verdict"] = _verdict_from_audit(audit_report)
        audit_report["strict_audit_pass"] = audit_report["final_verdict"] == "valid"
        _write_audit_reports(run_dir, audit_report)
        return run_dir

    def _evaluate_rows(
        self,
        rag_rows: list[dict[str, Any]],
        qa_by_id: dict[str, dict[str, Any]],
        gold_by_id: dict[str, list[str]],
        cfg: EvalConfig,
    ) -> list[dict[str, Any]]:
        ks = _metric_ks(cfg)
        evaluated = []
        embedder = None
        if not cfg.evaluation.retrieval_only and cfg.embedding_similarity.enabled:
            embedder = build_answer_embedder(
                cfg.embedding_similarity.provider,
                cfg.embedding_similarity.model_name,
                cfg.embedding_similarity.dimensions,
            )
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
                raise ValueError(f"Missing gold context for question {qid}")
            retrieval_eval_ids = _configured_retrieval_eval_ids(row, cfg.evaluation.retrieval_eval_field)
            qa_row = qa_by_id.get(qid, {})
            category_metrics = compute_category_metrics(
                row.get("detected_category"),
                qa_row.get("gold_kategorie"),
            )
            ground_truth = "" if cfg.evaluation.retrieval_only else resolve_ground_truth_answer(row, qa_by_id)
            raw_retrieved_ids = row.get("raw_retrieved_original_context_ids")
            if raw_retrieved_ids is not None and not isinstance(raw_retrieved_ids, list):
                raw_retrieved_ids = []
                errors.append("raw_retrieved_original_context_ids must be a list")
            raw_retrieval_eval_ids = _configured_raw_retrieval_eval_ids(row, cfg.evaluation.retrieval_eval_field)
            if cfg.evaluation.retrieval_only:
                answer_metrics = _null_answer_metrics()
            else:
                answer_metrics = compute_answer_metrics(
                    str(row.get("generated_answer", "")),
                    ground_truth,
                    question=str(row.get("question", "")),
                    abstention_patterns=cfg.answer_quality.abstention_patterns,
                )
                _emb_value = (
                    compute_embedding_similarity(str(row.get("generated_answer", "")), ground_truth, embedder)
                    if embedder is not None
                    else None
                )
                _emb_metric = embedder.metric_name if embedder is not None else "embedding_similarity"
                answer_metrics["embedding_similarity"] = _emb_value if _emb_metric == "embedding_similarity" else None
                answer_metrics["bow_token_overlap_similarity"] = _emb_value if _emb_metric == "bow_token_overlap_similarity" else None
            if generation_failed and not cfg.evaluation.retrieval_only:
                failure_status = "pipeline1_error" if pipeline1_error else "generation_failure"
                _fail_emb_metric = embedder.metric_name if embedder is not None else "embedding_similarity"
                answer_metrics.update(
                    {
                        "exact_match": 0.0,
                        "literal_exact_match": 0.0,
                        "canonical_exact_match": 0.0,
                        "german_canonical_exact_match": 0.0,
                        "umlaut_expanded_exact_match": 0.0,
                        "non_empty_answer_rate": 0.0,
                        "answer_coverage_rate": 0.0,
                        "abstention_rate": 0.0,
                        "answer_relevancy_score": 0.0,
                        "rouge_l": 0.0,
                        "rouge_1": 0.0,
                        "embedding_similarity": 0.0 if _fail_emb_metric == "embedding_similarity" else None,
                        "bow_token_overlap_similarity": 0.0 if _fail_emb_metric == "bow_token_overlap_similarity" else None,
                        "normalized_generated_answer": "",
                        "answer_match_status": failure_status,
                    }
                )
            # UNKNOWN-specific flag (distinct from general abstention)
            # Covers English "UNKNOWN" and German "UNBEKANNT" as the canonical unknown sentinel.
            generated_str = str(row.get("generated_answer", ""))
            _unknown_sentinels = {"unknown", "unbekannt"}
            is_unknown = 1.0 if generated_str.strip().lower() in _unknown_sentinels else 0.0

            retrieval_metrics = compute_retrieval_metrics_for_ks(retrieval_eval_ids, gold_ids, ks, raw_retrieval_eval_ids)

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
                **retrieval_metrics,
                "exact_match": answer_metrics["exact_match"],
                "literal_exact_match": answer_metrics["literal_exact_match"],
                "canonical_exact_match": answer_metrics["canonical_exact_match"],
                "german_canonical_exact_match": answer_metrics.get("german_canonical_exact_match"),
                "umlaut_expanded_exact_match": answer_metrics.get("umlaut_expanded_exact_match"),
                "non_empty_answer_rate": answer_metrics["non_empty_answer_rate"],
                "answer_coverage_rate": answer_metrics["answer_coverage_rate"],
                "abstention_rate": answer_metrics["abstention_rate"],
                "is_unknown": is_unknown,
                "answer_relevancy_score": answer_metrics["answer_relevancy_score"],
                "rouge_l": answer_metrics["rouge_l"],
                "rouge_1": answer_metrics.get("rouge_1"),
                "embedding_similarity": answer_metrics["embedding_similarity"],
                "bow_token_overlap_similarity": answer_metrics.get("bow_token_overlap_similarity"),
                "normalized_generated_answer": answer_metrics["normalized_generated_answer"],
                "normalized_gold_answer": answer_metrics["normalized_gold_answer"],
                "answer_match_status": answer_metrics["answer_match_status"],
                "category_correct": category_metrics["category_correct"],
                "category_predicted": category_metrics["category_predicted"],
                "category_gold": category_metrics["category_gold"],
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
    for key in (
        "ground_truth_answer",
        "answer",
        "gold_answer",
        "expected_answer",
        "program_answer",
        "original_answer",
        "referenzantwort",
    ):
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


def _validate_pipeline1_questions_have_gold_contexts(
    rag_rows: list[dict[str, Any]],
    gold_by_id: dict[str, list[str]],
) -> None:
    missing = [
        str(row.get("question_id", ""))
        for row in rag_rows
        if not gold_by_id.get(str(row.get("question_id", "")))
    ]
    if missing:
        sample = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(f"Missing gold context for question {sample}{suffix}")


def _validate_no_duplicate_pipeline1_question_ids(rag_rows: list[dict[str, Any]]) -> None:
    keys = [_experiment_question_key(row) for row in rag_rows if _experiment_question_key(row)[1]]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        sample = ", ".join(_format_experiment_question_key(key) for key in duplicates[:20])
        suffix = "" if len(duplicates) <= 20 else f", ... ({len(duplicates)} total)"
        raise ValueError(
            "Pipeline 1 result files contain duplicate question_id values within the same experiment: "
            f"{sample}{suffix}"
        )


def build_three_way_alignment_report(
    questions_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    rag_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    questions = _id_collection_report(questions_rows, "questions_fixed.jsonl")
    qa = _id_collection_report(qa_rows, "qa_ground_truth_fixed.jsonl")
    gold = _id_collection_report(gold_rows, "qa_ground_truth_fixed.jsonl")
    sets = {
        "questions": set(questions["ids"]),
        "qa_ground_truth": set(qa["ids"]),
        "retrieval_evidence": set(gold["ids"]),
    }
    union = set().union(*sets.values()) if sets else set()
    exact = sets["questions"] == sets["qa_ground_truth"] == sets["retrieval_evidence"]
    report = {
        "files": {
            "questions": questions,
            "qa_ground_truth": qa,
            "retrieval_evidence": gold,
        },
        "aligned_id_count": len(sets["questions"] & sets["qa_ground_truth"] & sets["retrieval_evidence"]),
        "exact_set_equality": exact,
        "missing_from_questions": sorted(union - sets["questions"]),
        "missing_from_qa_ground_truth": sorted(union - sets["qa_ground_truth"]),
        "missing_from_retrieval_evidence": sorted(union - sets["retrieval_evidence"]),
        "extra_in_questions": sorted(sets["questions"] - (sets["qa_ground_truth"] & sets["retrieval_evidence"])),
        "extra_in_qa_ground_truth": sorted(sets["qa_ground_truth"] - (sets["questions"] & sets["retrieval_evidence"])),
        "extra_in_retrieval_evidence": sorted(sets["retrieval_evidence"] - (sets["questions"] & sets["qa_ground_truth"])),
        "duplicate_id_summary": {
            "questions": questions["duplicates"],
            "qa_ground_truth": qa["duplicates"],
            "retrieval_evidence": gold["duplicates"],
            "pipeline1_results": _duplicate_experiment_question_ids_from_rows(rag_rows or []),
        },
    }
    return report


def _validate_three_way_alignment(report: dict[str, Any]) -> None:
    duplicates = {
        name: values
        for name, values in report["duplicate_id_summary"].items()
        if values
    }
    if duplicates:
        pieces = [f"{name}: {', '.join(values[:10])}" for name, values in duplicates.items()]
        raise ValueError(f"Strict audit failed because duplicate IDs were found. {'; '.join(pieces)}")
    if not report["exact_set_equality"]:
        raise ValueError(
            "Strict audit failed because questions_fixed.jsonl and qa_ground_truth_fixed.jsonl ID sets are not identical. "
            f"missing_from_questions={report['missing_from_questions'][:10]} "
            f"missing_from_qa_ground_truth={report['missing_from_qa_ground_truth'][:10]} "
            f"missing_from_retrieval_evidence={report['missing_from_retrieval_evidence'][:10]}"
        )


def _id_collection_report(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    ids = []
    missing_rows = []
    key_counts = {"uid": 0, "id": 0, "question_id": 0}
    for line_number, row in enumerate(rows, start=1):
        key_name, raw_id = _resolve_qa_row_id(row)
        if raw_id is None or str(raw_id).strip() == "":
            missing_rows.append(line_number)
            continue
        qid = str(raw_id).strip()
        ids.append(qid)
        if key_name in key_counts:
            key_counts[key_name] += 1
    duplicates = sorted(qid for qid, count in Counter(ids).items() if count > 1)
    return {
        "label": label,
        "row_count": len(rows),
        "unique_id_count": len(set(ids)),
        "ids": sorted(set(ids)),
        "duplicates": duplicates,
        "missing_id_rows": missing_rows,
        "id_field_counts": key_counts,
    }


def _duplicate_ids_from_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
    ids = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                ids.append(str(value).strip())
                break
    return sorted(qid for qid, count in Counter(ids).items() if count > 1)


def _duplicate_experiment_question_ids_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    keys = [_experiment_question_key(row) for row in rows if _experiment_question_key(row)[1]]
    return [_format_experiment_question_key(key) for key, count in sorted(Counter(keys).items()) if count > 1]


def _experiment_question_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("experiment_id", "")).strip(), str(row.get("question_id", "")).strip())


def _format_experiment_question_key(key: tuple[str, str]) -> str:
    experiment_id, question_id = key
    return f"{experiment_id or '<missing_experiment>'}:{question_id}"


def build_eval_diagnostics(
    rag_rows: list[dict[str, Any]],
    questions_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    qa_by_id: dict[str, dict[str, Any]],
    gold_by_id: dict[str, list[str]],
    strict_alignment: dict[str, Any],
    cfg: EvalConfig,
) -> dict[str, Any]:
    rag_ids = [str(row.get("question_id", "")) for row in rag_rows if str(row.get("question_id", "")).strip()]
    qa_ids = list(qa_by_id)
    gold_ids = [qid for qid, ids in gold_by_id.items() if ids]
    rag_set = set(rag_ids)
    qa_set = set(qa_ids)
    gold_set = set(gold_ids)
    retrieved_field = cfg.evaluation.retrieval_eval_field
    generated_present = [row for row in rag_rows if str(row.get("generated_answer", "")).strip()]
    retrieved_present = [
        row
        for row in rag_rows
        if isinstance(row.get(retrieved_field), list) and any(str(item).strip() for item in row.get(retrieved_field) or [])
    ]
    missing_generated_ids = [str(row.get("question_id", "")) for row in rag_rows if not str(row.get("generated_answer", "")).strip()]
    missing_retrieved_ids = [
        str(row.get("question_id", ""))
        for row in rag_rows
        if not (isinstance(row.get(retrieved_field), list) and any(str(item).strip() for item in row.get(retrieved_field) or []))
    ]
    return {
        "pipeline1_result_rows": len(rag_rows),
        "questions_rows": len(questions_rows),
        "qa_rows": len(qa_rows),
        "gold_context_rows": len(gold_rows),
        "qa_indexed_rows": len(qa_by_id),
        "gold_indexed_rows": len(gold_set),
        "qa_intersection_size": len(rag_set & qa_set),
        "gold_intersection_size": len(rag_set & gold_set),
        "evaluated_rows_expected": len(rag_rows),
        "skipped_rows": 0,
        "missing_generated_answers": len(missing_generated_ids),
        "missing_retrieved_field_values": len(missing_retrieved_ids),
        "generated_answer_coverage": len(generated_present) / len(rag_rows) if rag_rows else 0.0,
        "retrieved_field": retrieved_field,
        "retrieved_field_coverage": len(retrieved_present) / len(rag_rows) if rag_rows else 0.0,
        "first_5_pipeline1_question_ids": rag_ids[:5],
        "first_5_qa_question_ids": qa_ids[:5],
        "first_5_gold_question_ids": gold_ids[:5],
        "missing_in_qa_examples": sorted(rag_set - qa_set)[:5],
        "missing_in_gold_examples": sorted(rag_set - gold_set)[:5],
        "missing_generated_answer_examples": missing_generated_ids[:5],
        "missing_retrieved_field_examples": missing_retrieved_ids[:5],
        "strict_alignment": strict_alignment,
        "retrieval_level": "document",
        "chunk_level_metrics": "not_computed_no_chunk_gold_available",
    }


def _print_eval_diagnostics(diagnostics: dict[str, Any]) -> None:
    print(
        "Evaluation input diagnostics: "
        f"pipeline1_rows={diagnostics['pipeline1_result_rows']} "
        f"qa_rows={diagnostics['qa_rows']} "
        f"gold_rows={diagnostics['gold_context_rows']} "
        f"qa_intersection={diagnostics['qa_intersection_size']} "
        f"gold_intersection={diagnostics['gold_intersection_size']} "
        f"generated_answer_coverage={diagnostics['generated_answer_coverage']:.3f} "
        f"retrieved_field={diagnostics['retrieved_field']} "
        f"retrieved_field_coverage={diagnostics['retrieved_field_coverage']:.3f}"
    )


def _validate_eval_diagnostics(diagnostics: dict[str, Any], cfg: EvalConfig) -> None:
    if diagnostics["pipeline1_result_rows"] == 0:
        raise ValueError("Pipeline 2 loaded zero Pipeline 1 result rows.")
    if diagnostics["evaluated_rows_expected"] == 0:
        raise ValueError("Pipeline 2 would evaluate zero rows.")
    if diagnostics["qa_intersection_size"] == 0:
        raise ValueError("Pipeline 2 found zero matching question IDs between Pipeline 1 results and QA file.")
    if diagnostics["gold_intersection_size"] == 0:
        raise ValueError("Pipeline 2 found zero matching question IDs between Pipeline 1 results and gold contexts/source_files.")
    if not cfg.evaluation.retrieval_only and diagnostics["generated_answer_coverage"] == 0.0:
        raise ValueError("Pipeline 2 found no generated_answer values in Pipeline 1 results.")
    if diagnostics["retrieved_field_coverage"] == 0.0:
        raise ValueError(
            f"Pipeline 2 found no non-empty values for retrieval_eval_field={diagnostics['retrieved_field']!r}."
        )


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _gold_by_question(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    duplicates: set[str] = set()
    missing_rows: list[int] = []
    for line_number, row in enumerate(rows, start=1):
        _, raw_id = _resolve_qa_row_id(row)
        if raw_id is None or str(raw_id).strip() == "":
            missing_rows.append(line_number)
            continue
        qid = str(raw_id).strip()
        if qid in output:
            duplicates.add(qid)
        ids = row.get("context_id")
        if ids is None:
            ids = row.get("source_files")
        if ids is None:
            ids = [
                evidence.get("source_document")
                for evidence in row.get("partner_retrieval_evidence", [])
                if isinstance(evidence, dict) and evidence.get("source_document")
            ]
        if ids is None:
            ids = []
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list):
            ids = []
        output[qid] = [str(item) for item in ids if item is not None and str(item).strip()]
    if missing_rows:
        sample = ", ".join(str(item) for item in missing_rows[:20])
        suffix = "" if len(missing_rows) <= 20 else f", ... ({len(missing_rows)} total)"
        raise ValueError(f"Gold context rows are missing uid/id/question_id on line(s): {sample}{suffix}")
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:20])
        suffix = "" if len(duplicates) <= 20 else f", ... ({len(duplicates)} total)"
        raise ValueError(f"Gold context rows contain duplicate resolved IDs: {sample}{suffix}")
    return output


def _merge_gold_with_qa_fallback(gold_by_id: dict[str, list[str]], qa_by_id: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    raise RuntimeError(
        "QA source_files fallback for retrieval gold is disabled. "
        "Use qa_ground_truth_fixed.jsonl retrieval evidence entries only."
    )


def _null_answer_metrics() -> dict[str, Any]:
    return {
        "exact_match": None,
        "literal_exact_match": None,
        "canonical_exact_match": None,
        "german_canonical_exact_match": None,
        "umlaut_expanded_exact_match": None,
        "non_empty_answer_rate": None,
        "answer_coverage_rate": None,
        "abstention_rate": None,
        "answer_relevancy_score": None,
        "rouge_l": None,
        "rouge_1": None,
        "embedding_similarity": None,
        "bow_token_overlap_similarity": None,
        "normalized_generated_answer": "",
        "normalized_gold_answer": "",
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
        metric_fields.extend([
            f"hit_at_{k}",
            f"recall_at_{k}",
            f"mrr_at_{k}",
            f"context_precision_at_{k}",
            f"ndcg_at_{k}",
            f"duplicate_count_at_{k}",
            f"duplicate_rate_at_{k}",
            f"deduped_hit_at_{k}",
            f"deduped_recall_at_{k}",
            f"deduped_mrr_at_{k}",
            f"deduped_ndcg_at_{k}",
        ])
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
        "raw_retrieved_count",
        "unique_retrieved_document_count",
        "duplicate_document_count",
        "duplicate_document_rate",
        "exact_match",
        "literal_exact_match",
        "canonical_exact_match",
        "german_canonical_exact_match",
        "umlaut_expanded_exact_match",
        "non_empty_answer_rate",
        "answer_coverage_rate",
        "abstention_rate",
        "is_unknown",
        "answer_relevancy_score",
        "rouge_l",
        "rouge_1",
        "embedding_similarity",
        "bow_token_overlap_similarity",
        "normalized_generated_answer",
        "normalized_gold_answer",
        "answer_match_status",
        "category_correct",
        "category_predicted",
        "category_gold",
        "retrieval_time_ms",
        "rerank_time_ms",
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


def _benchmark_validity_report(
    per_question: list[dict[str, Any]],
    input_diagnostics: dict[str, Any],
    strict_alignment: dict[str, Any],
    ks: list[int],
) -> dict[str, Any]:
    """Produce a benchmark_validity_status of VALID, WARNING, or INVALID."""
    blocking_issues: list[str] = []
    warnings_list: list[str] = []

    # Hard check: NDCG must be in [0, 1]
    for row in per_question:
        qid = row.get("question_id", "?")
        for k in ks:
            for prefix in (f"ndcg_at_{k}", f"deduped_ndcg_at_{k}"):
                val = row.get(prefix)
                if val is not None and val > 1.0 + 1e-9:
                    blocking_issues.append(f"{prefix}={val:.6f} > 1.0 for question_id={qid}")

    # Hard check: duplicate gold IDs
    dup_summary = strict_alignment.get("duplicate_id_summary", {})
    for src, dups in dup_summary.items():
        if dups:
            blocking_issues.append(f"Duplicate IDs in {src}: {dups[:5]}")

    # Hard check: missing question IDs
    if strict_alignment.get("missing_from_questions"):
        blocking_issues.append(f"IDs missing from questions: {strict_alignment['missing_from_questions'][:5]}")
    if strict_alignment.get("missing_from_qa_ground_truth"):
        blocking_issues.append(f"IDs missing from qa_ground_truth: {strict_alignment['missing_from_qa_ground_truth'][:5]}")

    # Soft check: answer and retrieval coverage
    answer_coverage = input_diagnostics.get("generated_answer_coverage", 1.0)
    if answer_coverage < 1.0:
        warnings_list.append(f"answer_coverage={answer_coverage:.3f} < 1.0")

    retrieval_coverage = input_diagnostics.get("retrieved_field_coverage", 1.0)
    if retrieval_coverage < 1.0:
        warnings_list.append(f"retrieval_coverage={retrieval_coverage:.3f} < 1.0")

    if blocking_issues:
        status = "INVALID"
    elif warnings_list:
        status = "WARNING"
    else:
        status = "VALID"

    return {
        "benchmark_validity_status": status,
        "blocking_issues": blocking_issues,
        "warnings": warnings_list,
    }


def compare_reported_vs_recomputed_metrics(
    reported_rows: list[dict[str, Any]],
    recomputed_rows: list[dict[str, Any]],
    ks: list[int],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    by_id = {_experiment_question_key(row): row for row in recomputed_rows}
    metric_names = [
        *[name for k in ks for name in (
            f"hit_at_{k}",
            f"recall_at_{k}",
            f"mrr_at_{k}",
            f"ndcg_at_{k}",
            f"context_precision_at_{k}",
        )],
        "exact_match",
        "german_canonical_exact_match",
        "umlaut_expanded_exact_match",
        "non_empty_answer_rate",
        "abstention_rate",
        "rouge_l",
        "rouge_1",
        "embedding_similarity",
        "bow_token_overlap_similarity",
    ]
    comparisons = []
    for reported in reported_rows:
        recomputed = by_id.get(_experiment_question_key(reported))
        if recomputed is None:
            continue
        for name in metric_names:
            if name not in reported:
                continue
            recomputed_name = name
            if recomputed_name not in recomputed:
                continue
            reported_value = _as_float_or_none(reported.get(name))
            recomputed_value = _as_float_or_none(recomputed.get(recomputed_name))
            if reported_value is None or recomputed_value is None:
                continue
            difference = abs(reported_value - recomputed_value)
            comparisons.append(
                {
                    "question_id": str(reported.get("question_id", "")),
                    "metric": name,
                    "reported_value": reported_value,
                    "recomputed_value": recomputed_value,
                    "absolute_difference": difference,
                    "passed": difference <= tolerance,
                }
            )
    failed = [item for item in comparisons if not item["passed"]]
    return {
        "message": "No reported metrics found to compare against." if not comparisons else None,
        "tolerance": tolerance,
        "comparison_count": len(comparisons),
        "failure_count": len(failed),
        "passed": bool(comparisons) and not failed,
        "comparisons": comparisons[:200],
        "failed_examples": failed[:20],
    }


def build_leakage_audit(rag_paths: list[Path]) -> dict[str, Any]:
    forbidden_terms = (
        "answer-bearing question file",
        "legacy gold context file",
        "gold_answer",
        "ground_truth_answer",
        "expected_answer",
        "program_answer",
        "original_answer",
        "gold_context_id",
        "gold_context_ids",
        "context_id",
        "source_files",
    )
    findings = []
    checked_files = []
    for rag_path in rag_paths:
        run_dir = rag_path.parent
        candidates = [
            run_dir / "run_manifest.json",
            run_dir / "logs.txt",
            run_dir / "config.yaml",
            run_dir / "pipeline1_config.yaml",
            *sorted(run_dir.glob("*prompt*")),
            *sorted(run_dir.glob("*config*.json")),
            *sorted(run_dir.glob("*config*.yaml")),
            *sorted(run_dir.glob("*config*.yml")),
        ]
        seen_candidates: set[Path] = set()
        for path in candidates:
            if path in seen_candidates:
                continue
            seen_candidates.add(path)
            if not path.exists() or not path.is_file():
                continue
            checked_files.append(str(path))
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").casefold()
            except OSError:
                continue
            for term in forbidden_terms:
                if term.casefold() in text:
                    findings.append({"path": str(path), "term": term})
    if not checked_files:
        return {
            "checked_files": [],
            "critical_leakage_found": False,
            "findings": [],
            "result": "skipped",
            "message": "Leakage audit skipped: Pipeline 1 artifacts not found.",
        }
    return {
        "checked_files": checked_files,
        "critical_leakage_found": bool(findings),
        "findings": findings[:100],
        "result": "fail" if findings else "pass",
        "message": None,
    }


def _validate_leakage_audit(report: dict[str, Any]) -> None:
    if report.get("critical_leakage_found"):
        examples = ", ".join(f"{item['term']} in {item['path']}" for item in report.get("findings", [])[:5])
        raise ValueError(f"Strict audit failed because possible gold-data leakage was found: {examples}")


def _as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_fake_run_detection(
    cfg: EvalConfig,
    rag_paths: list[Path],
    rag_rows: list[dict[str, Any]],
    questions_rows: list[dict[str, Any]],
    per_question: list[dict[str, Any]],
    reported_metric_comparison: dict[str, Any],
    leakage_audit: dict[str, Any],
    run_validity: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = []
    checks.append(_fake_check("pipeline1_result_file_missing", any(not path.exists() for path in rag_paths), [str(path) for path in rag_paths if not path.exists()]))
    checks.append(_fake_check("pipeline1_result_file_zero_rows", len(rag_rows) == 0, []))
    row_count_report = _result_row_count_report(rag_rows, questions_rows)
    checks.append(
        _fake_check(
            "result_rows_do_not_match_questions",
            row_count_report["suspicious"],
            row_count_report,
        )
    )
    duplicate_result_ids = _duplicate_experiment_question_ids_from_rows(rag_rows)
    checks.append(_fake_check("duplicate_pipeline1_result_question_ids_within_experiment", bool(duplicate_result_ids), duplicate_result_ids[:50]))
    checks.append(_fake_check("pipeline2_metrics_exist_but_raw_rows_missing", bool(per_question) and not rag_rows, []))
    checks.append(_fake_check("reported_metrics_differ_from_recomputed", reported_metric_comparison.get("failure_count", 0) > 0, reported_metric_comparison.get("failed_examples", [])))
    checks.append(_fake_check("leakage_detected", leakage_audit.get("critical_leakage_found", False), leakage_audit.get("findings", [])))
    checks.extend(_fake_run_row_checks(cfg, rag_rows))
    checks.extend(_pipeline1_manifest_checks(cfg, rag_paths, rag_rows))
    suspicious = [check for check in checks if check["suspicious"]]
    return {
        "suspicious": bool(suspicious),
        "suspicious_count": len(suspicious),
        "checks": checks,
        "suspicious_examples": suspicious[:20],
    }


def _result_row_count_report(rag_rows: list[dict[str, Any]], questions_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_questions = len(questions_rows)
    by_experiment: dict[str, int] = {}
    for row in rag_rows:
        by_experiment.setdefault(str(row.get("experiment_id", "")), 0)
        by_experiment[str(row.get("experiment_id", ""))] += 1
    mismatches = {
        experiment_id: count
        for experiment_id, count in by_experiment.items()
        if expected_questions and count != expected_questions
    }
    return {
        "suspicious": bool(expected_questions and mismatches),
        "result_rows": len(rag_rows),
        "question_rows": expected_questions,
        "experiment_count": len(by_experiment),
        "rows_by_experiment": by_experiment,
        "mismatches": mismatches,
    }


def _fake_run_row_checks(cfg: EvalConfig, rag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rag_rows:
        return [
            _fake_check("many_generated_answers_identical", False, {}),
            _fake_check("all_retrieval_lists_empty", False, {}),
            _fake_check("all_latencies_zero_or_missing", False, {}),
            _fake_check("result_model_fields_mismatch_config", False, {}),
        ]
    answers = [str(row.get("generated_answer", "")).strip() for row in rag_rows]
    non_empty_answers = [answer for answer in answers if answer]
    answer_counts = Counter(non_empty_answers)
    most_common_answer, most_common_count = answer_counts.most_common(1)[0] if answer_counts else ("", 0)
    repeated_answer_rate = most_common_count / len(non_empty_answers) if non_empty_answers else 0.0
    retrieval_field = cfg.evaluation.retrieval_eval_field
    all_retrieval_empty = all(not _list_field(row, retrieval_field) for row in rag_rows)
    latencies = [
        _as_float_or_none(row.get("total_latency_ms", row.get("latency_ms")))
        for row in rag_rows
    ]
    all_latency_missing_or_zero = all(value is None or value == 0.0 for value in latencies)
    mismatches = []
    llm_models = {str(row.get("llm_model")) for row in rag_rows if row.get("llm_model")}
    embedding_models = {str(row.get("embedding_model")) for row in rag_rows if row.get("embedding_model")}
    retriever_types = {str(row.get("retriever_type")) for row in rag_rows if row.get("retriever_type")}
    if len(llm_models) > 1:
        mismatches.append({"field": "llm_model", "values": sorted(llm_models)})
    if len(embedding_models) > 1:
        mismatches.append({"field": "embedding_model", "values": sorted(embedding_models)})
    if len(retriever_types) > 1:
        mismatches.append({"field": "retriever_type", "values": sorted(retriever_types)})
    return [
        _fake_check(
            "many_generated_answers_identical",
            len(non_empty_answers) >= 5 and repeated_answer_rate >= 0.8,
            {"answer": most_common_answer, "count": most_common_count, "rate": repeated_answer_rate},
        ),
        _fake_check("all_retrieval_lists_empty", all_retrieval_empty, {"field": retrieval_field}),
        _fake_check("all_latencies_zero_or_missing", all_latency_missing_or_zero, {}),
        _fake_check("result_model_fields_mismatch_config", bool(mismatches), mismatches),
    ]


def _pipeline1_manifest_checks(cfg: EvalConfig, rag_paths: list[Path], rag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for rag_path in rag_paths:
        manifest_path = rag_path.parent / "run_manifest.json"
        if not manifest_path.exists():
            manifest_path = rag_path.parent / "manifest.json"
        if not manifest_path.exists():
            checks.append(_fake_check("pipeline1_manifest_missing", True, {"result_path": str(rag_path)}))
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as ex:
            checks.append(_fake_check("pipeline1_manifest_unreadable", True, {"path": str(manifest_path), "error": str(ex)}))
            continue
        expected_hash = (
            manifest.get("artifacts", {}).get("results.jsonl", {}).get("sha256")
            or manifest.get("output_artifacts", {}).get("results.jsonl", {}).get("sha256")
        )
        actual_hash = file_sha256(rag_path) if rag_path.exists() else None
        checks.append(
            _fake_check(
                "hash_mismatch_between_manifest_and_evaluated_file",
                bool(expected_hash and actual_hash and expected_hash != actual_hash),
                {"manifest_path": str(manifest_path), "expected": expected_hash, "actual": actual_hash},
            )
        )
        stats = manifest.get("run_stats", {})
        manifest_questions = stats.get("n_queries")
        if manifest_questions is not None:
            result_row_count = _jsonl_row_count(rag_path) if rag_path.exists() else len(rag_rows)
            checks.append(
                _fake_check(
                    "pipeline1_manifest_question_count_mismatch",
                    int(manifest_questions) != result_row_count,
                    {"manifest_n_queries": manifest_questions, "result_rows": result_row_count},
                )
            )
        checks.append(
            _fake_check(
                "timestamps_impossible_or_missing",
                _timestamps_invalid(manifest.get("start_timestamp_utc"), manifest.get("end_timestamp_utc")),
                {
                    "manifest_path": str(manifest_path),
                    "start": manifest.get("start_timestamp_utc"),
                    "end": manifest.get("end_timestamp_utc"),
                },
            )
        )
    return checks


def _jsonl_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _timestamps_invalid(start: str | None, end: str | None) -> bool:
    if not start or not end:
        return True
    try:
        from datetime import datetime

        start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except Exception:
        return True
    return end_dt < start_dt


def _fake_check(name: str, suspicious: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "suspicious": bool(suspicious), "details": details}


def _artifact_hashes(paths: list[Path]) -> dict[str, dict[str, str | int | None]]:
    return {str(path): _artifact_hash(path) for path in paths}


def _artifact_hash(path: Path) -> dict[str, str | int | None]:
    if not path.exists():
        return {"sha256": None, "size_bytes": None, "exists": False}
    return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size, "exists": True}


def _skipped_real_run_audit(config_path: str, cfg: EvalConfig, rag_paths: list[Path], start_time: float) -> dict[str, Any]:
    from datetime import datetime, timezone

    return {
        "final_verdict": "partially_valid",
        "strict_audit_pass": False,
        "audit_status": "skipped",
        "message": "Real-run audit skipped: Pipeline 1 outputs not found on this machine.",
        "config_path": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "evaluation_run_id": cfg.evaluation.eval_run_id,
        "input_result_paths": [str(path) for path in rag_paths],
        "linked_pipeline1_runs": _linked_pipeline1_runs(rag_paths),
        "input_artifact_hashes": _artifact_hashes(rag_paths),
        "fake_run_detection": {
            "suspicious": True,
            "suspicious_count": 1,
            "checks": [_fake_check("pipeline1_result_file_missing", True, [str(path) for path in rag_paths])],
            "suspicious_examples": [_fake_check("pipeline1_result_file_missing", True, [str(path) for path in rag_paths])],
        },
        "start_timestamp_utc": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _linked_pipeline1_runs(rag_paths: list[Path]) -> list[dict[str, Any]]:
    runs = []
    for path in rag_paths:
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.exists():
            manifest_path = path.parent / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        runs.append(
            {
                "result_path": str(path),
                "result_sha256": file_sha256(path) if path.exists() else None,
                "manifest_path": str(manifest_path) if manifest_path.exists() else None,
                "manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else None,
                "pipeline1_run_id": manifest.get("run_id")
                or manifest.get("resolved_config", {}).get("experiment", {}).get("experiment_id")
                or manifest.get("config", {}).get("experiment", {}).get("experiment_id"),
                "manifest_recorded_result_sha256": (
                    manifest.get("artifacts", {}).get("results.jsonl", {}).get("sha256")
                    or manifest.get("output_artifacts", {}).get("results.jsonl", {}).get("sha256")
                ),
            }
        )
    return runs


def _write_audit_reports(run_dir: Path, audit_report: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "audit_report.json").write_text(
        json.dumps(audit_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "audit_report.md").write_text(_audit_report_markdown(audit_report), encoding="utf-8")


def _audit_report_markdown(report: dict[str, Any]) -> str:
    validity = report.get("benchmark_validity") or {}
    lines = [
        "# RAG Benchmark Audit Report",
        "",
        f"- Final verdict: `{report.get('final_verdict')}`",
        f"- Strict audit pass: `{report.get('strict_audit_pass')}`",
        f"- Benchmark validity status: `{validity.get('benchmark_validity_status', 'n/a')}`",
        f"- Total questions: `{report.get('total_questions', 'n/a')}`",
        f"- Aligned ID count: `{report.get('aligned_id_count', 'n/a')}`",
        f"- Message: {report.get('message') or 'n/a'}",
        "",
    ]
    if validity.get("blocking_issues"):
        lines.append("## Blocking Issues")
        for issue in validity["blocking_issues"]:
            lines.append(f"- {issue}")
        lines.append("")
    if validity.get("warnings"):
        lines.append("## Warnings")
        for w in validity["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Fake-Run Detection")
    fake = report.get("fake_run_detection") or {}
    lines.append(f"- Suspicious: `{fake.get('suspicious')}`")
    for check in fake.get("checks", []):
        if check.get("suspicious"):
            lines.append(f"- `{check.get('name')}`: {check.get('details')}")
    lines.extend([
        "",
        "## Metric Comparison",
        f"- Reported-vs-recomputed failures: `{(report.get('reported_vs_recomputed_comparison') or {}).get('failure_count', 'n/a')}`",
        "",
        "## Leakage Audit",
        f"- Result: `{(report.get('leakage_audit_result') or {}).get('result', 'n/a')}`",
        f"- Message: {(report.get('leakage_audit_result') or {}).get('message') or 'n/a'}",
        "",
        "## Category Routing",
    ])
    routing = report.get("category_routing") or {}
    lines.append(f"- Active: `{routing.get('category_routing_active', 'n/a')}`")
    if routing.get("category_routing_active"):
        lines.append(f"- Coverage: `{routing.get('category_routing_coverage', 'n/a')}`")
        lines.append(f"- Accuracy: `{routing.get('category_accuracy', 'n/a')}`")
        lines.append(f"- Macro Precision: `{routing.get('category_precision_macro', 'n/a')}`")
        lines.append(f"- Macro Recall: `{routing.get('category_recall_macro', 'n/a')}`")
    else:
        lines.append(f"- {routing.get('message', 'Category routing inactive.')}")
    return "\n".join(lines) + "\n"


def _verdict_from_audit(report: dict[str, Any]) -> str:
    fake = report.get("fake_run_detection") or {}
    validity = report.get("benchmark_validity") or {}
    if (
        fake.get("suspicious")
        or (report.get("reported_vs_recomputed_comparison") or {}).get("failure_count", 0) > 0
        or validity.get("benchmark_validity_status") == "INVALID"
    ):
        return "invalid"
    if report.get("strict_audit_pass"):
        return "valid"
    return "partially_valid"


def _category_summary_fields(ks: list[int]) -> list[str]:
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
        "category",
        "n_questions",
        "pipeline_success_rate",
        *metric_fields,
        "mean_category_accuracy",
        "mean_exact_match",
        "mean_literal_exact_match",
        "mean_canonical_exact_match",
        "mean_german_canonical_exact_match",
        "mean_umlaut_expanded_exact_match",
        "mean_rouge_l",
        "mean_rouge_1",
        "mean_embedding_similarity",
        "mean_bow_token_overlap_similarity",
        "mean_abstention_rate",
        "unknown_count",
        "unknown_rate",
        "mean_total_latency_ms",
        "mean_total_tokens",
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
        "mean_raw_retrieved_count",
        "mean_unique_retrieved_document_count",
        "mean_duplicate_document_count",
        "mean_duplicate_document_rate",
        "mean_exact_match",
        "mean_literal_exact_match",
        "mean_canonical_exact_match",
        "mean_german_canonical_exact_match",
        "mean_umlaut_expanded_exact_match",
        "mean_non_empty_answer_rate",
        "mean_answer_coverage_rate",
        "mean_abstention_rate",
        "unknown_count",
        "unknown_rate",
        "mean_rouge_l",
        "mean_rouge_1",
        "mean_embedding_similarity",
        "mean_bow_token_overlap_similarity",
        # diagnostic — lexical overlap between question and answer tokens, not a quality score
        "diagnostic_mean_answer_relevancy",
        "mean_category_accuracy",
        "mean_retrieval_time_ms",
        "mean_rerank_time_ms",
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
    questions_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    per_question: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    run_validity: dict[str, dict[str, Any]],
    input_diagnostics: dict[str, Any],
    strict_alignment: dict[str, Any],
    leakage_audit: dict[str, Any],
    reported_metric_comparison: dict[str, Any],
    category_routing_report: dict[str, Any],
    validity_report: dict[str, Any],
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
    comparison_pass = reported_metric_comparison.get("failure_count", 0) == 0
    blocking_audit_pass = (
        strict_alignment.get("exact_set_equality") is True
        and not any(strict_alignment.get("duplicate_id_summary", {}).values())
        and not leakage_audit.get("critical_leakage_found")
        and comparison_pass
    )
    strict_pass = blocking_audit_pass and not invalid_experiments
    final_verdict = "valid" if strict_pass else ("partially_valid" if blocking_audit_pass else "invalid")

    # Use experiment-level summary for aggregate metric reporting
    all_summary = summary[0] if summary else {}

    return {
        "final_verdict": final_verdict,
        "strict_audit_pass": strict_pass,
        "total_questions": len(per_question),
        "aligned_id_count": strict_alignment.get("aligned_id_count"),
        "duplicate_id_summary": strict_alignment.get("duplicate_id_summary"),
        "missing_extra_id_summary": {
            "missing_from_questions": strict_alignment.get("missing_from_questions", [])[:100],
            "missing_from_qa_ground_truth": strict_alignment.get("missing_from_qa_ground_truth", [])[:100],
            "missing_from_retrieval_evidence": strict_alignment.get("missing_from_retrieval_evidence", [])[:100],
            "extra_in_questions": strict_alignment.get("extra_in_questions", [])[:100],
            "extra_in_qa_ground_truth": strict_alignment.get("extra_in_qa_ground_truth", [])[:100],
            "extra_in_retrieval_evidence": strict_alignment.get("extra_in_retrieval_evidence", [])[:100],
            "exact_set_equality": strict_alignment.get("exact_set_equality"),
        },
        "recomputed_retrieval_metrics": {
            key: value
            for key, value in all_summary.items()
            if key.startswith((
                "mean_hit_at_",
                "mean_recall_at_",
                "mean_mrr_at_",
                "mean_context_precision_at_",
                "mean_ndcg_at_",
            ))
        },
        "recomputed_answer_metrics": {
            key: all_summary.get(key)
            for key in (
                "mean_exact_match",
                "mean_literal_exact_match",
                "mean_canonical_exact_match",
                "mean_german_canonical_exact_match",
                "mean_umlaut_expanded_exact_match",
                "mean_non_empty_answer_rate",
                "mean_abstention_rate",
                "unknown_count",
                "unknown_rate",
                "mean_rouge_l",
                "mean_rouge_1",
                "mean_embedding_similarity",
                "mean_bow_token_overlap_similarity",
            )
            if key in all_summary
        },
        "reported_vs_recomputed_comparison": reported_metric_comparison,
        "duplicate_retrieval_statistics": {
            "mean_raw_retrieved_count": _mean([row.get("raw_retrieved_count") for row in per_question]),
            "mean_unique_retrieved_document_count": _mean([row.get("unique_retrieved_document_count") for row in per_question]),
            "mean_duplicate_document_count": _mean([row.get("duplicate_document_count") for row in per_question]),
            "mean_duplicate_document_rate": _mean([row.get("duplicate_document_rate") for row in per_question]),
        },
        "leakage_audit_result": leakage_audit,
        "category_routing": category_routing_report,
        "benchmark_validity": validity_report,
        "suspicious_examples": {
            "missing_generated_answer_examples": input_diagnostics.get("missing_generated_answer_examples", []),
            "missing_retrieved_field_examples": input_diagnostics.get("missing_retrieved_field_examples", []),
            "reported_metric_mismatch_examples": reported_metric_comparison.get("failed_examples", []),
            "leakage_examples": leakage_audit.get("findings", [])[:20],
        },
        "config_path": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "input_result_paths": [str(path) for path in rag_paths],
        "input_result_hashes": {str(path): file_sha256(path) for path in rag_paths},
        "questions_path": str(_resolve(Path(__file__).resolve().parents[2], cfg.inputs.questions_path)),
        "qa_path": str(qa_path),
        "qa_hash": file_sha256(qa_path),
        "gold_contexts_path": str(gold_path),
        "gold_contexts_hash": file_sha256(gold_path) if gold_path.exists() else None,
        "row_counts": {
            "pipeline1_results": len(rag_rows),
            "questions_rows": len(questions_rows),
            "qa_rows": len(qa_rows),
            "gold_context_rows": len(gold_rows),
            "evaluated_rows": len(per_question),
            "pipeline1_failed_rows": sum(1 for row in per_question if row.get("pipeline1_error")),
            "generation_failure_count": sum(int(stats["generation_failure_count"]) for stats in run_validity.values()),
        },
        "input_diagnostics": input_diagnostics,
        "retrieval_only": cfg.evaluation.retrieval_only,
        "retrieval_eval_field": cfg.evaluation.retrieval_eval_field,
        "retrieval_level": "document",
        "chunk_level_metrics": "not_computed_no_chunk_gold_available",
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
        "metrics_used": [
            *[name for k in ks for name in (f"hit_at_{k}", f"recall_at_{k}", f"mrr_at_{k}", f"context_precision_at_{k}")],
            *[f"ndcg_at_{k}" for k in ks],
            "duplicate_context_rate",
            "raw_duplicate_rate",
            "duplicate_document_rate",
            "unique_retrieved_document_count",
            "exact_match",
            "literal_exact_match",
            "canonical_exact_match",
            "german_canonical_exact_match",
            "umlaut_expanded_exact_match",
            "non_empty_answer_rate",
            "answer_coverage_rate",
            "abstention_rate",
            "is_unknown",
            "answer_relevancy_score",
            "rouge_l",
            "rouge_1",
            "embedding_similarity",
            "bow_token_overlap_similarity",
            "retrieval_time_ms",
            "rerank_time_ms",
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
        "category_routing_behavior": (
            "category_accuracy, category_precision_macro, category_recall_macro, and per_class_metrics "
            "are only emitted when Pipeline 1 produces at least one detected_category prediction. "
            "When category_routing_active=false all category metrics are suppressed to avoid null entries."
        ),
        "embedding_similarity_behavior": (
            "embedding_similarity contains values only when provider=sentence_transformers. "
            "bow_token_overlap_similarity contains values only when provider=deterministic_hash. "
            "The two columns are mutually exclusive; exactly one is non-null per row."
        ),
        "deprecated_metrics": [
            "company_match_rate",
            "year_match_rate",
            "month_match_rate",
            "exact_year_month_match_rate",
            "metadata_match_rate",
            "leaderboard",
            "summary_by_difficulty",
        ],
        "summary_behavior": (
            "mean retrieval and answer metrics use all evaluated rows; generation failures are retained, "
            "score zero for answer correctness, and can mark run_valid=false"
        ),
        "start_timestamp_utc": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_timestamp_utc": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
    }
