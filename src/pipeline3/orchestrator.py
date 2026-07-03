from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline1.utils.hashing import file_sha256, stable_hash_dict
from src.pipeline3.aggregation.summarizer import summarize_semantic_metrics
from src.pipeline3.io.writer import (
    write_evaluation_manifest,
    write_judge_failures,
    write_judge_raw_outputs,
    write_per_question_csv,
    write_pipeline3_report,
    write_semantic_summary_csv,
)
from src.pipeline3.judge.ollama_client import OllamaClient
from src.pipeline3.metrics.ragas_metrics import RagasEvaluator
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config
from src.pipeline3.stages.judge_stage import JudgeRowResult, JudgeStageResult, run_judge_stage
from src.pipeline3.stages.loader_stage import LoaderResult, load_inputs
from src.pipeline3.stages.ragas_stage import run_ragas_stage
from src.pipeline3.stages.validation_stage import (
    ValidationReport,
    build_qa_index,
    validate_inputs,
    _resolve_id,
    _resolve_qa_answer,
)

logger = logging.getLogger("pipeline3")


class Pipeline3Orchestrator:
    def run(self, config_path: str) -> Path:
        start_time = time.time()
        cfg = Pipeline3Config.from_yaml(config_path)
        project_root = Path(__file__).resolve().parents[2]
        run_cfg = cfg.pipeline3
        run_dir = project_root / run_cfg.output_dir / run_cfg.run_id

        print(f"[Pipeline 3] Output directory: {run_dir}")
        if run_dir.exists() and not run_cfg.overwrite:
            raise FileExistsError(
                f"Pipeline 3 run already exists and overwrite=false: {run_dir}"
            )
        if run_dir.exists() and run_cfg.overwrite:
            for p in run_dir.iterdir():
                if p.is_file():
                    p.unlink()
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── Stage 1 ──────────────────────────────────────────────────────────
        print("[1/4] Loading evaluation inputs")
        loader_result: LoaderResult = load_inputs(cfg, project_root)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        print("[2/4] Validating data integrity")
        validation_report: ValidationReport = validate_inputs(
            loader_result.rag_rows,
            loader_result.qa_rows,
            loader_result.questions_rows,
        )
        _print_validation_summary(validation_report)
        qa_by_id = build_qa_index(loader_result.qa_rows)

        # ── Stage 3 ──────────────────────────────────────────────────────────
        print("[3/4] Running RAGAS evaluation")
        ragas_evaluator = RagasEvaluator(cfg.ragas)
        ragas_results = run_ragas_stage(
            loader_result.rag_rows, qa_by_id, ragas_evaluator
        )
        if ragas_results.skipped:
            reason = ragas_results.error or "disabled by config"
            print(f"       RAGAS skipped: {reason}")
            if cfg.ragas.enabled and cfg.ragas.fail_on_ragas_error:
                raise RuntimeError(
                    "RAGAS evaluation failed or was skipped while "
                    f"ragas.fail_on_ragas_error=true: {reason}"
                )
        else:
            print(f"       RAGAS completed with metrics: {ragas_results.enabled_metrics}")

        ragas_by_id: dict[str, dict[str, Any]] = {}
        for row in ragas_results.rows:
            qid = row.get("question_id", "")
            if qid:
                ragas_by_id[qid] = {k: v for k, v in row.items() if k != "question_id"}

        # ── Stage 4 ──────────────────────────────────────────────────────────
        print("[4/4] Running LLM-as-Judge evaluation")
        judge_stage_result: JudgeStageResult = JudgeStageResult()
        if cfg.llm_judge.enabled:
            judge_stage_result = run_judge_stage(
                loader_result.rag_rows,
                qa_by_id,
                cfg.judge,
                cfg.llm_judge,
            )
            print(
                f"       Judge: {judge_stage_result.successes}/{judge_stage_result.total} "
                f"succeeded ({judge_stage_result.failures} failures)"
            )
        else:
            print("       LLM-as-Judge disabled by config")

        # ── Merge results ─────────────────────────────────────────────────────
        judge_by_id: dict[str, JudgeRowResult] = {
            r.question_id: r for r in judge_stage_result.rows
        }
        per_question = _build_per_question(
            loader_result.rag_rows, qa_by_id, judge_by_id, ragas_by_id
        )

        # ── Aggregate ─────────────────────────────────────────────────────────
        summary = summarize_semantic_metrics(per_question)

        # ── Collect raw outputs ───────────────────────────────────────────────
        raw_outputs = [
            {
                "question_id": r.question_id,
                "success": r.success,
                "retry_count": r.retry_count,
                "latency_ms": r.latency_ms,
                "raw_response": r.raw_response,
                "error": r.error,
            }
            for r in judge_stage_result.rows
        ]
        failures = [
            {
                "question_id": r.question_id,
                "error": r.error,
                "retry_count": r.retry_count,
                "raw_response": r.raw_response,
            }
            for r in judge_stage_result.rows
            if not r.success
        ]

        # ── Reproducibility metadata ───────────────────────────────────────────
        judge_client = OllamaClient(
            base_url=cfg.judge.base_url,
            model=cfg.judge.model,
            temperature=cfg.judge.temperature,
            timeout_seconds=cfg.judge.timeout_seconds,
        )
        model_info = judge_client.get_model_info()
        judge_model_digest = (
            model_info.get("digest")
            or model_info.get("details", {}).get("digest")
            or "unknown"
        )

        config_fingerprint = stable_hash_dict({
            "judge": cfg.judge.model_dump(),
            "ragas": cfg.ragas.model_dump(),
            "llm_judge": cfg.llm_judge.model_dump(),
        })

        end_time = time.time()
        manifest = _build_manifest(
            config_path=config_path,
            cfg=cfg,
            loader_result=loader_result,
            validation_report=validation_report,
            judge_stage_result=judge_stage_result,
            ragas_results_enabled=ragas_results.enabled_metrics,
            ragas_skipped=ragas_results.skipped,
            ragas_error=ragas_results.error,
            ragas_nan_counts=ragas_results.nan_counts,
            ragas_valid_counts=ragas_results.valid_counts,
            summary=summary,
            judge_model_digest=judge_model_digest,
            config_fingerprint=config_fingerprint,
            start_time=start_time,
            end_time=end_time,
        )

        # ── Write outputs ──────────────────────────────────────────────────────
        print("[Pipeline 3] Writing outputs")
        if run_cfg.save_csv:
            write_per_question_csv(
                run_dir / "per_question_semantic_metrics.csv", per_question
            )
            write_semantic_summary_csv(
                run_dir / "semantic_summary.csv", summary, run_cfg.run_id
            )
        write_judge_raw_outputs(run_dir / "judge_raw_outputs.json", raw_outputs)
        write_judge_failures(run_dir / "judge_failures.json", failures)
        write_evaluation_manifest(run_dir / "evaluation_manifest.json", manifest)
        write_pipeline3_report(run_dir / "pipeline3_report.md", manifest, summary)

        print(f"[Pipeline 3] Done. Output: {run_dir}")
        return run_dir


def _build_per_question(
    rag_rows: list[dict[str, Any]],
    qa_by_id: dict[str, dict[str, Any]],
    judge_by_id: dict[str, JudgeRowResult],
    ragas_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    per_question = []
    for row in rag_rows:
        qid = _resolve_id(row)
        qa_row = qa_by_id.get(qid, {})
        ground_truth = _resolve_qa_answer(qa_row)

        output: dict[str, Any] = {
            "question_id": qid,
            "experiment_id": str(row.get("experiment_id", "")),
            "question": str(row.get("question", "")),
            "generated_answer": str(row.get("generated_answer", "")),
            "ground_truth": ground_truth,
            "context_truncated": False,
            "judge_success": False,
            "judge_correctness": None,
            "judge_faithfulness": None,
            "judge_relevancy": None,
            "judge_completeness": None,
            "judge_hallucination": None,
            "judge_context_relevance": None,
            "judge_overall_score": None,
            "judge_llm_overall_score": None,
            "judge_reasoning": None,
            "judge_retry_count": None,
            "judge_latency_ms": None,
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_context_precision": None,
            "ragas_context_recall": None,
        }

        judge_result = judge_by_id.get(qid)
        if judge_result is not None:
            output["context_truncated"] = judge_result.context_truncated
        if judge_result and judge_result.success and judge_result.response:
            resp = judge_result.response
            output["judge_success"] = True
            output["judge_correctness"] = resp.correctness
            output["judge_faithfulness"] = resp.faithfulness
            output["judge_relevancy"] = resp.relevancy
            output["judge_completeness"] = resp.completeness
            output["judge_hallucination"] = resp.hallucination
            output["judge_context_relevance"] = resp.context_relevance
            output["judge_overall_score"] = resp.overall_score
            output["judge_llm_overall_score"] = judge_result.llm_overall_score
            output["judge_reasoning"] = resp.reasoning
            output["judge_retry_count"] = judge_result.retry_count
            output["judge_latency_ms"] = judge_result.latency_ms

        ragas_row = ragas_by_id.get(qid, {})
        output.update(ragas_row)

        per_question.append(output)
    return per_question


def _print_validation_summary(report: ValidationReport) -> None:
    stats = report.stats
    print(
        f"       Validation passed: {report.passed} | "
        f"rag_rows={stats.get('rag_rows', '?')} | "
        f"qa_rows={stats.get('qa_rows', '?')} | "
        f"warnings={len(report.warnings)}"
    )


def _build_manifest(
    config_path: str,
    cfg: Pipeline3Config,
    loader_result: LoaderResult,
    validation_report: ValidationReport,
    judge_stage_result: JudgeStageResult,
    ragas_results_enabled: list[str],
    ragas_skipped: bool,
    ragas_error: str | None,
    ragas_nan_counts: dict[str, int],
    ragas_valid_counts: dict[str, int],
    summary: dict[str, Any],
    judge_model_digest: str,
    config_fingerprint: str,
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    run_cfg = cfg.pipeline3
    return {
        "run_id": run_cfg.run_id,
        "pipeline_version": run_cfg.version,
        "start_timestamp_utc": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_timestamp_utc": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
        "duration_seconds": round(end_time - start_time, 2),
        "config_path": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "judge_model": cfg.judge.model,
        "prompt_version": run_cfg.prompt_version,
        "reproducibility": {
            "judge_model": cfg.judge.model,
            "judge_model_digest": judge_model_digest,
            "judge_temperature": cfg.judge.temperature,
            "prompt_version": run_cfg.prompt_version,
            "config_fingerprint": config_fingerprint,
            "pipeline_version": run_cfg.version,
            "enabled_judge_metrics": [
                k for k, v in cfg.llm_judge.metrics.model_dump().items() if v
            ],
            "enabled_ragas_metrics": ragas_results_enabled,
            "ragas_embeddings_model": cfg.ragas.embeddings_model,
            "ragas_embeddings_device": cfg.ragas.embeddings_device,
            "ragas_require_cuda": cfg.ragas.require_cuda,
            "metric_weights": cfg.llm_judge.weights.model_dump(),
            "evaluation_timestamp_utc": datetime.fromtimestamp(
                end_time, timezone.utc
            ).isoformat(),
        },
        "inputs": {
            "pipeline1_results_path": str(loader_result.rag_path),
            "pipeline1_results_sha256": (
                file_sha256(loader_result.rag_path)
                if loader_result.rag_path.exists()
                else None
            ),
            "questions_path": str(loader_result.questions_path),
            "questions_sha256": (
                file_sha256(loader_result.questions_path)
                if loader_result.questions_path.exists()
                else None
            ),
            "qa_path": str(loader_result.qa_path),
            "qa_sha256": (
                file_sha256(loader_result.qa_path)
                if loader_result.qa_path.exists()
                else None
            ),
            "gold_contexts_path": str(loader_result.gold_contexts_path),
            "rag_rows": len(loader_result.rag_rows),
            "qa_rows": len(loader_result.qa_rows),
            "questions_rows": len(loader_result.questions_rows),
        },
        "validation": {
            "passed": validation_report.passed,
            "errors": validation_report.errors,
            "warnings": validation_report.warnings,
            "stats": validation_report.stats,
        },
        "judge_stats": {
            "enabled": cfg.llm_judge.enabled,
            "total": judge_stage_result.total,
            "successes": judge_stage_result.successes,
            "failures": judge_stage_result.failures,
            "failure_rate": judge_stage_result.failure_rate,
        },
        "ragas_stats": {
            "enabled": cfg.ragas.enabled,
            "fail_on_ragas_error": cfg.ragas.fail_on_ragas_error,
            "embeddings_model": cfg.ragas.embeddings_model,
            "embeddings_device": cfg.ragas.embeddings_device,
            "require_cuda": cfg.ragas.require_cuda,
            "skipped": ragas_skipped,
            "error": ragas_error,
            "enabled_metrics": ragas_results_enabled,
            "nan_counts": ragas_nan_counts,
            "valid_counts": ragas_valid_counts,
        },
        "summary": summary,
    }
