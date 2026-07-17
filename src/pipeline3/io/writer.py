from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.pipeline2.io.tabular import write_csv

logger = logging.getLogger("pipeline3.io")

PER_QUESTION_FIELDS = [
    "question_id",
    "experiment_id",
    "question",
    "generated_answer",
    "ground_truth",
    "context_truncated",
    "judge_success",
    "judge_correctness",
    "judge_faithfulness",
    "judge_completeness",
    "judge_hallucination",
    "judge_context_relevance",
    "judge_overall_score",
    "judge_llm_overall_score",
    "judge_reasoning",
    "judge_retry_count",
    "judge_latency_ms",
    "ragas_faithfulness",
    "ragas_faithfulness_status",
    "ragas_answer_relevancy",
    "ragas_context_recall",
]


def write_per_question_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, PER_QUESTION_FIELDS)


def write_semantic_summary_csv(
    path: Path, summary: dict[str, Any], run_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"run_id": run_id, **summary}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted(row.keys()), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerow(row)


def write_judge_raw_outputs(path: Path, raw_outputs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(raw_outputs, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_judge_failures(path: Path, failures: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_evaluation_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_pipeline3_report(
    path: Path, manifest: dict[str, Any], summary: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_build_report_markdown(manifest, summary), encoding="utf-8")


def _build_report_markdown(manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    run_id = manifest.get("run_id", "n/a")
    lines = [
        "# Pipeline 3 — Advanced Semantic Evaluation Report",
        "",
        f"**Run ID:** `{run_id}`",
        f"**Timestamp:** `{manifest.get('end_timestamp_utc', 'n/a')}`",
        f"**Judge model:** `{manifest.get('judge_model', 'n/a')}`",
        f"**Prompt version:** `{manifest.get('prompt_version', 'n/a')}`",
        f"**Total questions:** `{summary.get('n_questions', 'n/a')}`",
        f"**Judge success rate:** `{summary.get('judge_success_rate', 'n/a')}`",
        "",
        "## LLM-as-Judge Scores (mean)",
        "",
    ]
    judge_metrics = [
        "judge_correctness",
        "judge_faithfulness",
        "judge_completeness",
        "judge_hallucination",
        "judge_context_relevance",
        "judge_overall_score",
    ]
    for metric in judge_metrics:
        val = summary.get(f"mean_{metric}")
        label = metric.replace("judge_", "").replace("_", " ").title()
        lines.append(
            f"- {label}: `{val:.4f}`"
            if isinstance(val, float)
            else f"- {label}: `n/a`"
        )

    lines += ["", "## RAGAS Scores (mean)", ""]
    ragas_metrics = [
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        "ragas_context_recall",
    ]
    for metric in ragas_metrics:
        val = summary.get(f"mean_{metric}")
        label = metric.replace("ragas_", "").replace("_", " ").title()
        lines.append(
            f"- {label}: `{val:.4f}`"
            if isinstance(val, float)
            else f"- {label}: `n/a`"
        )

    ragas_nan = manifest.get("ragas_stats", {}).get("nan_counts", {})
    if ragas_nan and any(v > 0 for v in ragas_nan.values()):
        lines += ["", "## RAGAS Data Quality Warnings", ""]
        for metric, count in sorted(ragas_nan.items()):
            if count > 0:
                valid_rows = summary.get(f"{metric}_valid_rows", "n/a")
                total_rows = summary.get(f"{metric}_total_rows", "n/a")
                coverage_pct = summary.get(f"{metric}_coverage_percentage")
                coverage_text = (
                    f"{coverage_pct:.2f}%"
                    if isinstance(coverage_pct, float)
                    else "n/a"
                )
                lines.append(
                    f"- `{metric}`: **{count}** row(s) produced NaN — excluded from mean "
                    f"(valid={valid_rows}/{total_rows}, coverage={coverage_text})"
                )

    lines += ["", "## Reproducibility", ""]
    repro = manifest.get("reproducibility", {})
    lines.append(f"- Judge model: `{repro.get('judge_model', 'n/a')}`")
    lines.append(f"- Judge model digest: `{repro.get('judge_model_digest', 'n/a')}`")
    lines.append(f"- Prompt version: `{repro.get('prompt_version', 'n/a')}`")
    lines.append(f"- Config fingerprint: `{repro.get('config_fingerprint', 'n/a')}`")
    lines.append(f"- Pipeline version: `{repro.get('pipeline_version', 'n/a')}`")

    return "\n".join(lines) + "\n"
