from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class P2Summary:
    experiment_id: str
    n_questions: int
    run_valid: bool
    generation_failure_rate: float
    mean_recall_at_5: float
    mean_mrr_at_5: float
    mean_ndcg_at_5: float
    mean_context_precision_at_5: float
    unknown_rate: float
    mean_embedding_similarity: Optional[float]
    mean_official_bertscore_f1: Optional[float]
    qa_hash: Optional[str]
    gold_contexts_hash: Optional[str]
    p2_run_dir: str
    raw: dict = field(default_factory=dict)
    audit_manifest_present: bool = False
    final_verdict: Optional[str] = None
    strict_audit_pass: Optional[bool] = None
    fake_run_suspicious: Optional[bool] = None
    fake_run_suspicious_checks: list[str] = field(default_factory=list)
    row_counts: dict = field(default_factory=dict)
    duplicate_question_ids: list[str] = field(default_factory=list)
    question_ids: list[str] = field(default_factory=list)
    expected_question_count: Optional[int] = None
    required_outputs_present: bool = False
    missing_required_outputs: list[str] = field(default_factory=list)


@dataclass
class P3Summary:
    run_id: str
    experiment_id: str
    n_questions: int
    judge_model: str
    prompt_version: str
    qa_sha256: Optional[str]
    judge_success_rate: float
    judge_failure_count: int
    mean_judge_correctness: float
    mean_judge_faithfulness: float
    mean_judge_completeness: float
    mean_judge_hallucination: float
    mean_judge_context_relevance: float
    mean_judge_overall_score: float
    mean_ragas_faithfulness: Optional[float]
    mean_ragas_answer_relevancy: Optional[float]
    ragas_faithfulness_nan_rate: Optional[float]
    ragas_answer_relevancy_nan_rate: Optional[float]
    p3_run_dir: str
    raw: dict = field(default_factory=dict)
    # Context Recall is a new optional metric; defaults to None for backward compatibility
    # with P3 runs that pre-date its addition.
    mean_ragas_context_recall: Optional[float] = None
    ragas_context_recall_nan_rate: Optional[float] = None
    validation_passed: Optional[bool] = None
    inputs: dict = field(default_factory=dict)
    row_output_present: bool = False
    summary_present: bool = False
    question_ids: list[str] = field(default_factory=list)
    duplicate_question_ids: list[str] = field(default_factory=list)
    expected_question_count: Optional[int] = None


def load_p2_summary(run_dir: Path) -> P2Summary:
    summary_path = run_dir / "summary_metrics.json"
    manifest_path = run_dir / "eval_manifest.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"P2 summary_metrics.json not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    experiments = summary_data.get("summary_by_experiment", [])
    if not experiments:
        raise ValueError(f"summary_by_experiment is empty in {summary_path}")

    exp = experiments[0]
    experiment_id = exp["experiment_id"]

    for required in (
        "mean_recall_at_5",
        "mean_mrr_at_5",
        "mean_ndcg_at_5",
        "mean_context_precision_at_5",
    ):
        if required not in exp or exp[required] is None:
            raise ValueError(
                f"Required P2 metric '{required}' is missing or null in {summary_path}"
            )

    qa_hash: Optional[str] = None
    gold_contexts_hash: Optional[str] = None
    manifest_data: dict = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        qa_hash = manifest_data.get("qa_hash")
        gold_contexts_hash = manifest_data.get("gold_contexts_hash")
    row_counts = manifest_data.get("row_counts", {}) if manifest_data else {}
    fake_run = manifest_data.get("fake_run_detection", {}) if manifest_data else {}
    fake_checks = [
        str(check.get("name"))
        for check in fake_run.get("checks", [])
        if isinstance(check, dict) and check.get("suspicious")
    ]
    per_question_path = run_dir / "per_question.jsonl"
    question_ids = _read_jsonl_question_ids(per_question_path)
    required_outputs = [
        summary_path,
        manifest_path,
        per_question_path,
        run_dir / "per_question_metrics.jsonl",
    ]
    missing_required_outputs = [path.name for path in required_outputs if not path.exists()]

    return P2Summary(
        experiment_id=experiment_id,
        n_questions=int(exp["n_questions"]),
        run_valid=bool(exp.get("run_valid", False)),
        generation_failure_rate=float(exp.get("generation_failure_rate", 0.0)),
        mean_recall_at_5=float(exp["mean_recall_at_5"]),
        mean_mrr_at_5=float(exp["mean_mrr_at_5"]),
        mean_ndcg_at_5=float(exp["mean_ndcg_at_5"]),
        mean_context_precision_at_5=float(exp["mean_context_precision_at_5"]),
        unknown_rate=float(exp.get("unknown_rate", 0.0)),
        mean_embedding_similarity=exp.get("mean_embedding_similarity"),
        mean_official_bertscore_f1=exp.get("mean_official_bertscore_f1"),
        qa_hash=qa_hash,
        gold_contexts_hash=gold_contexts_hash,
        p2_run_dir=str(run_dir),
        raw=exp,
        audit_manifest_present=manifest_path.exists(),
        final_verdict=manifest_data.get("final_verdict") if manifest_data else None,
        strict_audit_pass=manifest_data.get("strict_audit_pass") if manifest_data else None,
        fake_run_suspicious=fake_run.get("suspicious") if fake_run else None,
        fake_run_suspicious_checks=fake_checks,
        row_counts=row_counts,
        duplicate_question_ids=_duplicate_ids(question_ids),
        question_ids=question_ids,
        expected_question_count=int(row_counts.get("questions_rows", exp["n_questions"])),
        required_outputs_present=not missing_required_outputs,
        missing_required_outputs=missing_required_outputs,
    )


def load_p3_summary(run_dir: Path) -> P3Summary:
    manifest_path = run_dir / "evaluation_manifest.json"
    per_question_path = run_dir / "per_question_semantic_metrics.csv"
    summary_path = run_dir / "semantic_summary.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"P3 evaluation_manifest.json not found: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_id = data["run_id"]
    summary = data["summary"]
    inputs = data.get("inputs", {})
    validation = data.get("validation", {})
    ragas_stats = data.get("ragas_stats", {})
    reproducibility = data.get("reproducibility", {})

    experiment_id = run_id[3:] if run_id.startswith("p3_") else run_id

    judge_model = data.get("judge_model") or reproducibility.get("judge_model", "unknown")
    prompt_version = data.get("prompt_version") or reproducibility.get(
        "prompt_version", "unknown"
    )

    n_questions = int(summary.get("n_questions", inputs.get("rag_rows", 0)))

    nan_counts = ragas_stats.get("nan_counts", {})
    ragas_faithfulness_nan_rate: Optional[float] = None
    ragas_answer_relevancy_nan_rate: Optional[float] = None
    ragas_context_recall_nan_rate: Optional[float] = None
    if n_questions > 0:
        if "ragas_faithfulness" in nan_counts:
            ragas_faithfulness_nan_rate = nan_counts["ragas_faithfulness"] / n_questions
        if "ragas_answer_relevancy" in nan_counts:
            ragas_answer_relevancy_nan_rate = (
                nan_counts["ragas_answer_relevancy"] / n_questions
            )
        if "ragas_context_recall" in nan_counts:
            ragas_context_recall_nan_rate = (
                nan_counts["ragas_context_recall"] / n_questions
            )

    for required in (
        "mean_judge_correctness",
        "mean_judge_faithfulness",
        "mean_judge_context_relevance",
    ):
        if required not in summary or summary[required] is None:
            raise ValueError(
                f"Required P3 metric '{required}' is missing or null in {manifest_path}"
            )

    question_ids = _read_csv_question_ids(per_question_path)
    return P3Summary(
        run_id=run_id,
        experiment_id=experiment_id,
        n_questions=n_questions,
        judge_model=judge_model,
        prompt_version=prompt_version,
        qa_sha256=inputs.get("qa_sha256"),
        judge_success_rate=float(summary.get("judge_success_rate", 0.0)),
        judge_failure_count=int(summary.get("judge_failure_count", 0)),
        mean_judge_correctness=float(summary["mean_judge_correctness"]),
        mean_judge_faithfulness=float(summary["mean_judge_faithfulness"]),
        mean_judge_completeness=float(summary.get("mean_judge_completeness", 0.0)),
        mean_judge_hallucination=float(summary.get("mean_judge_hallucination", 0.0)),
        mean_judge_context_relevance=float(summary["mean_judge_context_relevance"]),
        mean_judge_overall_score=float(summary.get("mean_judge_overall_score", 0.0)),
        mean_ragas_faithfulness=summary.get("mean_ragas_faithfulness"),
        mean_ragas_answer_relevancy=summary.get("mean_ragas_answer_relevancy"),
        mean_ragas_context_recall=summary.get("mean_ragas_context_recall"),
        ragas_faithfulness_nan_rate=ragas_faithfulness_nan_rate,
        ragas_answer_relevancy_nan_rate=ragas_answer_relevancy_nan_rate,
        ragas_context_recall_nan_rate=ragas_context_recall_nan_rate,
        p3_run_dir=str(run_dir),
        raw=data,
        validation_passed=validation.get("passed"),
        inputs=inputs,
        row_output_present=per_question_path.exists(),
        summary_present=summary_path.exists(),
        question_ids=question_ids,
        duplicate_question_ids=_duplicate_ids(question_ids),
        expected_question_count=int(inputs.get("questions_rows", n_questions)),
    )


def _read_jsonl_question_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ids.append(_resolve_question_id(json.loads(line)))
    return [qid for qid in ids if qid]


def _read_csv_question_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            qid
            for qid in (_resolve_question_id(row) for row in csv.DictReader(f))
            if qid
        ]


def _duplicate_ids(ids: list[str]) -> list[str]:
    return sorted(qid for qid, count in Counter(ids).items() if count > 1)


def _resolve_question_id(row: dict) -> str:
    for key in ("question_id", "uid", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
