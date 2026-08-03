from __future__ import annotations

import csv
import json
import math
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
    mean_recall_at_5: Optional[float]
    mean_mrr_at_5: Optional[float]
    mean_ndcg_at_5: Optional[float]
    mean_context_precision_at_5: Optional[float]
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
    primary_k: int = 5
    available_ks: list[int] = field(default_factory=lambda: [5])
    primary_recall: float = 0.0
    primary_mrr: float = 0.0
    primary_ndcg: float = 0.0
    primary_context_precision: Optional[float] = None
    primary_chunk_hit: Optional[float] = None
    primary_chunk_recall: Optional[float] = None
    primary_chunk_mrr: Optional[float] = None
    primary_chunk_ndcg: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.available_ks:
            self.available_ks = [5] if self.mean_recall_at_5 is not None else []
        if self.primary_k == 5:
            if self.mean_recall_at_5 is not None:
                self.primary_recall = self.mean_recall_at_5
            if self.mean_mrr_at_5 is not None:
                self.primary_mrr = self.mean_mrr_at_5
            if self.mean_ndcg_at_5 is not None:
                self.primary_ndcg = self.mean_ndcg_at_5
            if self.mean_context_precision_at_5 is not None:
                self.primary_context_precision = self.mean_context_precision_at_5


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
    if len(experiments) != 1:
        raise ValueError(
            f"Expected exactly one experiment in {summary_path}, found {len(experiments)}."
        )

    exp = experiments[0]
    experiment_id = exp["experiment_id"]
    available_ks = _available_metric_cutoffs(exp)
    primary_k = _primary_k_for_experiment(experiment_id, available_ks)
    primary_metrics = _primary_metrics(exp, primary_k, summary_path)

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
        generation_failure_rate=_bounded_float(
            exp.get("generation_failure_rate", 0.0),
            "generation_failure_rate",
            summary_path,
            minimum=0.0,
            maximum=1.0,
        ),
        primary_k=primary_k,
        available_ks=available_ks,
        primary_recall=primary_metrics["recall"],
        primary_mrr=primary_metrics["mrr"],
        primary_ndcg=primary_metrics["ndcg"],
        primary_context_precision=primary_metrics["context_precision"],
        primary_chunk_hit=_optional_float(exp.get(f"mean_chunk_hit_at_{primary_k}")),
        primary_chunk_recall=_optional_float(exp.get(f"mean_chunk_recall_at_{primary_k}")),
        primary_chunk_mrr=_optional_float(exp.get(f"mean_chunk_mrr_at_{primary_k}")),
        primary_chunk_ndcg=_optional_float(exp.get(f"mean_chunk_ndcg_at_{primary_k}")),
        mean_recall_at_5=_required_bounded_float(exp.get("mean_recall_at_5"), "mean_recall_at_5", summary_path),
        mean_mrr_at_5=_required_bounded_float(exp.get("mean_mrr_at_5"), "mean_mrr_at_5", summary_path),
        mean_ndcg_at_5=_required_bounded_float(exp.get("mean_ndcg_at_5"), "mean_ndcg_at_5", summary_path),
        mean_context_precision_at_5=_optional_bounded_float(exp.get("mean_context_precision_at_5"), "mean_context_precision_at_5", summary_path),
        unknown_rate=_bounded_float(exp.get("unknown_rate", 0.0), "unknown_rate", summary_path, minimum=0.0, maximum=1.0),
        mean_embedding_similarity=_required_bounded_float(
            exp.get("mean_embedding_similarity"), "mean_embedding_similarity", summary_path
        ),
        mean_official_bertscore_f1=_required_bounded_float(
            exp.get("mean_official_bertscore_f1"), "mean_official_bertscore_f1", summary_path
        ),
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
            ragas_faithfulness_nan_rate = _bounded_float(
                nan_counts["ragas_faithfulness"] / n_questions,
                "ragas_faithfulness_nan_rate",
                manifest_path,
                minimum=0.0,
                maximum=1.0,
            )
        if "ragas_answer_relevancy" in nan_counts:
            ragas_answer_relevancy_nan_rate = _bounded_float(
                nan_counts["ragas_answer_relevancy"] / n_questions,
                "ragas_answer_relevancy_nan_rate",
                manifest_path,
                minimum=0.0,
                maximum=1.0,
            )
        if "ragas_context_recall" in nan_counts:
            ragas_context_recall_nan_rate = _bounded_float(
                nan_counts["ragas_context_recall"] / n_questions,
                "ragas_context_recall_nan_rate",
                manifest_path,
                minimum=0.0,
                maximum=1.0,
            )

    for required in (
        "mean_judge_correctness",
        "mean_judge_faithfulness",
        "mean_judge_completeness",
        "mean_judge_context_relevance",
        "mean_ragas_faithfulness",
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
        judge_success_rate=_bounded_float(
            summary.get("judge_success_rate", 0.0),
            "judge_success_rate",
            manifest_path,
            minimum=0.0,
            maximum=1.0,
        ),
        judge_failure_count=int(summary.get("judge_failure_count", 0)),
        mean_judge_correctness=_bounded_float(summary["mean_judge_correctness"], "mean_judge_correctness", manifest_path, minimum=0.0, maximum=5.0),
        mean_judge_faithfulness=_bounded_float(summary["mean_judge_faithfulness"], "mean_judge_faithfulness", manifest_path, minimum=0.0, maximum=5.0),
        mean_judge_completeness=_bounded_float(summary["mean_judge_completeness"], "mean_judge_completeness", manifest_path, minimum=0.0, maximum=5.0),
        mean_judge_hallucination=_bounded_float(summary.get("mean_judge_hallucination", 0.0), "mean_judge_hallucination", manifest_path, minimum=0.0, maximum=5.0),
        mean_judge_context_relevance=_bounded_float(summary["mean_judge_context_relevance"], "mean_judge_context_relevance", manifest_path, minimum=0.0, maximum=5.0),
        mean_judge_overall_score=_bounded_float(summary.get("mean_judge_overall_score", 0.0), "mean_judge_overall_score", manifest_path, minimum=0.0, maximum=5.0),
        mean_ragas_faithfulness=_required_bounded_float(summary.get("mean_ragas_faithfulness"), "mean_ragas_faithfulness", manifest_path),
        mean_ragas_answer_relevancy=_optional_bounded_float(summary.get("mean_ragas_answer_relevancy"), "mean_ragas_answer_relevancy", manifest_path),
        mean_ragas_context_recall=_optional_bounded_float(summary.get("mean_ragas_context_recall"), "mean_ragas_context_recall", manifest_path),
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


def _available_metric_cutoffs(exp: dict) -> list[int]:
    prefix = "mean_recall_at_"
    cutoffs: list[int] = []
    for key, value in exp.items():
        if key.startswith(prefix) and value is not None:
            suffix = key[len(prefix):]
            if suffix.isdigit():
                cutoffs.append(int(suffix))
    return sorted(set(cutoffs))


def _primary_k_for_experiment(experiment_id: str, available_ks: list[int]) -> int:
    explicit = {
        "A04-K03": 3,
        "A04-K05": 5,
        "A04-K10": 10,
    }
    if experiment_id in explicit:
        return explicit[experiment_id]
    if 5 in available_ks:
        return 5
    if available_ks:
        return max(available_ks)
    raise ValueError(f"No retrieval metric cutoffs found for experiment {experiment_id!r}.")


def _primary_metrics(exp: dict, primary_k: int, summary_path: Path) -> dict[str, Optional[float]]:
    required = {
        "recall": f"mean_recall_at_{primary_k}",
        "mrr": f"mean_mrr_at_{primary_k}",
        "ndcg": f"mean_ndcg_at_{primary_k}",
    }
    metrics: dict[str, Optional[float]] = {}
    for name, key in required.items():
        if key not in exp or exp[key] is None:
            raise ValueError(
                f"Required primary P2 metric '{key}' is missing or null in {summary_path}"
            )
        metrics[name] = _bounded_float(exp[key], key, summary_path, minimum=0.0, maximum=1.0)
    metrics["context_precision"] = _optional_bounded_float(
        exp.get(f"mean_context_precision_at_{primary_k}"),
        f"mean_context_precision_at_{primary_k}",
        summary_path,
    )
    return metrics


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _required_bounded_float(value: object, metric_name: str, source_path: Path) -> float:
    if value is None:
        raise ValueError(f"Required metric '{metric_name}' is missing or null in {source_path}")
    return _bounded_float(value, metric_name, source_path, minimum=0.0, maximum=1.0)


def _optional_bounded_float(value: object, metric_name: str, source_path: Path) -> Optional[float]:
    if value is None:
        return None
    return _bounded_float(value, metric_name, source_path, minimum=0.0, maximum=1.0)


def _bounded_float(
    value: object,
    metric_name: str,
    source_path: Path,
    *,
    minimum: float,
    maximum: float,
) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Metric '{metric_name}' is not finite in {source_path}: {value!r}")
    if numeric < minimum or numeric > maximum:
        raise ValueError(
            f"Metric '{metric_name}' outside [{minimum}, {maximum}] in {source_path}: {numeric}"
        )
    return numeric


def _resolve_question_id(row: dict) -> str:
    for key in ("question_id", "uid", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
