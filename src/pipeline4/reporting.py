from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import Pipeline4Config
from src.pipeline4.scoring import (
    compute_retrieval_score,
    compute_rqi,
    retrieval_score_contributions,
    rqi_contributions,
)
from src.pipeline4.validation import ComparisonGroup, ExperimentValidation


@dataclass
class ExperimentRecord:
    experiment_id: str
    p2_run_dir: str
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
    has_p3: bool
    p3_run_dir: Optional[str]
    judge_model: Optional[str]
    prompt_version: Optional[str]
    judge_success_rate: Optional[float]
    judge_failure_count: Optional[int]
    mean_judge_correctness: Optional[float]
    mean_judge_faithfulness: Optional[float]
    mean_judge_completeness: Optional[float]
    mean_judge_hallucination: Optional[float]
    mean_judge_context_relevance: Optional[float]
    mean_judge_overall_score: Optional[float]
    mean_ragas_faithfulness: Optional[float]
    mean_ragas_answer_relevancy: Optional[float]
    ragas_faithfulness_nan_rate: Optional[float]
    ragas_answer_relevancy_nan_rate: Optional[float]
    qa_sha256: Optional[str]
    retrieval_score: float
    rqi: Optional[float]
    p2_status: str
    p3_status: Optional[str]
    p3_issues: list[str] = field(default_factory=list)
    ragas_warnings: list[str] = field(default_factory=list)
    retrieval_rank: Optional[int] = None
    rqi_rank: Optional[int] = None
    comparison_group_id: str = ""


def build_records(
    p2_summaries: list[P2Summary],
    p3_map: dict[str, Optional[P3Summary]],
    validations: dict[str, ExperimentValidation],
    comparison_groups: list[ComparisonGroup],
    cfg: Pipeline4Config,
) -> list[ExperimentRecord]:
    group_by_exp: dict[str, str] = {}
    for group in comparison_groups:
        for eid in group.experiment_ids:
            group_by_exp[eid] = group.group_id

    records: list[ExperimentRecord] = []
    for p2 in p2_summaries:
        val = validations[p2.experiment_id]
        p3 = p3_map.get(p2.experiment_id)

        retrieval_score = compute_retrieval_score(p2, cfg.retrieval_score_weights)
        rqi: Optional[float] = None
        if p3 is not None and not val.p3_invalid:
            rqi = compute_rqi(p2, p3, cfg.rqi_weights)

        rec = ExperimentRecord(
            experiment_id=p2.experiment_id,
            p2_run_dir=p2.p2_run_dir,
            n_questions=p2.n_questions,
            run_valid=p2.run_valid,
            generation_failure_rate=p2.generation_failure_rate,
            mean_recall_at_5=p2.mean_recall_at_5,
            mean_mrr_at_5=p2.mean_mrr_at_5,
            mean_ndcg_at_5=p2.mean_ndcg_at_5,
            mean_context_precision_at_5=p2.mean_context_precision_at_5,
            unknown_rate=p2.unknown_rate,
            mean_embedding_similarity=p2.mean_embedding_similarity,
            mean_official_bertscore_f1=p2.mean_official_bertscore_f1,
            qa_hash=p2.qa_hash,
            has_p3=p3 is not None,
            p3_run_dir=p3.p3_run_dir if p3 else None,
            judge_model=p3.judge_model if p3 else None,
            prompt_version=p3.prompt_version if p3 else None,
            judge_success_rate=p3.judge_success_rate if p3 else None,
            judge_failure_count=p3.judge_failure_count if p3 else None,
            mean_judge_correctness=p3.mean_judge_correctness if p3 else None,
            mean_judge_faithfulness=p3.mean_judge_faithfulness if p3 else None,
            mean_judge_completeness=p3.mean_judge_completeness if p3 else None,
            mean_judge_hallucination=p3.mean_judge_hallucination if p3 else None,
            mean_judge_context_relevance=p3.mean_judge_context_relevance if p3 else None,
            mean_judge_overall_score=p3.mean_judge_overall_score if p3 else None,
            mean_ragas_faithfulness=p3.mean_ragas_faithfulness if p3 else None,
            mean_ragas_answer_relevancy=p3.mean_ragas_answer_relevancy if p3 else None,
            ragas_faithfulness_nan_rate=p3.ragas_faithfulness_nan_rate if p3 else None,
            ragas_answer_relevancy_nan_rate=p3.ragas_answer_relevancy_nan_rate if p3 else None,
            qa_sha256=p3.qa_sha256 if p3 else None,
            retrieval_score=retrieval_score,
            rqi=rqi,
            p2_status=val.p2_status,
            p3_status=val.p3_status,
            p3_issues=val.p3_issues,
            ragas_warnings=val.ragas_warnings,
            comparison_group_id=group_by_exp.get(p2.experiment_id, "EXCLUDED"),
        )
        records.append(rec)
    return records


def write_retrieval_leaderboard(records: list[ExperimentRecord], out_path: Path) -> None:
    ranked = [r for r in records if r.retrieval_rank is not None]
    ranked.sort(key=lambda r: (r.retrieval_rank, r.experiment_id))

    fieldnames = [
        "rank",
        "experiment_id",
        "retrieval_score",
        "recall_at_5",
        "mrr_at_5",
        "ndcg_at_5",
        "context_precision_at_5",
        "unknown_rate",
        "n_questions",
        "p2_status",
        "comparison_group_id",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ranked:
            writer.writerow(
                {
                    "rank": r.retrieval_rank,
                    "experiment_id": r.experiment_id,
                    "retrieval_score": f"{r.retrieval_score:.6f}",
                    "recall_at_5": f"{r.mean_recall_at_5:.6f}",
                    "mrr_at_5": f"{r.mean_mrr_at_5:.6f}",
                    "ndcg_at_5": f"{r.mean_ndcg_at_5:.6f}",
                    "context_precision_at_5": f"{r.mean_context_precision_at_5:.6f}",
                    "unknown_rate": f"{r.unknown_rate:.6f}",
                    "n_questions": r.n_questions,
                    "p2_status": r.p2_status,
                    "comparison_group_id": r.comparison_group_id,
                }
            )


def write_rqi_leaderboard(records: list[ExperimentRecord], out_path: Path) -> None:
    ranked = [r for r in records if r.rqi_rank is not None and r.rqi is not None]
    ranked.sort(key=lambda r: (r.rqi_rank, r.experiment_id))

    fieldnames = [
        "rank",
        "experiment_id",
        "rqi",
        "retrieval_score",
        "judge_correctness_norm",
        "judge_faithfulness_norm",
        "judge_context_relevance_norm",
        "recall_at_5",
        "no_unknown",
        "n_questions",
        "judge_model",
        "prompt_version",
        "comparison_group_id",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ranked:
            writer.writerow(
                {
                    "rank": r.rqi_rank,
                    "experiment_id": r.experiment_id,
                    "rqi": f"{r.rqi:.6f}",
                    "retrieval_score": f"{r.retrieval_score:.6f}",
                    "judge_correctness_norm": (
                        f"{r.mean_judge_correctness / 5.0:.6f}"
                        if r.mean_judge_correctness is not None
                        else ""
                    ),
                    "judge_faithfulness_norm": (
                        f"{r.mean_judge_faithfulness / 5.0:.6f}"
                        if r.mean_judge_faithfulness is not None
                        else ""
                    ),
                    "judge_context_relevance_norm": (
                        f"{r.mean_judge_context_relevance / 5.0:.6f}"
                        if r.mean_judge_context_relevance is not None
                        else ""
                    ),
                    "recall_at_5": f"{r.mean_recall_at_5:.6f}",
                    "no_unknown": f"{1.0 - r.unknown_rate:.6f}",
                    "n_questions": r.n_questions,
                    "judge_model": r.judge_model or "",
                    "prompt_version": r.prompt_version or "",
                    "comparison_group_id": r.comparison_group_id,
                }
            )


def write_full_summary(records: list[ExperimentRecord], out_path: Path) -> None:
    fieldnames = [
        "experiment_id",
        "p2_status",
        "p3_status",
        "retrieval_rank",
        "rqi_rank",
        "retrieval_score",
        "rqi",
        "n_questions",
        "run_valid",
        "generation_failure_rate",
        "mean_recall_at_5",
        "mean_mrr_at_5",
        "mean_ndcg_at_5",
        "mean_context_precision_at_5",
        "unknown_rate",
        "mean_embedding_similarity",
        "mean_official_bertscore_f1",
        "has_p3",
        "judge_model",
        "prompt_version",
        "judge_success_rate",
        "mean_judge_correctness",
        "mean_judge_faithfulness",
        "mean_judge_completeness",
        "mean_judge_hallucination",
        "mean_judge_context_relevance",
        "mean_judge_overall_score",
        "mean_ragas_faithfulness",
        "mean_ragas_answer_relevancy",
        "ragas_faithfulness_nan_rate",
        "ragas_answer_relevancy_nan_rate",
        "qa_hash",
        "comparison_group_id",
        "p2_run_dir",
        "p3_run_dir",
    ]

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(records, key=lambda x: x.experiment_id):
            writer.writerow(
                {
                    "experiment_id": r.experiment_id,
                    "p2_status": r.p2_status,
                    "p3_status": r.p3_status or "",
                    "retrieval_rank": r.retrieval_rank if r.retrieval_rank is not None else "",
                    "rqi_rank": r.rqi_rank if r.rqi_rank is not None else "",
                    "retrieval_score": _fmt(r.retrieval_score),
                    "rqi": _fmt(r.rqi),
                    "n_questions": r.n_questions,
                    "run_valid": r.run_valid,
                    "generation_failure_rate": _fmt(r.generation_failure_rate),
                    "mean_recall_at_5": _fmt(r.mean_recall_at_5),
                    "mean_mrr_at_5": _fmt(r.mean_mrr_at_5),
                    "mean_ndcg_at_5": _fmt(r.mean_ndcg_at_5),
                    "mean_context_precision_at_5": _fmt(r.mean_context_precision_at_5),
                    "unknown_rate": _fmt(r.unknown_rate),
                    "mean_embedding_similarity": _fmt(r.mean_embedding_similarity),
                    "mean_official_bertscore_f1": _fmt(r.mean_official_bertscore_f1),
                    "has_p3": r.has_p3,
                    "judge_model": r.judge_model or "",
                    "prompt_version": r.prompt_version or "",
                    "judge_success_rate": _fmt(r.judge_success_rate),
                    "mean_judge_correctness": _fmt(r.mean_judge_correctness),
                    "mean_judge_faithfulness": _fmt(r.mean_judge_faithfulness),
                    "mean_judge_completeness": _fmt(r.mean_judge_completeness),
                    "mean_judge_hallucination": _fmt(r.mean_judge_hallucination),
                    "mean_judge_context_relevance": _fmt(r.mean_judge_context_relevance),
                    "mean_judge_overall_score": _fmt(r.mean_judge_overall_score),
                    "mean_ragas_faithfulness": _fmt(r.mean_ragas_faithfulness),
                    "mean_ragas_answer_relevancy": _fmt(r.mean_ragas_answer_relevancy),
                    "ragas_faithfulness_nan_rate": _fmt(r.ragas_faithfulness_nan_rate),
                    "ragas_answer_relevancy_nan_rate": _fmt(r.ragas_answer_relevancy_nan_rate),
                    "qa_hash": r.qa_hash or "",
                    "comparison_group_id": r.comparison_group_id,
                    "p2_run_dir": r.p2_run_dir,
                    "p3_run_dir": r.p3_run_dir or "",
                }
            )


def write_leaderboard_json(
    records: list[ExperimentRecord],
    comparison_groups: list[ComparisonGroup],
    cfg: Pipeline4Config,
    out_path: Path,
) -> None:
    retrieval_entries = []
    for r in sorted(
        [x for x in records if x.retrieval_rank is not None],
        key=lambda x: (x.retrieval_rank, x.experiment_id),
    ):
        retrieval_entries.append(
            {
                "rank": r.retrieval_rank,
                "experiment_id": r.experiment_id,
                "retrieval_score": round(r.retrieval_score, 6),
                "recall_at_5": round(r.mean_recall_at_5, 6),
                "mrr_at_5": round(r.mean_mrr_at_5, 6),
                "ndcg_at_5": round(r.mean_ndcg_at_5, 6),
                "context_precision_at_5": round(r.mean_context_precision_at_5, 6),
                "unknown_rate": round(r.unknown_rate, 6),
                "n_questions": r.n_questions,
                "comparison_group_id": r.comparison_group_id,
            }
        )

    rqi_entries = []
    for r in sorted(
        [x for x in records if x.rqi_rank is not None and x.rqi is not None],
        key=lambda x: (x.rqi_rank, x.experiment_id),
    ):
        rqi_entries.append(
            {
                "rank": r.rqi_rank,
                "experiment_id": r.experiment_id,
                "rqi": round(r.rqi, 6),
                "retrieval_score": round(r.retrieval_score, 6),
                "judge_correctness_norm": (
                    round(r.mean_judge_correctness / 5.0, 6)
                    if r.mean_judge_correctness is not None
                    else None
                ),
                "judge_faithfulness_norm": (
                    round(r.mean_judge_faithfulness / 5.0, 6)
                    if r.mean_judge_faithfulness is not None
                    else None
                ),
                "judge_context_relevance_norm": (
                    round(r.mean_judge_context_relevance / 5.0, 6)
                    if r.mean_judge_context_relevance is not None
                    else None
                ),
                "recall_at_5": round(r.mean_recall_at_5, 6),
                "no_unknown": round(1.0 - r.unknown_rate, 6),
                "n_questions": r.n_questions,
                "judge_model": r.judge_model,
                "prompt_version": r.prompt_version,
                "comparison_group_id": r.comparison_group_id,
            }
        )

    groups_data = []
    for group in comparison_groups:
        groups_data.append(
            {
                "group_id": group.group_id,
                "experiment_ids": group.experiment_ids,
                "has_complete_p3": group.has_complete_p3,
                "incomparability_reason": group.incomparability_reason,
            }
        )

    payload = {
        "ranking_mode": cfg.ranking_mode,
        "retrieval_leaderboard": retrieval_entries,
        "rqi_leaderboard": rqi_entries,
        "comparison_groups": groups_data,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_validation_report(
    records: list[ExperimentRecord],
    comparison_groups: list[ComparisonGroup],
    cfg: Pipeline4Config,
    out_path: Path,
) -> None:
    excluded = [r for r in records if r.p2_status != "VALID"]
    warnings_list = []
    for r in records:
        if r.ragas_warnings:
            warnings_list.append(
                {"experiment_id": r.experiment_id, "warnings": r.ragas_warnings}
            )
        if r.p3_issues:
            warnings_list.append(
                {"experiment_id": r.experiment_id, "p3_issues": r.p3_issues}
            )

    cross_group_issues = []
    for group in comparison_groups:
        if group.incomparability_reason:
            cross_group_issues.append(
                {
                    "group_id": group.group_id,
                    "issue": group.incomparability_reason,
                }
            )

    payload = {
        "ranking_mode": cfg.ranking_mode,
        "total_experiments": len(records),
        "ranked_retrieval": sum(1 for r in records if r.retrieval_rank is not None),
        "ranked_rqi": sum(1 for r in records if r.rqi_rank is not None),
        "excluded_experiments": [
            {
                "experiment_id": r.experiment_id,
                "p2_status": r.p2_status,
                "p2_issues": [],
            }
            for r in excluded
        ],
        "warnings": warnings_list,
        "cross_group_incomparability": cross_group_issues,
        "comparison_groups": [
            {
                "group_id": g.group_id,
                "experiment_ids": g.experiment_ids,
                "has_complete_p3": g.has_complete_p3,
            }
            for g in comparison_groups
        ],
        "thresholds": {
            "max_generation_failure_rate": cfg.validation.max_generation_failure_rate,
            "min_judge_success_rate": cfg.validation.min_judge_success_rate,
            "max_ragas_nan_rate": cfg.validation.max_ragas_nan_rate,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_comparison_report(
    records: list[ExperimentRecord],
    comparison_groups: list[ComparisonGroup],
    cfg: Pipeline4Config,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Pipeline 4 Comparison Report")
    lines.append("")
    lines.append(f"**Ranking mode:** `{cfg.ranking_mode}`")
    lines.append(f"**Total experiments:** {len(records)}")
    lines.append(
        f"**Ranked (retrieval):** {sum(1 for r in records if r.retrieval_rank is not None)}"
    )
    lines.append(
        f"**Ranked (RQI):** {sum(1 for r in records if r.rqi_rank is not None)}"
    )
    lines.append("")

    lines.append("## Retrieval Score Weights")
    lines.append("")
    w = cfg.retrieval_score_weights
    lines.append(
        f"| Component | Weight | Formula |"
    )
    lines.append("|-----------|--------|---------|")
    lines.append(f"| Recall@5 | {w.recall_at_5} | `{w.recall_at_5} × Recall@5` |")
    lines.append(f"| MRR@5 | {w.mrr_at_5} | `{w.mrr_at_5} × MRR@5` |")
    lines.append(f"| NDCG@5 | {w.ndcg_at_5} | `{w.ndcg_at_5} × NDCG@5` |")
    lines.append(
        f"| ContextPrecision@5 | {w.context_precision_at_5} | `{w.context_precision_at_5} × ContextPrecision@5` |"
    )
    lines.append("")

    if cfg.ranking_mode == "overall_rag":
        lines.append("## RQI Weights")
        lines.append("")
        rw = cfg.rqi_weights
        lines.append("| Component | Weight | Normalization |")
        lines.append("|-----------|--------|---------------|")
        lines.append(
            f"| Correctness | {rw.correctness} | `judge_correctness / 5` |"
        )
        lines.append(
            f"| Faithfulness | {rw.faithfulness} | `judge_faithfulness / 5` |"
        )
        lines.append(
            f"| Context Relevance | {rw.context_relevance} | `judge_context_relevance / 5` |"
        )
        lines.append(f"| Recall@5 | {rw.recall_at_5} | raw |")
        lines.append(f"| No-Unknown | {rw.no_unknown} | `1 - unknown_rate` |")
        lines.append("")

    lines.append("## Retrieval Leaderboard")
    lines.append("")
    ranked_ret = [r for r in records if r.retrieval_rank is not None]
    ranked_ret.sort(key=lambda r: (r.retrieval_rank, r.experiment_id))
    if ranked_ret:
        lines.append(
            "| Rank | Experiment | Retrieval Score | Recall@5 | MRR@5 | NDCG@5 | CP@5 | Unknown Rate |"
        )
        lines.append("|------|-----------|----------------|----------|-------|--------|------|-------------|")
        for r in ranked_ret:
            lines.append(
                f"| {r.retrieval_rank} | `{r.experiment_id}` | {r.retrieval_score:.4f} "
                f"| {r.mean_recall_at_5:.4f} | {r.mean_mrr_at_5:.4f} "
                f"| {r.mean_ndcg_at_5:.4f} | {r.mean_context_precision_at_5:.4f} "
                f"| {r.unknown_rate:.4f} |"
            )
    else:
        lines.append("_No experiments passed validation for retrieval ranking._")
    lines.append("")

    ranked_rqi = [r for r in records if r.rqi_rank is not None and r.rqi is not None]
    if ranked_rqi:
        ranked_rqi.sort(key=lambda r: (r.rqi_rank, r.experiment_id))
        lines.append("## RQI Leaderboard")
        lines.append("")
        lines.append(
            "| Rank | Experiment | RQI | Correctness/5 | Faithfulness/5 | CR/5 | Recall@5 | 1-Unknown |"
        )
        lines.append("|------|-----------|-----|---------------|----------------|------|----------|-----------|")
        for r in ranked_rqi:
            c_norm = f"{r.mean_judge_correctness / 5.0:.4f}" if r.mean_judge_correctness is not None else "—"
            f_norm = f"{r.mean_judge_faithfulness / 5.0:.4f}" if r.mean_judge_faithfulness is not None else "—"
            cr_norm = f"{r.mean_judge_context_relevance / 5.0:.4f}" if r.mean_judge_context_relevance is not None else "—"
            lines.append(
                f"| {r.rqi_rank} | `{r.experiment_id}` | {r.rqi:.4f} "
                f"| {c_norm} | {f_norm} | {cr_norm} "
                f"| {r.mean_recall_at_5:.4f} | {1.0 - r.unknown_rate:.4f} |"
            )
        lines.append("")

    lines.append("## Comparison Groups")
    lines.append("")
    for group in comparison_groups:
        lines.append(f"### Group: `{group.group_id}`")
        lines.append(f"- **Experiments ({len(group.experiment_ids)}):** {', '.join(group.experiment_ids)}")
        lines.append(f"- **P3 complete:** {group.has_complete_p3}")
        if group.incomparability_reason:
            lines.append(f"- **Warning:** {group.incomparability_reason}")
        lines.append("")

    excluded = [r for r in records if r.p2_status != "VALID"]
    if excluded:
        lines.append("## Excluded Experiments")
        lines.append("")
        for r in excluded:
            lines.append(f"- `{r.experiment_id}`: **{r.p2_status}**")
        lines.append("")

    lines.append("## Metric Definitions")
    lines.append("")
    lines.append(
        "**Retrieval Score** = "
        f"{cfg.retrieval_score_weights.recall_at_5}×Recall@5 + "
        f"{cfg.retrieval_score_weights.mrr_at_5}×MRR@5 + "
        f"{cfg.retrieval_score_weights.ndcg_at_5}×NDCG@5 + "
        f"{cfg.retrieval_score_weights.context_precision_at_5}×ContextPrecision@5"
    )
    lines.append("")
    if cfg.ranking_mode == "overall_rag":
        rw = cfg.rqi_weights
        lines.append(
            "**RQI** = "
            f"{rw.correctness}×(Correctness/5) + "
            f"{rw.faithfulness}×(Faithfulness/5) + "
            f"{rw.context_relevance}×(ContextRelevance/5) + "
            f"{rw.recall_at_5}×Recall@5 + "
            f"{rw.no_unknown}×(1−UnknownRate)"
        )
        lines.append("")
    lines.append(
        "All judge metrics (correctness, faithfulness, context_relevance) are on a 0–5 scale "
        "and are normalized to [0, 1] by dividing by 5. "
        "`judge_overall_score` from Pipeline 3 is **not** used in the RQI formula."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
