from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import ValidationThresholds

P2_VALID = "VALID"
P2_EXCLUDED_RUN_INVALID = "EXCLUDED_RUN_INVALID"
P2_EXCLUDED_FAILURE_RATE = "EXCLUDED_FAILURE_RATE"

P3_VALID = "VALID"
P3_JUDGE_WARNING = "JUDGE_WARNING"
P3_NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass
class ExperimentValidation:
    experiment_id: str
    p2_status: str
    p2_issues: list[str] = field(default_factory=list)
    p3_status: Optional[str] = None
    p3_issues: list[str] = field(default_factory=list)
    ragas_warnings: list[str] = field(default_factory=list)

    @property
    def p2_excluded(self) -> bool:
        return self.p2_status in (P2_EXCLUDED_RUN_INVALID, P2_EXCLUDED_FAILURE_RATE)

    @property
    def p3_invalid(self) -> bool:
        return self.p3_status == P3_JUDGE_WARNING


def validate_p2(p2: P2Summary, thresholds: ValidationThresholds) -> ExperimentValidation:
    issues: list[str] = []

    if not p2.run_valid:
        issues.append("P2 run_valid=False: experiment marked invalid by Pipeline 2")
        return ExperimentValidation(
            experiment_id=p2.experiment_id,
            p2_status=P2_EXCLUDED_RUN_INVALID,
            p2_issues=issues,
        )

    if p2.generation_failure_rate > thresholds.max_generation_failure_rate:
        issues.append(
            f"generation_failure_rate={p2.generation_failure_rate:.4f} exceeds "
            f"max={thresholds.max_generation_failure_rate:.4f}"
        )
        return ExperimentValidation(
            experiment_id=p2.experiment_id,
            p2_status=P2_EXCLUDED_FAILURE_RATE,
            p2_issues=issues,
        )

    return ExperimentValidation(
        experiment_id=p2.experiment_id,
        p2_status=P2_VALID,
        p2_issues=issues,
    )


def validate_p3(
    p3: Optional[P3Summary], thresholds: ValidationThresholds
) -> tuple[str, list[str], list[str]]:
    if p3 is None:
        return P3_NOT_AVAILABLE, [], []

    issues: list[str] = []
    ragas_warnings: list[str] = []

    if p3.judge_success_rate < thresholds.min_judge_success_rate:
        issues.append(
            f"judge_success_rate={p3.judge_success_rate:.4f} below "
            f"min={thresholds.min_judge_success_rate:.4f}"
        )
        return P3_JUDGE_WARNING, issues, ragas_warnings

    if p3.ragas_faithfulness_nan_rate is not None:
        if p3.ragas_faithfulness_nan_rate > thresholds.max_ragas_nan_rate:
            ragas_warnings.append(
                f"ragas_faithfulness NaN rate={p3.ragas_faithfulness_nan_rate:.4f} "
                f"exceeds max={thresholds.max_ragas_nan_rate:.4f}"
            )

    if p3.ragas_answer_relevancy_nan_rate is not None:
        if p3.ragas_answer_relevancy_nan_rate > thresholds.max_ragas_nan_rate:
            ragas_warnings.append(
                f"ragas_answer_relevancy NaN rate={p3.ragas_answer_relevancy_nan_rate:.4f} "
                f"exceeds max={thresholds.max_ragas_nan_rate:.4f}"
            )

    return P3_VALID, issues, ragas_warnings


def validate_all(
    p2_summaries: list[P2Summary],
    p3_map: dict[str, Optional[P3Summary]],
    thresholds: ValidationThresholds,
) -> dict[str, ExperimentValidation]:
    results: dict[str, ExperimentValidation] = {}
    for p2 in p2_summaries:
        val = validate_p2(p2, thresholds)
        p3 = p3_map.get(p2.experiment_id)
        p3_status, p3_issues, ragas_warnings = validate_p3(p3, thresholds)
        val.p3_status = p3_status
        val.p3_issues = p3_issues
        val.ragas_warnings = ragas_warnings
        results[p2.experiment_id] = val
    return results


@dataclass
class ComparisonGroup:
    key: tuple
    experiment_ids: list[str] = field(default_factory=list)
    has_complete_p3: bool = False
    incomparability_reason: Optional[str] = None

    @property
    def group_id(self) -> str:
        return "|".join(str(k) for k in self.key)


def build_comparison_groups(
    p2_summaries: list[P2Summary],
    p3_map: dict[str, Optional[P3Summary]],
    validations: dict[str, ExperimentValidation],
    ranking_mode: str,
) -> list[ComparisonGroup]:
    groups: dict[tuple, ComparisonGroup] = {}

    for p2 in p2_summaries:
        val = validations[p2.experiment_id]
        if val.p2_excluded:
            continue

        qa_hash = p2.qa_hash or f"UNKNOWN_QA_{p2.experiment_id}"
        n_q = p2.n_questions

        if ranking_mode == "overall_rag":
            p3 = p3_map.get(p2.experiment_id)
            judge_model = p3.judge_model if p3 else "NO_P3"
            prompt_version = p3.prompt_version if p3 else "NO_P3"
            key = (qa_hash, n_q, judge_model, prompt_version)
        else:
            key = (qa_hash, n_q)

        if key not in groups:
            groups[key] = ComparisonGroup(key=key)
        groups[key].experiment_ids.append(p2.experiment_id)

    group_list = list(groups.values())

    for group in group_list:
        all_have_p3 = all(
            p3_map.get(eid) is not None for eid in group.experiment_ids
        )
        group.has_complete_p3 = all_have_p3
        if not all_have_p3 and ranking_mode == "overall_rag":
            missing = [
                eid
                for eid in group.experiment_ids
                if p3_map.get(eid) is None
            ]
            group.incomparability_reason = (
                f"P3 missing for experiments: {missing}; RQI ranking skipped for this group"
            )

    return group_list
