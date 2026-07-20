from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import ValidationThresholds

P2_VALID = "VALID"
P2_EXCLUDED_RUN_INVALID = "EXCLUDED_RUN_INVALID"
P2_EXCLUDED_FAILURE_RATE = "EXCLUDED_FAILURE_RATE"
P2_EXCLUDED_STRICT_AUDIT = "EXCLUDED_STRICT_AUDIT"

P3_VALID = "VALID"
P3_JUDGE_WARNING = "JUDGE_WARNING"
P3_NOT_AVAILABLE = "NOT_AVAILABLE"
P3_INCOMPLETE = "INCOMPLETE"


@dataclass
class ExperimentValidation:
    experiment_id: str
    p2_status: str
    p2_issues: list[str] = field(default_factory=list)
    p3_status: Optional[str] = None
    p3_issues: list[str] = field(default_factory=list)
    ragas_warnings: list[str] = field(default_factory=list)
    p2_eligible: bool = False
    p3_eligible: bool = False
    retrieval_leaderboard_eligible: bool = False
    overall_leaderboard_eligible: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)

    @property
    def p2_excluded(self) -> bool:
        return not self.retrieval_leaderboard_eligible

    @property
    def p3_invalid(self) -> bool:
        return not self.p3_eligible


def validate_p2(p2: P2Summary, thresholds: ValidationThresholds) -> ExperimentValidation:
    issues: list[str] = []
    reasons: list[str] = []

    if not p2.audit_manifest_present:
        issues.append("P2 eval_manifest.json is missing; legacy runs are not official-rank eligible.")
        reasons.append("p2_manifest_missing")
    if p2.final_verdict != "valid":
        issues.append(f"P2 final_verdict={p2.final_verdict!r}; expected 'valid'.")
        reasons.append("p2_final_verdict_invalid")
    if p2.strict_audit_pass is not True:
        issues.append(f"P2 strict_audit_pass={p2.strict_audit_pass!r}; expected True.")
        reasons.append("p2_strict_audit_failed")
    if p2.fake_run_suspicious:
        issues.append(f"P2 fake-run evidence present: {p2.fake_run_suspicious_checks}")
        reasons.append("p2_fake_run_detected")
    if not p2.required_outputs_present:
        issues.append(f"P2 required output file(s) missing: {p2.missing_required_outputs}")
        reasons.append("p2_required_output_missing")

    expected = p2.expected_question_count or p2.n_questions
    evaluated = int(p2.row_counts.get("evaluated_rows", p2.n_questions))
    pipeline1_rows = int(p2.row_counts.get("pipeline1_results", evaluated))
    if expected <= 0 or p2.n_questions != expected or evaluated != expected or pipeline1_rows != expected:
        issues.append(
            "P2 row-count mismatch: "
            f"summary={p2.n_questions}, evaluated={evaluated}, pipeline1={pipeline1_rows}, expected={expected}."
        )
        reasons.append("p2_row_count_mismatch")
    if p2.duplicate_question_ids:
        issues.append(f"P2 duplicate question IDs: {p2.duplicate_question_ids[:10]}")
        reasons.append("p2_duplicate_question_ids")
    if p2.question_ids and len(set(p2.question_ids)) != expected:
        issues.append("P2 question ID set size does not match expected official question count.")
        reasons.append("p2_question_id_mismatch")

    if not p2.run_valid:
        issues.append("P2 run_valid=False: experiment marked invalid by Pipeline 2")
        reasons.append("p2_run_invalid")
        return ExperimentValidation(
            experiment_id=p2.experiment_id,
            p2_status=P2_EXCLUDED_RUN_INVALID,
            p2_issues=issues,
            exclusion_reasons=_dedupe_reasons(reasons),
        )

    if p2.generation_failure_rate > thresholds.max_generation_failure_rate:
        issues.append(
            f"generation_failure_rate={p2.generation_failure_rate:.4f} exceeds "
            f"max={thresholds.max_generation_failure_rate:.4f}"
        )
        reasons.append("p2_generation_failure_rate_exceeded")
        return ExperimentValidation(
            experiment_id=p2.experiment_id,
            p2_status=P2_EXCLUDED_FAILURE_RATE,
            p2_issues=issues,
            exclusion_reasons=_dedupe_reasons(reasons),
        )

    if reasons:
        return ExperimentValidation(
            experiment_id=p2.experiment_id,
            p2_status=P2_EXCLUDED_STRICT_AUDIT,
            p2_issues=issues,
            exclusion_reasons=_dedupe_reasons(reasons),
        )

    return ExperimentValidation(
        experiment_id=p2.experiment_id,
        p2_status=P2_VALID,
        p2_issues=issues,
        p2_eligible=True,
        retrieval_leaderboard_eligible=True,
    )


def validate_p3(
    p3: Optional[P3Summary],
    thresholds: ValidationThresholds,
    p2: Optional[P2Summary] = None,
) -> tuple[str, list[str], list[str]]:
    if p3 is None:
        return P3_NOT_AVAILABLE, [], []

    issues: list[str] = []
    ragas_warnings: list[str] = []
    expected = p2.expected_question_count if p2 and p2.expected_question_count else p3.expected_question_count
    if not p3.summary_present:
        issues.append("P3 semantic_summary.csv is missing.")
    if not p3.row_output_present:
        issues.append("P3 per_question_semantic_metrics.csv is missing.")
    if p3.validation_passed is not True:
        issues.append(f"P3 validation.passed={p3.validation_passed!r}; expected True.")
    if expected is None or expected <= 0:
        issues.append("P3 expected official question count is unavailable.")
    elif p3.expected_question_count != expected:
        issues.append(f"P3 expected count={p3.expected_question_count}; expected {expected}.")
    if expected and p3.n_questions != expected:
        issues.append(f"P3 summary n_questions={p3.n_questions}; expected {expected}.")
    if expected and len(p3.question_ids) != expected:
        issues.append(f"P3 row output question count={len(p3.question_ids)}; expected {expected}.")
    if p3.duplicate_question_ids:
        issues.append(f"P3 duplicate question IDs: {p3.duplicate_question_ids[:10]}")
    if p2 is not None and p2.question_ids and p3.question_ids and set(p3.question_ids) != set(p2.question_ids):
        issues.append("P3 question IDs do not match Pipeline 2 evaluated question IDs.")
    if issues:
        return P3_INCOMPLETE, issues, ragas_warnings

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

    if p3.ragas_context_recall_nan_rate is not None:
        if p3.ragas_context_recall_nan_rate > thresholds.max_ragas_nan_rate:
            ragas_warnings.append(
                f"ragas_context_recall NaN rate={p3.ragas_context_recall_nan_rate:.4f} "
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
        p3_status, p3_issues, ragas_warnings = validate_p3(p3, thresholds, p2)
        val.p3_status = p3_status
        val.p3_issues = p3_issues
        val.ragas_warnings = ragas_warnings
        val.p3_eligible = p3_status == P3_VALID
        val.overall_leaderboard_eligible = val.retrieval_leaderboard_eligible and val.p3_eligible
        if p3_status == P3_INCOMPLETE:
            val.exclusion_reasons = _dedupe_reasons([*val.exclusion_reasons, "p3_partial_run"])
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
        if not val.retrieval_leaderboard_eligible:
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
            validations[eid].overall_leaderboard_eligible for eid in group.experiment_ids
        )
        group.has_complete_p3 = all_have_p3
        if not all_have_p3 and ranking_mode == "overall_rag":
            missing = [
                eid
                for eid in group.experiment_ids
                if not validations[eid].overall_leaderboard_eligible
            ]
            group.incomparability_reason = (
                f"P3 incomplete or invalid for experiments: {missing}; RQI ranking skipped for those runs"
            )

    return group_list


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    return list(dict.fromkeys(reasons))
