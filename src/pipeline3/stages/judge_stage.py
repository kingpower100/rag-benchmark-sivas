from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tqdm.auto import tqdm

from src.pipeline3.judge.ollama_client import OllamaClient, OllamaClientError
from src.pipeline3.judge.prompts import build_combined_judge_prompt, format_context
from src.pipeline3.judge.response_parser import JudgeResponse, ParseResult, parse_judge_response
from src.pipeline3.metrics.judge_metrics import compute_weighted_overall_score
from src.pipeline3.schemas.pipeline3_config_schema import P3JudgeConfig, P3LLMJudgeConfig
from src.pipeline3.stages.validation_stage import (
    _extract_context_texts,
    _resolve_id,
    _resolve_qa_answer,
)

logger = logging.getLogger("pipeline3.judge")


@dataclass
class JudgeRowResult:
    question_id: str
    success: bool
    response: JudgeResponse | None = None
    error: str | None = None
    retry_count: int = 0
    latency_ms: float = 0.0
    raw_prompt: str = ""
    raw_response: str = ""
    context_truncated: bool = False
    # The LLM's self-assessed overall_score, saved for audit transparency.
    # The official judge_overall_score in response.overall_score is recomputed
    # deterministically by compute_weighted_overall_score() to ensure reproducibility
    # regardless of LLM arithmetic errors.
    llm_overall_score: float | None = None


@dataclass
class JudgeStageResult:
    rows: list[JudgeRowResult] = field(default_factory=list)
    total: int = 0
    successes: int = 0
    failures: int = 0
    failure_rate: float = 0.0


def run_judge_stage(
    rag_rows: list[dict[str, Any]],
    qa_by_id: dict[str, dict[str, Any]],
    judge_cfg: P3JudgeConfig,
    llm_judge_cfg: P3LLMJudgeConfig,
) -> JudgeStageResult:
    """Stage 4: LLM-as-Judge evaluation."""
    client = OllamaClient(
        base_url=judge_cfg.base_url,
        model=judge_cfg.model,
        temperature=judge_cfg.temperature,
        timeout_seconds=judge_cfg.timeout_seconds,
    )

    results: list[JudgeRowResult] = []
    for row in tqdm(rag_rows, desc="LLM-as-Judge", unit="question"):
        row_result = _evaluate_single(row, qa_by_id, client, judge_cfg, llm_judge_cfg)
        results.append(row_result)

    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes
    return JudgeStageResult(
        rows=results,
        total=len(results),
        successes=successes,
        failures=failures,
        failure_rate=failures / len(results) if results else 0.0,
    )


def _evaluate_single(
    row: dict[str, Any],
    qa_by_id: dict[str, dict[str, Any]],
    client: OllamaClient,
    judge_cfg: P3JudgeConfig,
    llm_judge_cfg: P3LLMJudgeConfig,
) -> JudgeRowResult:
    qid = _resolve_id(row)
    question = str(row.get("question", "")).strip()
    generated_answer = str(row.get("generated_answer", "")).strip()
    context_texts = _extract_context_texts(row)
    context_str, context_truncated = format_context(
        context_texts,
        max_chars=llm_judge_cfg.max_context_chars,
        question_id=qid,
    )
    qa_row = qa_by_id.get(qid, {})
    ground_truth = _resolve_qa_answer(qa_row)

    prompt = build_combined_judge_prompt(
        question=question,
        ground_truth=ground_truth,
        context=context_str,
        generated_answer=generated_answer,
    )

    last_error: str = ""
    raw_response: str = ""
    latency_ms: float = 0.0

    for attempt in range(judge_cfg.max_retries):
        t0 = time.perf_counter()
        try:
            raw_response = client.generate(prompt)
        except OllamaClientError as ex:
            latency_ms = (time.perf_counter() - t0) * 1000
            last_error = str(ex)
            logger.warning(
                "Judge call failed for qid=%s attempt=%d: %s", qid, attempt + 1, ex
            )
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        parse_result: ParseResult = parse_judge_response(
            raw_response,
            scale_min=llm_judge_cfg.scoring.scale_min,
            scale_max=llm_judge_cfg.scoring.scale_max,
        )
        if parse_result.success and parse_result.response is not None:
            # Save the LLM's own overall_score before overwriting it.
            # The official score is computed deterministically from configured weights.
            llm_overall = parse_result.response.overall_score
            recomputed_overall = compute_weighted_overall_score(
                parse_result.response,
                llm_judge_cfg.weights,
                llm_judge_cfg.scoring,
            )
            parse_result.response.overall_score = recomputed_overall
            return JudgeRowResult(
                question_id=qid,
                success=True,
                response=parse_result.response,
                retry_count=attempt,
                latency_ms=latency_ms,
                raw_prompt=prompt,
                raw_response=raw_response,
                context_truncated=context_truncated,
                llm_overall_score=llm_overall,
            )
        last_error = parse_result.error or "Unknown parse error"
        logger.warning(
            "Judge parse failed for qid=%s attempt=%d: %s",
            qid,
            attempt + 1,
            last_error,
        )

    return JudgeRowResult(
        question_id=qid,
        success=False,
        error=last_error,
        retry_count=judge_cfg.max_retries,
        latency_ms=latency_ms,
        raw_prompt=prompt,
        raw_response=raw_response,
        context_truncated=context_truncated,
    )
