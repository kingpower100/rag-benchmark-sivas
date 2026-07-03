from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pipeline3.judge")

REQUIRED_SCORE_KEYS = frozenset({
    "correctness",
    "faithfulness",
    "relevancy",
    "completeness",
    "hallucination",
    "context_relevance",
})
# overall_score is required here only for response completeness validation.
# judge_stage.py immediately overwrites it with compute_weighted_overall_score()
# to guarantee reproducibility regardless of LLM arithmetic errors.
# The LLM's original value is preserved separately as JudgeRowResult.llm_overall_score.
ALL_REQUIRED_KEYS = REQUIRED_SCORE_KEYS | {"overall_score", "reasoning"}


@dataclass
class JudgeResponse:
    correctness: int
    faithfulness: int
    relevancy: int
    completeness: int
    hallucination: int
    context_relevance: int
    overall_score: float
    reasoning: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_correctness": self.correctness,
            "judge_faithfulness": self.faithfulness,
            "judge_relevancy": self.relevancy,
            "judge_completeness": self.completeness,
            "judge_hallucination": self.hallucination,
            "judge_context_relevance": self.context_relevance,
            "judge_overall_score": self.overall_score,
            "judge_reasoning": self.reasoning,
        }


@dataclass
class ParseResult:
    success: bool
    response: JudgeResponse | None = None
    error: str | None = None
    raw_text: str = ""


def parse_judge_response(
    raw_text: str, scale_min: int = 0, scale_max: int = 5
) -> ParseResult:
    if not raw_text.strip():
        return ParseResult(
            success=False, error="Empty response from judge", raw_text=raw_text
        )
    text = _extract_json_block(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as ex:
        return ParseResult(
            success=False, error=f"JSON parse error: {ex}", raw_text=raw_text
        )
    if not isinstance(data, dict):
        return ParseResult(
            success=False,
            error="Response is not a JSON object",
            raw_text=raw_text,
        )
    missing = ALL_REQUIRED_KEYS - set(data.keys())
    if missing:
        return ParseResult(
            success=False,
            error=f"Missing keys in judge response: {sorted(missing)}",
            raw_text=raw_text,
        )
    for key in REQUIRED_SCORE_KEYS:
        val = data[key]
        if not isinstance(val, (int, float)):
            return ParseResult(
                success=False,
                error=f"Key '{key}' must be numeric, got {type(val).__name__}",
                raw_text=raw_text,
            )
        if not (scale_min <= float(val) <= scale_max):
            return ParseResult(
                success=False,
                error=f"Key '{key}'={val} out of range [{scale_min}, {scale_max}]",
                raw_text=raw_text,
            )
    overall = data.get("overall_score")
    if not isinstance(overall, (int, float)):
        return ParseResult(
            success=False,
            error="'overall_score' must be numeric",
            raw_text=raw_text,
        )
    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    response = JudgeResponse(
        correctness=int(data["correctness"]),
        faithfulness=int(data["faithfulness"]),
        relevancy=int(data["relevancy"]),
        completeness=int(data["completeness"]),
        hallucination=int(data["hallucination"]),
        context_relevance=int(data["context_relevance"]),
        overall_score=float(data["overall_score"]),
        reasoning=reasoning,
        raw=data,
    )
    return ParseResult(success=True, response=response, raw_text=raw_text)


def _extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [line for line in lines if not line.startswith("```")]
        text = "\n".join(inner).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
