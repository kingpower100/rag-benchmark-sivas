from __future__ import annotations

import logging
from typing import Any

from src.pipeline3.metrics.ragas_metrics import RagasEvaluator, RagasResults, RagasRow
from src.pipeline3.stages.validation_stage import (
    _extract_context_texts,
    _resolve_id,
    _resolve_qa_answer,
)

logger = logging.getLogger("pipeline3.ragas")


def run_ragas_stage(
    rag_rows: list[dict[str, Any]],
    qa_by_id: dict[str, dict[str, Any]],
    evaluator: RagasEvaluator,
) -> RagasResults:
    """Stage 3: RAGAS evaluation."""
    ragas_rows: list[RagasRow] = []
    for row in rag_rows:
        qid = _resolve_id(row)
        question = str(row.get("question", "")).strip()
        generated_answer = str(row.get("generated_answer", "")).strip()
        context_texts = _extract_context_texts(row)
        qa_row = qa_by_id.get(qid, {})
        ground_truth = _resolve_qa_answer(qa_row)
        ragas_rows.append(
            RagasRow(
                question_id=qid,
                question=question,
                answer=generated_answer,
                contexts=context_texts,
                ground_truth=ground_truth,
            )
        )
    return evaluator.evaluate(ragas_rows)
