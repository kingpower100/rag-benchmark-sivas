from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from tqdm.auto import tqdm

from src.pipeline1.generation.factory import build_generator
from src.pipeline1.observability.events import EventType
from src.pipeline1.orchestration.parser import parse_orchestration_response
from src.pipeline1.orchestration.prompt import ORCHESTRATION_PROMPT_VERSION, build_orchestration_prompt
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.stages.base import BaseStage, StageInput, StageOutput

MAX_ORCHESTRATION_RETRIES = 3
ORCHESTRATION_BACKOFF_BASE_SECONDS = 1.0


@dataclass(frozen=True)
class OrchestrationStageOutput(StageOutput):
    queries: list[QueryRecord] = field(default_factory=list)
    generator: object = None


class OrchestrationStage(BaseStage):
    stage_name = "orchestration"

    def __init__(
        self,
        cfg: PipelineConfig,
        chunks: list,
        event_writer=None,
        logger=None,
        generator_factory: Callable = build_generator,
    ) -> None:
        self.cfg = cfg
        self.chunks = chunks
        self.event_writer = event_writer
        self.logger = logger
        self.generator_factory = generator_factory

    def run(self, stage_input: StageInput) -> OrchestrationStageOutput:
        queries = list(stage_input.payload["queries"])
        categories = _categories_from_chunks(self.chunks, self.cfg.retrieval.category_field)
        generator = self.generator_factory(self.cfg.orchestration)
        orchestrated: list[QueryRecord] = []
        for row_index, query in enumerate(tqdm(queries, desc="Orchestrating questions", unit="question"), start=1):
            orchestrated.append(self._orchestrate_query(generator, query, categories, row_index, len(queries)))
        return OrchestrationStageOutput(
            stage_name=self.stage_name,
            artifacts={"queries": orchestrated},
            diagnostics={"orchestrated_rows": len(orchestrated), "available_categories": categories},
            metadata={
                "llm_model": self.cfg.orchestration.model_name,
                "prompt_version": ORCHESTRATION_PROMPT_VERSION,
                "tasks": list(self.cfg.orchestration.tasks),
            },
            queries=orchestrated,
            generator=generator,
        )

    def _orchestrate_query(
        self,
        generator,
        query: QueryRecord,
        categories: list[str],
        row_index: int,
        total_rows: int,
    ) -> QueryRecord:
        if self.logger:
            self.logger.info(
                "row_start phase=orchestration question_id=%s row=%s/%s",
                query.question_id,
                row_index,
                total_rows,
            )
        prompt = build_orchestration_prompt(query.question, categories)
        self._write_event(
            stage="orchestration",
            event_type=EventType.GENERATION_START,
            message="Question orchestration started.",
            question_id=query.question_id,
            metrics={"llm_model": self.cfg.orchestration.model_name},
        )
        start = time.perf_counter()
        parsed, result, error, attempts = self._orchestrate_with_retries(
            generator,
            prompt,
            query,
            categories,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        orchestrated = query.model_copy(update=parsed)
        self._write_event(
            stage="orchestration",
            event_type=EventType.GENERATION_END,
            message="Question orchestration completed.",
            question_id=query.question_id,
            duration_ms=duration_ms,
            metrics={
                "input_tokens": result.input_tokens if result is not None else 0,
                "output_tokens": result.output_tokens if result is not None else 0,
                "category_validated": orchestrated.category_validated,
                "orchestration_failed": error is not None,
                "orchestration_attempts": attempts,
            },
            diagnostics={
                "decision": "Category Validation",
                "cleaned_question": orchestrated.cleaned_question,
                "detected_category": orchestrated.detected_category,
                "category_validated": orchestrated.category_validated,
                "category_validation_reason": orchestrated.category_validation_reason,
                "error": error,
                "orchestration_attempts": attempts,
            },
        )
        return orchestrated

    def _orchestrate_with_retries(
        self,
        generator,
        prompt: str,
        query: QueryRecord,
        categories: list[str],
    ):
        last_error = None
        last_result = None
        for attempt in range(1, MAX_ORCHESTRATION_RETRIES + 1):
            try:
                result = generator.generate(prompt)
                last_result = result
                parsed = parse_orchestration_response(result.answer, query.question, categories)
                parsed["orchestration_error"] = None
                return parsed, result, None, attempt
            except Exception as ex:
                last_error = str(ex)
                if self.logger:
                    self.logger.warning(
                        "row_orchestration_attempt_failed question_id=%s attempt=%s/%s error=%s",
                        query.question_id,
                        attempt,
                        MAX_ORCHESTRATION_RETRIES,
                        last_error,
                        exc_info=True,
                    )
                self._write_event(
                    stage="orchestration",
                    event_type=EventType.GENERATION_ERROR,
                    message="Orchestration attempt failed.",
                    question_id=query.question_id,
                    diagnostics={
                        "error": last_error,
                        "attempt": attempt,
                        "max_retries": MAX_ORCHESTRATION_RETRIES,
                        "will_retry": attempt < MAX_ORCHESTRATION_RETRIES,
                    },
                )
                if attempt < MAX_ORCHESTRATION_RETRIES:
                    time.sleep(ORCHESTRATION_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
        self._write_event(
            stage="orchestration",
            event_type=EventType.GENERATION_ERROR,
            message="Orchestration failed after retries; falling back to raw question and global retrieval.",
            question_id=query.question_id,
            diagnostics={"error": last_error, "attempts": MAX_ORCHESTRATION_RETRIES},
        )
        return (
            {
                "cleaned_question": query.question,
                "detected_category": None,
                "category_validated": False,
                "category_validation_reason": "orchestration failed before category validation",
                "orchestration_error": last_error,
            },
            last_result,
            last_error,
            MAX_ORCHESTRATION_RETRIES,
        )

    def _write_event(self, **kwargs) -> None:
        if self.event_writer is not None:
            self.event_writer.write(**kwargs)


def _categories_from_chunks(chunks: list, category_field: str) -> list[str]:
    categories = {
        str(chunk.metadata.get(category_field)).strip()
        for chunk in chunks
        if getattr(chunk, "metadata", None) and chunk.metadata.get(category_field) is not None and str(chunk.metadata.get(category_field)).strip()
    }
    return sorted(categories)
