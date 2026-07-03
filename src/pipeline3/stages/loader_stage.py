from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.pipeline2.io.jsonl import read_jsonl

logger = logging.getLogger("pipeline3.loader")


@dataclass
class LoaderResult:
    rag_rows: list[dict[str, Any]] = field(default_factory=list)
    questions_rows: list[dict[str, Any]] = field(default_factory=list)
    qa_rows: list[dict[str, Any]] = field(default_factory=list)
    gold_rows: list[dict[str, Any]] = field(default_factory=list)
    rag_path: Path = field(default_factory=Path)
    questions_path: Path = field(default_factory=Path)
    qa_path: Path = field(default_factory=Path)
    gold_contexts_path: Path = field(default_factory=Path)


def load_inputs(cfg: Any, project_root: Path) -> LoaderResult:
    """Stage 1: Load all evaluation inputs."""
    inputs = cfg.inputs

    rag_path = _resolve(project_root, inputs.pipeline1_results_path)
    questions_path = _resolve(project_root, inputs.questions_path)
    qa_path = _resolve(project_root, inputs.qa_path)
    gold_path = _resolve(project_root, inputs.gold_contexts_path)

    logger.info("Loading Pipeline 1 results: %s", rag_path)
    rag_rows = read_jsonl(rag_path)
    logger.info("Loaded %d Pipeline 1 result rows", len(rag_rows))

    logger.info("Loading questions: %s", questions_path)
    questions_rows = read_jsonl(questions_path)
    logger.info("Loaded %d question rows", len(questions_rows))

    logger.info("Loading QA ground truth: %s", qa_path)
    qa_rows = read_jsonl(qa_path)
    logger.info("Loaded %d QA rows", len(qa_rows))

    gold_rows: list[dict[str, Any]] = []
    if gold_path.exists():
        logger.info("Loading gold contexts: %s", gold_path)
        gold_rows = read_jsonl(gold_path)
        logger.info("Loaded %d gold context rows", len(gold_rows))
    else:
        logger.warning(
            "Gold contexts file not found at %s, proceeding without it", gold_path
        )

    return LoaderResult(
        rag_rows=rag_rows,
        questions_rows=questions_rows,
        qa_rows=qa_rows,
        gold_rows=gold_rows,
        rag_path=rag_path,
        questions_path=questions_path,
        qa_path=qa_path,
        gold_contexts_path=gold_path,
    )


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path
