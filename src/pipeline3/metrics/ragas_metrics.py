from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pipeline3.ragas")


@dataclass
class RagasRow:
    question_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


@dataclass
class RagasResults:
    rows: list[dict[str, Any]] = field(default_factory=list)
    enabled_metrics: list[str] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False
    nan_counts: dict[str, int] = field(default_factory=dict)
    valid_counts: dict[str, int] = field(default_factory=dict)


def build_ragas_evaluator(cfg: Any) -> "RagasEvaluator":
    return RagasEvaluator(cfg)


class RagasEvaluator:
    def __init__(self, ragas_cfg: Any) -> None:
        self._cfg = ragas_cfg

    def evaluate(self, rows: list[RagasRow]) -> RagasResults:
        if not self._cfg.enabled:
            return RagasResults(skipped=True)
        try:
            return self._run_ragas(rows)
        except Exception as ex:
            logger.error("RAGAS evaluation failed: %s", ex, exc_info=True)
            return RagasResults(error=str(ex), skipped=True)

    def _run_ragas(self, rows: list[RagasRow]) -> RagasResults:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI
        except ImportError as ex:
            logger.error("RAGAS or langchain_openai not available: %s", ex)
            return RagasResults(error=f"Import error: {ex}", skipped=True)

        llm = LangchainLLMWrapper(
            ChatOpenAI(
                base_url=self._cfg.llm_base_url,
                api_key="ollama",
                model=self._cfg.llm_model,
                temperature=self._cfg.llm_temperature,
                timeout=self._cfg.timeout_seconds,
            )
        )

        try:
            embeddings = self._build_embeddings()
        except Exception as ex:
            logger.warning(
                "Could not build RAGAS embeddings, AnswerRelevancy will be skipped: %s", ex
            )
            embeddings = None

        metrics = self._build_metrics(llm, embeddings)
        if not metrics:
            return RagasResults(skipped=True, error="No RAGAS metrics enabled")

        enabled_names = [m.name for m in metrics]
        logger.info(
            "Running RAGAS evaluation with metrics: %s on %d rows",
            enabled_names,
            len(rows),
        )

        try:
            from datasets import Dataset
        except ImportError as ex:
            return RagasResults(
                error=f"datasets package not available: {ex}", skipped=True
            )

        dataset = Dataset.from_dict(
            {
                "question": [r.question for r in rows],
                "answer": [r.answer for r in rows],
                "contexts": [r.contexts if r.contexts else [""] for r in rows],
                "ground_truth": [r.ground_truth for r in rows],
            }
        )

        try:
            result = ragas_evaluate(dataset=dataset, metrics=metrics)
        except Exception as ex:
            logger.error("ragas.evaluate() failed: %s", ex, exc_info=True)
            return RagasResults(error=str(ex), skipped=True)

        result_df = result.to_pandas()

        # Count NaN values per metric for transparency in the manifest and report.
        nan_counts: dict[str, int] = {}
        valid_counts: dict[str, int] = {}
        for metric_name in enabled_names:
            col = metric_name
            if col in result_df.columns:
                n_nan = int(result_df[col].isna().sum())
                nan_counts[f"ragas_{col}"] = n_nan
                valid_counts[f"ragas_{col}"] = len(rows) - n_nan
                if n_nan > 0:
                    logger.warning(
                        "RAGAS metric '%s' produced NaN for %d/%d rows",
                        col, n_nan, len(rows),
                    )
            else:
                nan_counts[f"ragas_{col}"] = len(rows)
                valid_counts[f"ragas_{col}"] = 0
                logger.warning("RAGAS metric '%s' column missing from result dataframe", col)

        per_row = []
        for i, row in enumerate(rows):
            row_metrics: dict[str, Any] = {"question_id": row.question_id}
            for metric_name in enabled_names:
                col = metric_name
                if col in result_df.columns:
                    val = result_df.iloc[i][col]
                    # NaN check: NaN != NaN is True
                    row_metrics[f"ragas_{col}"] = None if val != val else float(val)
                else:
                    row_metrics[f"ragas_{col}"] = None
            per_row.append(row_metrics)

        return RagasResults(
            rows=per_row,
            enabled_metrics=enabled_names,
            nan_counts=nan_counts,
            valid_counts=valid_counts,
        )

    def _build_embeddings(self) -> Any:
        from sentence_transformers import SentenceTransformer
        from ragas.embeddings import LangchainEmbeddingsWrapper

        model = SentenceTransformer(self._cfg.embeddings_model)

        class _STAdapter:
            def __init__(self, _model: Any) -> None:
                self._m = _model

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._m.encode(texts, show_progress_bar=False).tolist()

            def embed_query(self, text: str) -> list[float]:
                return self._m.encode([text], show_progress_bar=False)[0].tolist()

        return LangchainEmbeddingsWrapper(_STAdapter(model))

    def _build_metrics(self, llm: Any, embeddings: Any) -> list[Any]:
        metrics: list[Any] = []
        cfg_metrics = self._cfg.metrics
        try:
            from ragas.metrics import (
                Faithfulness,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
            )
        except ImportError as ex:
            logger.error("Could not import RAGAS metric classes: %s", ex)
            return []

        if cfg_metrics.faithfulness:
            metrics.append(Faithfulness(llm=llm))
        if cfg_metrics.context_precision:
            metrics.append(ContextPrecision(llm=llm))
        if cfg_metrics.context_recall:
            metrics.append(ContextRecall(llm=llm))
        if cfg_metrics.answer_relevancy:
            if embeddings is not None:
                metrics.append(AnswerRelevancy(llm=llm, embeddings=embeddings))
            else:
                logger.warning("AnswerRelevancy skipped: embeddings not available")
        return metrics
