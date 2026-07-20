from __future__ import annotations

from collections.abc import Iterable
import math
import warnings

from src.pipeline1.schemas.retrieval import RetrievalItem


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.requested_device = device
        self.model = CrossEncoder(model_name, device=device)
        self.runtime_device = self._resolve_runtime_device()
        self._validate_device_selection()

    def rerank(self, question: str, items: list[RetrievalItem], top_k: int) -> list[RetrievalItem]:
        if not items:
            return []
        scores = self._normalize_scores(self.model.predict([(question, item.text) for item in items]), len(items))
        scored = [
            item.model_copy(
                update={
                    "score": float(score) + item.metadata_boost,
                    "rerank_score": float(score),
                    "ranking_score_type": "rerank_score_plus_metadata" if item.metadata_boost else "rerank_score",
                    "retrieval_source": item.retrieval_source,
                    "metadata": {**item.metadata, "reranker_original_rank": original_rank},
                }
            )
            for original_rank, (item, score) in enumerate(zip(items, scores), start=1)
        ]
        # Deterministic tie policy: rerank score descending, original retrieval
        # rank ascending, then chunk ID ascending if an item lacks rank metadata.
        return sorted(
            scored,
            key=lambda item: (
                -float(item.score),
                int(item.metadata.get("reranker_original_rank", 10**12)),
                str(item.chunk_id),
            ),
        )[:top_k]

    @staticmethod
    def _normalize_scores(raw_scores, candidate_count: int) -> list[float]:
        if raw_scores is None:
            raise RuntimeError(f"Reranker returned None scores for {candidate_count} candidates.")
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, (str, bytes)) or not isinstance(raw_scores, Iterable):
            raise RuntimeError(f"Reranker returned non-iterable scores for {candidate_count} candidates.")
        scores = list(raw_scores)
        if len(scores) != candidate_count:
            raise RuntimeError(f"Reranker returned {len(scores)} scores for {candidate_count} candidates.")
        normalized: list[float] = []
        for index, score in enumerate(scores, start=1):
            try:
                value = float(score)
            except (TypeError, ValueError) as ex:
                raise RuntimeError(f"Reranker score {index} is not numeric: {score!r}.") from ex
            if not math.isfinite(value):
                raise RuntimeError(f"Reranker score {index} is not finite: {score!r}.")
            normalized.append(value)
        return normalized

    def _resolve_runtime_device(self) -> str:
        device = getattr(self.model, "device", None)
        if device is not None:
            return str(device)
        if hasattr(self.model, "model"):
            try:
                parameter = next(self.model.model.parameters())
                return str(parameter.device)
            except Exception:
                pass
        target = getattr(self.model, "_target_device", None)
        if target is not None:
            return str(target)
        return str(self.requested_device)

    def _validate_device_selection(self) -> None:
        requested_cuda = str(self.requested_device).startswith("cuda")
        runtime_cuda = str(self.runtime_device).startswith("cuda")
        if requested_cuda and not runtime_cuda:
            warnings.warn(
                f"CrossEncoder requested device={self.requested_device!r} but runtime device resolved to {self.runtime_device!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
