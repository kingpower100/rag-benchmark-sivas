from __future__ import annotations

from src.pipeline1.schemas.retrieval import RetrievalItem


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, items: list[RetrievalItem], top_k: int) -> list[RetrievalItem]:
        if not items:
            return []
        scores = self.model.predict([(question, item.text) for item in items])
        scored = [
            item.model_copy(
                update={
                    "score": float(score) + item.metadata_boost,
                    "rerank_score": float(score),
                    "ranking_score_type": "rerank_score_plus_metadata" if item.metadata_boost else "rerank_score",
                }
            )
            for item, score in zip(items, scores)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
