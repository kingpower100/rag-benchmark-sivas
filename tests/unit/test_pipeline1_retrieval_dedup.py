from src.pipeline1.orchestrator import dedupe_retrieval_by_original_context_id, retrieve_top_k_unique_contexts
from src.pipeline1.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.pipeline1.schemas.retrieval import RetrievalItem


def test_dedup_by_original_context_id_preserves_alignment():
    items = [
        RetrievalItem(chunk_id="c1_a", original_context_id="ctx1", text="one", score=0.9, dense_score=0.9),
        RetrievalItem(chunk_id="c1_b", original_context_id="ctx1", text="one duplicate", score=0.8, dense_score=0.8),
        RetrievalItem(chunk_id="c2", original_context_id="ctx2", text="two", score=0.7, dense_score=0.7),
        RetrievalItem(chunk_id="c3", original_context_id="ctx3", text="three", score=0.6, dense_score=0.6),
    ]

    deduped = dedupe_retrieval_by_original_context_id(items, top_k=2)

    assert [item.chunk_id for item in deduped] == ["c1_a", "c2"]
    assert [item.original_context_id for item in deduped] == ["ctx1", "ctx2"]
    assert [item.text for item in deduped] == ["one", "two"]
    assert [item.score for item in deduped] == [0.9, 0.7]


def test_retrieve_top_k_unique_backfills_from_additional_candidates():
    class FakeRetriever:
        def retrieve(self, question, top_k):
            items = [
                RetrievalItem(chunk_id="c1_a", original_context_id="ctx1", text="one", score=0.9, dense_score=0.9),
                RetrievalItem(chunk_id="c1_b", original_context_id="ctx1", text="one duplicate", score=0.8, dense_score=0.8),
                RetrievalItem(chunk_id="c2", original_context_id="ctx2", text="two", score=0.7, dense_score=0.7),
                RetrievalItem(chunk_id="c3", original_context_id="ctx3", text="three", score=0.6, dense_score=0.6),
            ]
            return items[:top_k]

    _, retrieved, warnings, _ = retrieve_top_k_unique_contexts(
        "Q?",
        FakeRetriever(),
        reranker=None,
        top_k=3,
        fetch_k=2,
        max_candidates=4,
    )

    assert [item.original_context_id for item in retrieved] == ["ctx1", "ctx2", "ctx3"]
    assert warnings == []


def test_reranker_preserves_dense_score_and_adds_rerank_score():
    class FakeModel:
        def predict(self, pairs):
            return [0.1, 0.9]

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = FakeModel()
    items = [
        RetrievalItem(chunk_id="c1", original_context_id="ctx1", text="one", score=0.8, dense_score=0.8),
        RetrievalItem(chunk_id="c2", original_context_id="ctx2", text="two", score=0.7, dense_score=0.7),
    ]

    reranked = reranker.rerank("Q?", items, top_k=2)

    assert [item.chunk_id for item in reranked] == ["c2", "c1"]
    assert [item.dense_score for item in reranked] == [0.7, 0.8]
    assert [item.rerank_score for item in reranked] == [0.9, 0.1]
    assert all(item.ranking_score_type == "rerank_score" for item in reranked)
