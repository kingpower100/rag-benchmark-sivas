from src.pipeline1.orchestrator import _duplicate_rate, dedupe_retrieval_by_chunk_id, retrieve_top_k_unique_contexts
from src.pipeline1.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.pipeline1.schemas.retrieval import RetrievalItem


def test_dedup_by_chunk_id_preserves_multiple_chunks_from_same_context():
    items = [
        RetrievalItem(chunk_id="c1_a", original_context_id="ctx1", text="one", score=0.9, dense_score=0.9),
        RetrievalItem(chunk_id="c1_b", original_context_id="ctx1", text="same source different chunk", score=0.8, dense_score=0.8),
        RetrievalItem(chunk_id="c2", original_context_id="ctx2", text="two", score=0.7, dense_score=0.7),
        RetrievalItem(chunk_id="c3", original_context_id="ctx3", text="three", score=0.6, dense_score=0.6),
    ]

    deduped = dedupe_retrieval_by_chunk_id(items, top_k=2)

    assert [item.chunk_id for item in deduped] == ["c1_a", "c1_b"]
    assert [item.original_context_id for item in deduped] == ["ctx1", "ctx1"]
    assert [item.text for item in deduped] == ["one", "same source different chunk"]
    assert [item.score for item in deduped] == [0.9, 0.8]


def test_retrieve_top_k_unique_does_not_backfill_beyond_fetch_k():
    class FakeRetriever:
        def __init__(self):
            self.requested = []

        def retrieve(self, question, top_k):
            self.requested.append(top_k)
            items = [
                RetrievalItem(chunk_id="c1_a", original_context_id="ctx1", text="one", score=0.9, dense_score=0.9),
                RetrievalItem(chunk_id="c1_b", original_context_id="ctx1", text="one duplicate", score=0.8, dense_score=0.8),
                RetrievalItem(chunk_id="c2", original_context_id="ctx2", text="two", score=0.7, dense_score=0.7),
                RetrievalItem(chunk_id="c3", original_context_id="ctx3", text="three", score=0.6, dense_score=0.6),
            ]
            return items[:top_k]

    retriever = FakeRetriever()

    raw, retrieved, warnings, _ = retrieve_top_k_unique_contexts(
        "Q?",
        retriever,
        reranker=None,
        top_k=3,
        fetch_k=2,
        max_candidates=4,
    )

    assert retriever.requested == [2]
    assert [item.chunk_id for item in raw] == ["c1_a", "c1_b"]
    assert [item.chunk_id for item in retrieved] == ["c1_a", "c1_b"]
    assert warnings == ["Only 2 unique chunks were available after deduplication within fetch_k=2; requested top_k=3."]


def test_no_reranker_still_fetches_raw_fetch_k_candidates():
    class FakeRetriever:
        def __init__(self):
            self.requested = []

        def retrieve(self, question, top_k):
            self.requested.append(top_k)
            return [
                RetrievalItem(chunk_id=f"c{idx}", original_context_id=f"ctx{idx}", text=str(idx), score=1.0 / idx)
                for idx in range(1, top_k + 1)
            ]

    retriever = FakeRetriever()

    raw, retrieved, warnings, reranker_used = retrieve_top_k_unique_contexts(
        "Q?",
        retriever,
        reranker=None,
        top_k=10,
        fetch_k=50,
        max_candidates=100,
    )

    assert retriever.requested == [50]
    assert len(raw) == 50
    assert len(retrieved) == 10
    assert warnings == []
    assert reranker_used is False


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


def test_raw_duplicate_rate_helper_measures_pre_dedup_redundancy():
    assert _duplicate_rate(["ctx1", "ctx1", "ctx2"]) == 1 / 3
    assert _duplicate_rate([]) == 0.0
