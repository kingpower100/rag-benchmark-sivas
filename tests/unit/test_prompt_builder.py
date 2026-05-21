from types import SimpleNamespace

from src.pipeline1.generation.prompt_builder import PromptBudget, build_prompt_with_stats
from src.pipeline1.schemas.output_record import OutputRecord


def _item(text: str):
    return SimpleNamespace(text=text, metadata={})


def test_oversized_chunk_is_truncated_and_stats_reported():
    prompt, stats = build_prompt_with_stats(
        "Answer from context.",
        "Q?",
        [_item("word " * 100)],
        budget=PromptBudget(max_prompt_tokens=200, max_context_tokens=20, max_chunk_tokens=10, max_context_chars=1000, max_chunk_chars=1000),
    )

    assert stats["chunks_truncated"] == 1
    assert stats["chunks_after"] == 1
    assert stats["context_tokens_after"] <= 10
    assert "word" in prompt


def test_lower_ranked_chunks_are_dropped_first():
    _, stats = build_prompt_with_stats(
        "Use context.",
        "Q?",
        [_item("first " * 5), _item("second " * 5), _item("third " * 5)],
        budget=PromptBudget(max_prompt_tokens=200, max_context_tokens=10, max_chunk_tokens=10, max_context_chars=1000, max_chunk_chars=1000),
    )

    assert stats["chunks_after"] == 2
    assert stats["chunks_dropped"] == 1


def test_prompt_stays_under_token_budget_with_word_fallback():
    prompt, stats = build_prompt_with_stats(
        "Use context.",
        "Q?",
        [_item("x " * 100)],
        budget=PromptBudget(max_prompt_tokens=12, max_context_tokens=100, max_chunk_tokens=100, max_context_chars=1000, max_chunk_chars=1000, tokenizer_name="missing"),
    )

    assert len(prompt.split()) <= 12
    assert stats["prompt_tokens"] <= 12


def test_prompt_stats_are_saved_on_output_record():
    record = OutputRecord(
        experiment_id="exp",
        question_id="q1",
        question="Q?",
        generated_answer="1",
        retrieved_chunk_ids=["c1"],
        retrieved_original_context_ids=["d1"],
        retrieved_context_texts=["ctx"],
        retrieval_scores=[1.0],
        top_k=1,
        chunking_strategy="fixed_word",
        chunk_size=10,
        chunk_overlap=0,
        embedding_model="e",
        retriever_type="dense",
        reranker_used=False,
        llm_model="m",
        retrieval_time_ms=1,
        generation_time_ms=1,
        total_latency_ms=2,
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        prompt_stats={"prompt_chars": 12, "prompt_tokens": 3, "chunks_dropped": 1},
    )

    assert record.prompt_chars == 12
    assert record.prompt_tokens == 3
    assert record.chunks_dropped == 1
