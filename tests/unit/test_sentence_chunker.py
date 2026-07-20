from src.pipeline1.chunking.fixed_word_chunker import FixedWordChunker
from src.pipeline1.chunking.sentence_chunker import SentenceChunker
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.document import DocumentRecord
import pytest


def test_sentence_chunker_does_not_subclass_fixed_word_chunker():
    assert not issubclass(SentenceChunker, FixedWordChunker)


def test_sentence_chunker_respects_sentence_boundaries_and_overlap():
    doc = DocumentRecord(
        document_id="doc1",
        original_context_id="ctx1",
        text="Revenue increased in 2020. Costs declined in 2021. Net income improved. Cash flow remained stable.",
        metadata={"subset": "FinQA", "source_split": "train", "file_name": "report.pdf"},
    )

    chunks = SentenceChunker(chunk_size=8, chunk_overlap=1).chunk_documents([doc])

    assert [chunk.text for chunk in chunks] == [
        "Revenue increased in 2020. Costs declined in 2021.",
        "Costs declined in 2021. Net income improved.",
        "Net income improved. Cash flow remained stable.",
    ]
    assert chunks[0].text.endswith("2021.")
    assert chunks[1].text.startswith("Costs declined in 2021.")
    assert chunks[2].text.startswith("Net income improved.")
    assert all(chunk.metadata["chunk_unit"] == "sentence" for chunk in chunks)


def test_sentence_chunker_preserves_original_context_id_and_metadata():
    doc = DocumentRecord(
        document_id="doc2",
        original_context_id="ctx2",
        text="First sentence. Second sentence.",
        metadata={"subset": "ConvFinQA", "split": "test", "source_file": "source.jsonl"},
    )

    chunk = SentenceChunker(chunk_size=20, chunk_overlap=0).chunk_documents([doc])[0]

    assert chunk.document_id == "doc2"
    assert chunk.original_context_id == "ctx2"
    assert chunk.metadata["doc_id"] == "doc2"
    assert chunk.metadata["original_context_id"] == "ctx2"
    assert chunk.metadata["subset"] == "ConvFinQA"
    assert chunk.metadata["split"] == "test"
    assert chunk.metadata["source_file"] == "source.jsonl"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["chunk_strategy"] == "sentence"
    assert chunk.metadata["chunk_unit"] == "sentence"
    assert chunk.metadata["chunk_id"] == chunk.chunk_id


def test_sentence_chunker_marks_fallback_when_no_boundaries_found():
    doc = DocumentRecord(document_id="doc3", original_context_id="ctx3", text="No sentence boundary here")

    chunks = SentenceChunker(chunk_size=5, chunk_overlap=1).chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].text == "No sentence boundary here"
    assert chunks[0].metadata["chunk_unit"] == "sentence_fallback"


def test_sentence_chunk_ids_include_strategy_config():
    doc = DocumentRecord(document_id="doc4", original_context_id="ctx4", text="First sentence. Second sentence.")

    first = SentenceChunker(chunk_size=20, chunk_overlap=0).chunk_documents([doc])[0]
    second = SentenceChunker(chunk_size=20, chunk_overlap=1).chunk_documents([doc])[0]

    assert first.chunk_id != second.chunk_id


def test_sentence_chunker_explicit_word_overlap_uses_word_budget():
    doc = DocumentRecord(
        document_id="doc5",
        original_context_id="ctx5",
        text="One two three. Four five six. Seven eight nine. Ten eleven twelve.",
    )

    chunks = SentenceChunker(
        chunk_size=6,
        chunk_overlap=3,
        chunk_size_unit="words",
        chunk_overlap_unit="words",
    ).chunk_documents([doc])

    assert [chunk.text for chunk in chunks] == [
        "One two three. Four five six.",
        "Four five six. Seven eight nine.",
        "Seven eight nine. Ten eleven twelve.",
    ]
    assert chunks[0].metadata["chunk_size_unit"] == "words"
    assert chunks[0].metadata["chunk_overlap_unit"] == "words"


def test_sentence_chunker_explicit_token_units_are_recorded_and_progress():
    doc = DocumentRecord(
        document_id="doc6",
        original_context_id="ctx6",
        text="Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu.",
    )

    chunks = SentenceChunker(
        chunk_size=8,
        chunk_overlap=3,
        chunk_size_unit="tokens",
        chunk_overlap_unit="tokens",
        tokenizer_name="cl100k_base",
    ).chunk_documents([doc])

    assert chunks
    assert all(chunk.metadata["chunk_size_unit"] == "tokens" for chunk in chunks)
    assert all(chunk.metadata["chunk_overlap_unit"] == "tokens" for chunk in chunks)
    assert len({chunk.text for chunk in chunks}) == len(chunks)


def test_sentence_chunker_zero_overlap_makes_forward_progress():
    doc = DocumentRecord(
        document_id="doc7",
        original_context_id="ctx7",
        text="One two three. Four five six. Seven eight nine.",
    )

    chunks = SentenceChunker(
        chunk_size=3,
        chunk_overlap=0,
        chunk_size_unit="words",
        chunk_overlap_unit="words",
    ).chunk_documents([doc])

    assert [chunk.text for chunk in chunks] == ["One two three.", "Four five six.", "Seven eight nine."]


def test_sentence_chunker_overlap_chunk_size_minus_one_does_not_duplicate_full_chunks():
    doc = DocumentRecord(
        document_id="doc8",
        original_context_id="ctx8",
        text="A b. C d. E f. G h.",
    )

    chunks = SentenceChunker(
        chunk_size=4,
        chunk_overlap=3,
        chunk_size_unit="words",
        chunk_overlap_unit="words",
    ).chunk_documents([doc])

    assert len(chunks) > 1
    assert len({chunk.text for chunk in chunks}) == len(chunks)


def test_sentence_chunker_rejects_overlap_greater_or_equal_to_size():
    with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
        SentenceChunker(
            chunk_size=4,
            chunk_overlap=4,
            chunk_size_unit="words",
            chunk_overlap_unit="words",
        )


def test_sentence_chunker_emits_oversized_single_sentence():
    doc = DocumentRecord(
        document_id="doc9",
        original_context_id="ctx9",
        text="one two three four five six seven eight nine ten.",
    )

    chunks = SentenceChunker(
        chunk_size=3,
        chunk_overlap=1,
        chunk_size_unit="words",
        chunk_overlap_unit="words",
    ).chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].text == doc.text


def test_sentence_chunker_empty_document_returns_no_chunks():
    doc = DocumentRecord(document_id="doc10", original_context_id="ctx10", text="")

    assert SentenceChunker(
        chunk_size=5,
        chunk_overlap=1,
        chunk_size_unit="words",
        chunk_overlap_unit="words",
    ).chunk_documents([doc]) == []


def test_sentence_chunker_keeps_common_abbreviation_with_sentence():
    doc = DocumentRecord(
        document_id="doc11",
        original_context_id="ctx11",
        text="Dr. Mueller prueft den Auftrag. Danach wird geliefert.",
    )

    chunks = SentenceChunker(
        chunk_size=20,
        chunk_overlap=0,
        chunk_size_unit="words",
        chunk_overlap_unit="words",
    ).chunk_documents([doc])

    assert chunks[0].text.startswith("Dr. Mueller")


def test_sentence_config_rejects_overlap_equal_to_size():
    with pytest.raises(ValueError, match="chunk_overlap must be < chunking.chunk_size"):
        PipelineConfig.model_validate(
            {
                "experiment": {"experiment_id": "exp", "output_dir": "runs"},
                "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
                "chunking": {
                    "strategy": "sentence",
                    "chunk_size": 5,
                    "chunk_size_unit": "words",
                    "chunk_overlap": 5,
                    "chunk_overlap_unit": "words",
                },
                "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                "index": {"type": "faiss", "metric": "cosine"},
                "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
                "reranker": {"enabled": False},
                "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
                "telemetry": {},
                "runtime": {},
            }
        )
