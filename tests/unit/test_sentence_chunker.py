from src.pipeline1.chunking.fixed_word_chunker import FixedWordChunker
from src.pipeline1.chunking.sentence_chunker import SentenceChunker
from src.pipeline1.schemas.document import DocumentRecord


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
