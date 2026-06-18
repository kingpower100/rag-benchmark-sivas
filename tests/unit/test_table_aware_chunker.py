from src.pipeline1.chunking.fixed_word_chunker import FixedWordChunker
from src.pipeline1.chunking.table_aware_chunker import TableAwareChunker
from src.pipeline1.schemas.document import DocumentRecord


def test_table_aware_chunker_does_not_subclass_fixed_word_chunker():
    assert not issubclass(TableAwareChunker, FixedWordChunker)


def test_table_aware_overlap_uses_word_budget_not_block_count():
    doc = DocumentRecord(
        document_id="doc",
        original_context_id="doc",
        text="\n\n".join(
            [
                "one two three four",
                "five six seven eight",
                "nine ten eleven twelve",
                "thirteen fourteen fifteen sixteen",
            ]
        ),
    )

    chunks = TableAwareChunker(chunk_size=8, chunk_overlap=4).chunk_documents([doc])

    assert len(chunks) == 3
    assert chunks[0].text == "one two three four\n\nfive six seven eight"
    assert chunks[1].text == "five six seven eight\n\nnine ten eleven twelve"
    assert chunks[2].text == "nine ten eleven twelve\n\nthirteen fourteen fifteen sixteen"


def test_table_aware_chunker_keeps_markdown_table_intact():
    text = """Amounts in millions.

| Year | Revenue | Costs |
| --- | ---: | ---: |
| 2020 | 100 | 80 |
| 2021 | 120 | 70 |

Net income improved after cost reductions."""
    doc = DocumentRecord(
        document_id="doc1",
        original_context_id="ctx1",
        text=text,
        metadata={
            "file_name": "sivas_manual_01.md",
            "source_file": "docs/sivas_manual_01.md",
            "source_id": "doc-key-1",
            "doc_key": "doc-key-1",
            "kategorie": "ERP",
        },
    )

    chunks = TableAwareChunker(chunk_size=12, chunk_overlap=0).chunk_documents([doc])
    table_chunks = [chunk for chunk in chunks if "| Year | Revenue | Costs |" in chunk.text]

    assert len(table_chunks) == 1
    table_chunk = table_chunks[0]
    assert "Amounts in millions." in table_chunk.text
    assert "| 2020 | 100 | 80 |" in table_chunk.text
    assert "| 2021 | 120 | 70 |" in table_chunk.text
    assert table_chunk.original_context_id == "ctx1"
    assert table_chunk.metadata["chunk_strategy"] == "table_aware"
    assert table_chunk.metadata["chunk_unit"] == "table_or_text_block"
    assert table_chunk.metadata["contains_table"] is True
    assert table_chunk.metadata["file_name"] == "sivas_manual_01.md"
    assert table_chunk.metadata["source_file"] == "docs/sivas_manual_01.md"
    assert table_chunk.metadata["source_id"] == "doc-key-1"
    assert table_chunk.metadata["doc_key"] == "doc-key-1"
    assert table_chunk.metadata["chunk_id"] == table_chunk.chunk_id


def test_table_aware_chunker_allows_oversized_tables_without_splitting_rows():
    text = """Before table.

| Metric | 2020 | 2021 | 2022 |
| --- | ---: | ---: | ---: |
| Revenue | 100 | 120 | 140 |
| Cost of revenue | 60 | 70 | 85 |
| Net income | 10 | 18 | 24 |

After table."""
    doc = DocumentRecord(document_id="doc2", original_context_id="ctx2", text=text)

    chunks = TableAwareChunker(chunk_size=5, chunk_overlap=0).chunk_documents([doc])
    table_chunks = [chunk for chunk in chunks if "| Metric | 2020 | 2021 | 2022 |" in chunk.text]

    assert len(table_chunks) == 1
    table_chunk = table_chunks[0]
    assert "| Revenue | 100 | 120 | 140 |" in table_chunk.text
    assert "| Cost of revenue | 60 | 70 | 85 |" in table_chunk.text
    assert "| Net income | 10 | 18 | 24 |" in table_chunk.text
    assert table_chunk.metadata["contains_table"] is True
    assert table_chunk.metadata["oversized_table"] is True


def test_table_aware_chunker_splits_huge_table_and_preserves_metadata():
    rows = ["| Col | Value |", "| --- | --- |"]
    rows.extend(f"| row{i} | {'x ' * 20} |" for i in range(30))
    doc = DocumentRecord(
        document_id="doc3",
        original_context_id="ctx3",
        text="\n".join(rows),
        metadata={"file_name": "source.txt", "source_file": "source.txt"},
    )

    chunks = TableAwareChunker(
        chunk_size=20,
        chunk_overlap=0,
        max_chunk_chars=300,
        max_chunk_tokens=80,
        oversized_chunk_policy="split",
    ).chunk_documents([doc])

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert all(len(chunk.text.split()) <= 80 for chunk in chunks)
    assert all(chunk.metadata["file_name"] == "source.txt" for chunk in chunks)
    assert all(chunk.metadata["source_file"] == "source.txt" for chunk in chunks)
    assert any(chunk.metadata["oversized_split"] is True for chunk in chunks)


def test_table_aware_chunker_version_marks_oversized_guard():
    from src.pipeline1.chunking.table_aware_chunker import TABLE_AWARE_CHUNKER_VERSION

    assert TABLE_AWARE_CHUNKER_VERSION == "table_aware_v3_oversized_guard"
