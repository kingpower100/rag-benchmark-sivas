from src.pipeline1.chunking.fixed_word_chunker import FixedWordChunker
from src.pipeline1.chunking.table_aware_chunker import TableAwareChunker
from src.pipeline1.schemas.document import DocumentRecord


def test_table_aware_chunker_does_not_subclass_fixed_word_chunker():
    assert not issubclass(TableAwareChunker, FixedWordChunker)


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
            "file_name": "treasury_bulletin_1941_01.txt",
            "source_file": "treasury_bulletin_1941_01.txt",
            "source_id": "treasury_bulletin_1941_01",
            "year": 1941,
            "month": "01",
            "subset": "finance",
            "company_name": "ACME",
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
    assert table_chunk.metadata["file_name"] == "treasury_bulletin_1941_01.txt"
    assert table_chunk.metadata["source_file"] == "treasury_bulletin_1941_01.txt"
    assert table_chunk.metadata["source_id"] == "treasury_bulletin_1941_01"
    assert table_chunk.metadata["year"] == 1941
    assert table_chunk.metadata["month"] == "01"
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
