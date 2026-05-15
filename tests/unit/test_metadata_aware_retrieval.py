from src.pipeline1.chunking.fixed_word_chunker import FixedWordChunker
from src.pipeline1.generation.prompt_builder import build_prompt
from src.pipeline1.io.jsonl_reader import JsonlReader
from src.pipeline1.metadata import normalize_metadata, safe_int
from src.pipeline1.retrieval.dense_retriever import DenseRetriever
from src.pipeline1.retrieval.metadata import extract_query_metadata
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import MetadataBoostingConfig, MetadataFilteringConfig, PipelineConfig
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline2.metrics.retrieval_metrics import compute_metadata_match_metrics


class _Embedder:
    def encode_query(self, question: str):
        return [1.0]


class _Index:
    def __init__(self, scores, idxs):
        self.scores = scores
        self.idxs = idxs

    def search(self, query_vec, k):
        return self.scores[:k], self.idxs[:k]


def _chunk(chunk_id, company, year, symbol=""):
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=chunk_id,
        original_context_id=chunk_id,
        text=chunk_id,
        chunk_start=0,
        chunk_end=1,
        metadata={
            "company_name": company,
            "company_symbol": symbol or None,
            "report_year": year,
            "file_name": f"{company}_{year}.pdf",
            "source_dataset": "finqa",
        },
    )


def test_metadata_normalization_preserves_malformed_values_without_silent_drop():
    normalized = normalize_metadata(
        {
            "company_name": "  Apple   Inc. ",
            "company_symbol": " AAPL ",
            "report_year": "2021",
            "page_number": "not-a-page",
        },
        "ctx-1",
    )

    assert normalized["company_name"] == "Apple Inc."
    assert normalized["company_symbol"] == "AAPL"
    assert normalized["report_year"] == 2021
    assert normalized["page_number"] is None
    assert normalized["original_context_id"] == "ctx-1"
    assert safe_int("20.5") is None


def test_chunking_propagates_canonical_metadata():
    doc = DocumentRecord(
        document_id="doc-1",
        original_context_id="ctx-1",
        text="one two three",
        metadata={"company_name": "Apple", "report_year": 2021, "page_number": 7, "source_dataset": "finqa"},
    )

    chunk = FixedWordChunker(chunk_size=10, chunk_overlap=0).chunk_documents([doc])[0]

    assert chunk.metadata["company_name"] == "Apple"
    assert chunk.metadata["report_year"] == 2021
    assert chunk.metadata["page_number"] == 7
    assert chunk.metadata["source_dataset"] == "finqa"
    assert chunk.metadata["original_context_id"] == "ctx-1"


def test_document_loading_normalizes_metadata():
    path = Path("tests/unit/_tmp_docs_metadata.jsonl")
    try:
        path.write_text(
            '{"context_id":"ctx-1","cleaned_context":"text","company_name":" Apple ","report_year":"2021","page_number":"7"}\n',
            encoding="utf-8",
        )

        doc = JsonlReader.read_documents(str(path), require_context_id=True)[0]

        assert doc.metadata["company_name"] == "Apple"
        assert doc.metadata["report_year"] == 2021
        assert doc.metadata["page_number"] == 7
        assert doc.metadata["original_context_id"] == "ctx-1"
    finally:
        path.unlink(missing_ok=True)


def test_query_metadata_extraction_uses_known_names_symbols_years_and_quarters():
    metadata = [{"company_name": "Apple", "company_symbol": "AAPL", "file_name": "apple_2021.pdf"}]
    extracted = extract_query_metadata("What was Apple (AAPL) revenue in Q2 2021?", metadata)

    assert extracted.company_names == frozenset({"apple"})
    assert extracted.company_symbols == frozenset({"aapl"})
    assert extracted.years == frozenset({2021})
    assert extracted.report_periods == frozenset({"q2"})


def test_metadata_boosting_changes_dense_ranking():
    chunks = [_chunk("microsoft", "Microsoft", 2021, "MSFT"), _chunk("apple", "Apple", 2021, "AAPL")]
    retriever = DenseRetriever(
        _Embedder(),
        _Index([0.9, 0.8], [0, 1]),
        chunks,
        fetch_k=2,
        metadata_boosting=MetadataBoostingConfig(enabled=True, company_weight=0.3, year_weight=0.0, symbol_weight=0.0),
        metadata_filtering=MetadataFilteringConfig(),
    )

    items = retriever.retrieve("What did Apple report?", top_k=2)

    assert [item.chunk_id for item in items] == ["apple", "microsoft"]
    assert items[0].metadata_boost == 0.3


def test_metadata_filtering_falls_back_if_no_candidate_matches():
    chunks = [_chunk("microsoft", "Microsoft", 2021), _chunk("apple", "Apple", 2020)]
    retriever = DenseRetriever(
        _Embedder(),
        _Index([0.9, 0.8], [0, 1]),
        chunks,
        fetch_k=2,
        metadata_boosting=MetadataBoostingConfig(),
        metadata_filtering=MetadataFilteringConfig(enabled=True, strict=True),
    )

    items = retriever.retrieve("What did Apple report in 2021?", top_k=2)

    assert [item.chunk_id for item in items] == ["microsoft", "apple"]


def test_prompt_can_include_metadata_header_without_changing_default():
    item = RetrievalItem(
        chunk_id="c1",
        original_context_id="ctx1",
        text="Revenue was 10.",
        score=1.0,
        dense_score=1.0,
        metadata={"company_name": "Apple", "report_year": 2021, "page_number": 32},
    )

    plain = build_prompt("System", "Question?", [item])
    enriched = build_prompt("System", "Question?", [item], include_metadata_headers=True)

    assert "[Company:" not in plain
    assert "[Company: Apple | Year: 2021 | Page: 32]" in enriched


def test_prompt_builder_renders_explicit_context_and_question_placeholders():
    item = RetrievalItem(
        chunk_id="c1",
        original_context_id="ctx1",
        text="Revenue was 10.",
        score=1.0,
        dense_score=1.0,
    )

    prompt = build_prompt("Context:\n{context}\n\nQuestion:\n{question}", "What was revenue?", [item])

    assert "{context}" not in prompt
    assert "{question}" not in prompt
    assert "[1] Revenue was 10." in prompt
    assert "What was revenue?" in prompt


def test_metadata_match_metrics_use_explicit_query_metadata_payload():
    metrics = compute_metadata_match_metrics(
        "What did Apple report in 2021?",
        [{"company_name": "Apple", "report_year": 2021}, {"company_name": "Microsoft", "report_year": 2020}],
        {"company_names": ["apple"], "years": [2021]},
    )

    assert metrics == {"metadata_match_rate": 0.5, "company_match_rate": 0.5, "year_match_rate": 0.5}


def test_backward_compatible_config_defaults_metadata_features_off():
    path = Path("tests/unit/_tmp_metadata_config.yaml")
    try:
        path.write_text(
            """
experiment: {experiment_id: x, output_dir: out}
data: {documents_path: docs, questions_path: qs}
chunking: {strategy: fixed_word, chunk_size: 10, chunk_overlap: 0}
embedding: {provider: sentence_transformers, model_name: m}
index: {type: faiss}
retrieval: {top_k: 1, fetch_k: 1}
reranker: {enabled: false}
generation: {provider: ollama, model_name: m, system_prompt: s}
telemetry: {}
runtime: {}
""",
            encoding="utf-8",
        )

        cfg = PipelineConfig.from_yaml(str(path))

        assert cfg.retrieval.metadata_boosting.enabled is False
        assert cfg.retrieval.metadata_filtering.enabled is False
        assert cfg.generation.include_metadata_headers is False
    finally:
        path.unlink(missing_ok=True)
from pathlib import Path
