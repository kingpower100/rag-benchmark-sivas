import pytest

from src.pipeline1.chunking.fixed_token_chunker import FixedTokenChunker
from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.io.jsonl_reader import JsonlReader
from src.pipeline1.metadata import parse_treasury_filename
from src.pipeline1.orchestrator import _chunker_versions
from src.pipeline1.preflight import run_preflight_checks
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.utils.hashing import stable_hash_dict


def test_missing_context_id_fails_when_required(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text('{"document_id":"doc1","cleaned_context":"text"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required context_id"):
        JsonlReader.read_documents(str(path), require_context_id=True)


def test_document_text_field_uses_cleaned_context(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text('{"context_id":"ctx1","context":"raw text","cleaned_context":"clean text","file_name":"f"}\n', encoding="utf-8")

    docs = JsonlReader.read_documents(str(path), require_context_id=True, text_field="cleaned_context")

    assert docs[0].text == "clean text"
    assert docs[0].original_context_id == "ctx1"
    assert docs[0].metadata["file_name"] == "f"


def test_document_text_field_missing_fails_without_explicit_fallback(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text('{"context_id":"ctx1","context":"raw text"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="cleaned_context"):
        JsonlReader.read_documents(str(path), require_context_id=True, text_field="cleaned_context")


def test_txt_folder_reader_creates_officeqa_document_records(tmp_path):
    folder = tmp_path / "transformed"
    folder.mkdir()
    (folder / "treasury_bulletin_1944_01.txt").write_text("OfficeQA text", encoding="utf-8")
    (folder / "ignored.md").write_text("ignore", encoding="utf-8")

    docs = JsonlReader.read_txt_folder(str(folder), "*.txt")

    assert len(docs) == 1
    assert docs[0].document_id == "treasury_bulletin_1944_01.txt"
    assert docs[0].original_context_id == "treasury_bulletin_1944_01.txt"
    assert docs[0].text == "OfficeQA text"
    assert docs[0].metadata["file_name"] == "treasury_bulletin_1944_01.txt"
    assert docs[0].metadata["source_file"] == "treasury_bulletin_1944_01.txt"
    assert docs[0].metadata["source_id"] == "treasury_bulletin_1944_01"
    assert docs[0].metadata["year"] == 1944
    assert docs[0].metadata["month"] == "01"
    assert docs[0].metadata["report_year"] == 1944
    assert docs[0].metadata["source_dataset"] == "officeqa"


def test_parse_treasury_filename_extracts_metadata():
    metadata = parse_treasury_filename("treasury_bulletin_1941_01.txt")

    assert metadata["source_file"] == "treasury_bulletin_1941_01.txt"
    assert metadata["file_name"] == "treasury_bulletin_1941_01.txt"
    assert metadata["source_id"] == "treasury_bulletin_1941_01"
    assert metadata["year"] == 1941
    assert metadata["month"] == "01"
    assert metadata["report_year"] == 1941
    assert metadata["source_dataset"] == "officeqa"


def test_parse_treasury_filename_falls_back_for_unknown_name():
    metadata = parse_treasury_filename("other_report.txt")

    assert metadata["source_file"] == "other_report.txt"
    assert metadata["file_name"] == "other_report.txt"
    assert metadata["source_id"] == "other_report"
    assert metadata["year"] is None
    assert metadata["month"] is None


def test_pipeline1_rejects_answer_bearing_query_file(tmp_path, monkeypatch):
    (tmp_path / "documents.jsonl").write_text('{"context_id":"c1","cleaned_context":"text"}\n', encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text(
        '{"id":"q1","question":"Q?","program_answer":"100"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")

    errors = run_preflight_checks(_cfg(False, top_k=1, fetch_k=1), tmp_path)

    assert any("questions_only.jsonl-style" in error for error in errors)


def test_preflight_accepts_txt_folder_documents(tmp_path, monkeypatch):
    folder = tmp_path / "transformed"
    folder.mkdir()
    (folder / "treasury_bulletin_1944_01.txt").write_text("OfficeQA text", encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text('{"question_id":"q1","question":"Q?"}\n', encoding="utf-8")
    cfg = _cfg(False, top_k=1, fetch_k=1)
    cfg.data.documents_path = "transformed"
    cfg.data.documents_source_type = "txt_folder"
    cfg.data.documents_file_glob = "*.txt"
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")

    errors = run_preflight_checks(cfg, tmp_path)

    assert errors == []


def test_fixed_token_fallback_fails_unless_explicitly_allowed(monkeypatch):
    monkeypatch.setattr(FixedTokenChunker, "_load_encoding", staticmethod(lambda tokenizer_name: None))

    with pytest.raises(RuntimeError, match="allow_word_fallback=true"):
        FixedTokenChunker(10, 0, "missing_tokenizer", allow_word_fallback=False)

    chunker = FixedTokenChunker(10, 0, "missing_tokenizer", allow_word_fallback=True)
    chunks = chunker.chunk_documents([DocumentRecord(document_id="doc", original_context_id="ctx", text="one two")])

    assert chunker.encoding is None
    assert chunks[0].metadata["chunk_unit"] == "word_fallback"


def _cfg(reranker_enabled: bool, top_k: int, fetch_k: int) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": "dense", "top_k": top_k, "fetch_k": fetch_k},
            "reranker": {"enabled": reranker_enabled, "model_name": "fake" if reranker_enabled else None},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {"estimate_cost": False},
            "runtime": {"resume": False, "overwrite": True},
        }
    )


def test_fetch_k_must_exceed_top_k_when_reranking(tmp_path, monkeypatch):
    (tmp_path / "documents.jsonl").write_text('{"context_id":"c1","cleaned_context":"text"}\n', encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text('{"question_id":"q1","question":"Q?"}\n', encoding="utf-8")
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")

    errors = run_preflight_checks(_cfg(True, top_k=5, fetch_k=5), tmp_path)

    assert any("must be > final top_k" in error for error in errors)


def test_fetch_k_below_top_k_fails_without_reranking(tmp_path, monkeypatch):
    (tmp_path / "documents.jsonl").write_text('{"context_id":"c1","cleaned_context":"text"}\n', encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text('{"question_id":"q1","question":"Q?"}\n', encoding="utf-8")
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")

    errors = run_preflight_checks(_cfg(False, top_k=5, fetch_k=4), tmp_path)

    assert any("must be >= retrieval.top_k" in error for error in errors)


def test_experiment_id_must_match_experiment_config_filename(tmp_path):
    config_dir = tmp_path / "configs" / "pipeline1" / "experiments"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "actual_name.yaml"
    config_path.write_text(
        """
experiment:
  experiment_id: "different_name"
  output_dir: "runs"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must match config filename stem"):
        load_pipeline_config_payload(str(config_path))


def test_documents_config_block_normalizes_to_data_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment: {experiment_id: x, output_dir: out}
documents:
  path: data/raw/transformed
  source_type: txt_folder
  text_field: cleaned_context
  file_glob: "*.txt"
data:
  questions_path: data/raw/questions_only.jsonl
chunking: {strategy: fixed_word, chunk_size: 10, chunk_overlap: 0}
embedding: {provider: sentence_transformers, model_name: fake}
index: {type: faiss}
retrieval: {top_k: 1, fetch_k: 1}
reranker: {enabled: false}
generation: {provider: ollama, model_name: fake, system_prompt: s}
telemetry: {}
runtime: {}
""",
        encoding="utf-8",
    )

    cfg = PipelineConfig.from_yaml(str(config_path))

    assert cfg.data.documents_path == "data/raw/transformed"
    assert cfg.data.documents_source_type == "txt_folder"
    assert cfg.data.document_text_field == "cleaned_context"
    assert cfg.data.documents_file_glob == "*.txt"


def test_sentence_chunker_version_changes_chunk_cache_key(monkeypatch):
    cfg = PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "sentence", "chunk_size": 10, "chunk_overlap": 1},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 2},
            "reranker": {"enabled": False},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {"estimate_cost": False},
            "runtime": {"resume": False, "overwrite": True},
        }
    )
    first = stable_hash_dict({"documents_sha256": "doc_hash", "chunking": cfg.chunking.model_dump(), "chunker_versions": _chunker_versions(cfg)})
    monkeypatch.setattr("src.pipeline1.orchestrator.SENTENCE_CHUNKER_VERSION", "sentence_changed")
    second = stable_hash_dict({"documents_sha256": "doc_hash", "chunking": cfg.chunking.model_dump(), "chunker_versions": _chunker_versions(cfg)})

    assert first != second
