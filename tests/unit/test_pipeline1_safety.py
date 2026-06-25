import pytest
import sys
import types

import pytest
import torch

from src.pipeline1.chunking.fixed_token_chunker import FixedTokenChunker
from src.pipeline1.chunking.fixed_word_chunker import FixedWordChunker
from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.embedding.bge_encoder import BGEEncoder
from src.pipeline1.io.jsonl_reader import JsonlReader
from src.pipeline1.orchestrator import _chunker_versions
from src.pipeline1.preflight import run_preflight_checks
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.utils.hashing import stable_hash_dict


def test_missing_context_id_fails_when_required(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text('{"document_id":"doc1","cleaned_context":"text"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required context_id"):
        JsonlReader.read_documents(str(path), require_context_id=True, dataset_schema="generic")


def test_document_text_field_uses_cleaned_context(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text('{"context_id":"ctx1","context":"raw text","cleaned_context":"clean text","file_name":"f"}\n', encoding="utf-8")

    docs = JsonlReader.read_documents(
        str(path),
        require_context_id=True,
        text_field="cleaned_context",
        dataset_schema="generic",
    )

    assert docs[0].text == "clean text"
    assert docs[0].original_context_id == "ctx1"
    assert docs[0].metadata["file_name"] == "f"


def test_document_text_field_missing_fails_without_explicit_fallback(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text('{"context_id":"ctx1","context":"raw text"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="cleaned_context"):
        JsonlReader.read_documents(
            str(path),
            require_context_id=True,
            text_field="cleaned_context",
            dataset_schema="generic",
        )


def test_sivas_document_schema_maps_doc_key_and_text(tmp_path):
    path = tmp_path / "kb_documents_fixed.jsonl"
    path.write_text(
        '{"doc_id":1,"doc_key":"doc-a","doc_name":"a.md","kategorie":"IT","wissensart":"FAQ",'
        '"titel":"Title","quellpfad":"kb/a.md","sprache":"de","text":"SIVAS text"}\n',
        encoding="utf-8",
    )

    docs = JsonlReader.read_documents(str(path), dataset_schema="sivas")

    assert docs[0].document_id == "doc-a"
    assert docs[0].original_context_id == "doc-a"
    assert docs[0].text == "SIVAS text"
    assert docs[0].metadata["source_dataset"] == "sivas"
    assert docs[0].metadata["kategorie"] == "IT"


def test_sivas_document_reader_accepts_utf8_bom(tmp_path):
    path = tmp_path / "kb_documents_fixed.jsonl"
    path.write_text(
        '{"doc_id":1,"doc_key":"doc-a","doc_name":"a.md","kategorie":"IT","wissensart":"FAQ",'
        '"titel":"Title","quellpfad":"kb/a.md","sprache":"de","text":"SIVAS text"}\n',
        encoding="utf-8-sig",
    )

    docs = JsonlReader.read_documents(str(path), dataset_schema="sivas")

    assert docs[0].document_id == "doc-a"
    assert docs[0].text == "SIVAS text"


def test_sivas_metadata_and_stable_chunk_ids_survive_fixed_word_chunking(tmp_path):
    path = tmp_path / "kb_documents_fixed.jsonl"
    path.write_text(
        '{"doc_id":1,"doc_key":"doc-a","doc_name":"a.md","kategorie":"Einkauf","wissensart":"FAQ",'
        '"titel":"Bestellung","quellpfad":"kb/a.md","sprache":"de","text":"eins zwei drei vier fünf sechs"}\n',
        encoding="utf-8",
    )
    doc = JsonlReader.read_documents(str(path), dataset_schema="sivas")[0]

    chunks = FixedWordChunker(chunk_size=3, chunk_overlap=0).chunk_documents([doc])

    assert [chunk.chunk_id for chunk in chunks] == ["doc-a:chunk:0001", "doc-a:chunk:0002"]
    for chunk in chunks:
        assert chunk.document_id == "doc-a"
        assert chunk.original_context_id == "doc-a"
        assert chunk.metadata["doc_id"] == 1
        assert chunk.metadata["doc_key"] == "doc-a"
        assert chunk.metadata["doc_name"] == "a.md"
        assert chunk.metadata["kategorie"] == "Einkauf"
        assert chunk.metadata["wissensart"] == "FAQ"
        assert chunk.metadata["titel"] == "Bestellung"
        assert chunk.metadata["quellpfad"] == "kb/a.md"
        assert chunk.metadata["sprache"] == "de"


def test_sivas_document_schema_falls_back_to_doc_id(tmp_path):
    path = tmp_path / "kb_documents_fixed.jsonl"
    path.write_text('{"doc_id":1,"text":"SIVAS text"}\n', encoding="utf-8")

    docs = JsonlReader.read_documents(str(path), dataset_schema="sivas")

    assert docs[0].document_id == "1"
    assert docs[0].original_context_id == "1"


def test_sivas_document_missing_text_fails(tmp_path):
    path = tmp_path / "kb_documents_fixed.jsonl"
    path.write_text('{"doc_id":1,"doc_key":"doc-a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing non-empty text field 'text'"):
        JsonlReader.read_documents(str(path), dataset_schema="sivas")


def test_empty_document_file_fails(tmp_path):
    path = tmp_path / "kb_documents_fixed.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No documents loaded"):
        JsonlReader.read_documents(str(path), dataset_schema="sivas")


def test_sivas_question_schema_maps_frage(tmp_path):
    path = tmp_path / "questions_fixed.jsonl"
    path.write_text('{"question_id":"Q001","frage":"Wie geht das?"}\n', encoding="utf-8")

    queries = list(JsonlReader.iter_queries(str(path), "question_id", "question", dataset_schema="sivas"))

    assert queries[0].question_id == "Q001"
    assert queries[0].question == "Wie geht das?"


def test_sivas_question_reader_accepts_utf8_bom(tmp_path):
    path = tmp_path / "questions_fixed.jsonl"
    path.write_text('{"question_id":"Q001","frage":"Wie geht das?"}\n', encoding="utf-8-sig")

    queries = list(JsonlReader.iter_queries(str(path), "question_id", "question", dataset_schema="sivas"))

    assert queries[0].question_id == "Q001"
    assert queries[0].question == "Wie geht das?"


def test_sivas_question_missing_frage_fails(tmp_path):
    path = tmp_path / "questions_fixed.jsonl"
    path.write_text('{"question_id":"Q001","question":"wrong field"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing non-empty question field 'frage'"):
        list(JsonlReader.iter_queries(str(path), "question_id", "question", dataset_schema="sivas"))


def test_duplicate_question_id_fails(tmp_path):
    path = tmp_path / "questions_fixed.jsonl"
    path.write_text(
        '{"question_id":"Q001","frage":"Eine Frage?"}\n{"question_id":"Q001","frage":"Andere Frage?"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate question_id: Q001"):
        list(JsonlReader.iter_queries(str(path), "question_id", "question", dataset_schema="sivas"))


def test_empty_questions_file_fails(tmp_path):
    path = tmp_path / "questions_fixed.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No questions loaded"):
        list(JsonlReader.iter_queries(str(path), "question_id", "question", dataset_schema="sivas"))


def test_txt_folder_reader_creates_neutral_document_records(tmp_path):
    folder = tmp_path / "transformed"
    folder.mkdir()
    (folder / "sivas_manual_01.txt").write_text("SIVAS text", encoding="utf-8")
    (folder / "ignored.md").write_text("ignore", encoding="utf-8")

    docs = JsonlReader.read_txt_folder(str(folder), "*.txt")

    assert len(docs) == 1
    assert docs[0].document_id == "sivas_manual_01.txt"
    assert docs[0].original_context_id == "sivas_manual_01.txt"
    assert docs[0].text == "SIVAS text"
    assert docs[0].metadata["file_name"] == "sivas_manual_01.txt"
    assert docs[0].metadata["source_file"] == "sivas_manual_01.txt"
    assert docs[0].metadata["source_id"] == "sivas_manual_01"
    assert docs[0].metadata["source_dataset"] is None


def test_txt_folder_reader_recurses_and_preserves_relative_paths(tmp_path):
    folder = tmp_path / "transformed"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (nested / "sivas_manual_02.txt").write_text("Nested SIVAS text", encoding="utf-8")

    docs = JsonlReader.read_txt_folder(str(folder), "*.txt", recursive=True)

    assert len(docs) == 1
    assert docs[0].document_id == "nested/sivas_manual_02.txt"
    assert docs[0].original_context_id == "nested/sivas_manual_02.txt"
    assert docs[0].metadata["file_name"] == "sivas_manual_02.txt"
    assert docs[0].metadata["source_file"] == "nested/sivas_manual_02.txt"
    assert docs[0].metadata["source_path"] == "nested/sivas_manual_02.txt"


def test_pipeline1_rejects_answer_bearing_query_file(tmp_path, monkeypatch):
    (tmp_path / "documents.jsonl").write_text(
        '{"doc_key":"c1","doc_name":"c1.md","text":"text","kategorie":"ERP"}\n',
        encoding="utf-8",
    )
    (tmp_path / "questions.jsonl").write_text(
        '{"question_id":"q1","frage":"Q?","program_answer":"100"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")

    errors = run_preflight_checks(_cfg(False, top_k=1, fetch_k=1), tmp_path)

    assert any("Pipeline 1 query file must contain questions only" in error for error in errors)


def test_preflight_accepts_txt_folder_documents(tmp_path, monkeypatch):
    folder = tmp_path / "transformed"
    folder.mkdir()
    (folder / "sivas_manual_01.txt").write_text("SIVAS text", encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text('{"question_id":"q1","frage":"Q?"}\n', encoding="utf-8")
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


def test_require_cuda_true_without_cuda_fails_preflight(tmp_path, monkeypatch):
    (tmp_path / "documents.jsonl").write_text('{"context_id":"c1","cleaned_context":"text"}\n', encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text('{"question_id":"q1","question":"Q?"}\n', encoding="utf-8")
    cfg = _cfg(False, top_k=1, fetch_k=2)
    cfg.embedding.require_cuda = True
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    errors = run_preflight_checks(cfg, tmp_path)

    assert any("embedding.device is cuda or embedding.require_cuda=true" in error for error in errors)


def test_bge_encoder_propagates_requested_device(monkeypatch):
    captured = {}

    class FakeTensor:
        def __init__(self, device: str) -> None:
            self.device = torch.device(device)

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            self.device = torch.device("cuda:0" if str(device).startswith("cuda") else "cpu")
            self._target_device = self.device

        def encode(self, texts, convert_to_tensor=False, show_progress_bar=False, batch_size=None, normalize_embeddings=False):
            if convert_to_tensor:
                return FakeTensor("cuda:0" if str(captured["device"]).startswith("cuda") else "cpu")
            return __import__("numpy").zeros((len(texts), 3), dtype="float32")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    encoder = BGEEncoder("fake-model", device="cuda", require_cuda=False)

    assert captured["device"] == "cuda"
    assert str(encoder.requested_device) == "cuda"
    assert str(encoder.runtime_device).startswith("cuda")
    assert str(encoder.embedding_tensor_device).startswith("cuda")


def test_reranker_propagates_requested_device(monkeypatch):
    from src.pipeline1.retrieval.cross_encoder_reranker import CrossEncoderReranker
    from src.pipeline1.schemas.retrieval import RetrievalItem

    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            self.device = torch.device("cuda:0" if str(device).startswith("cuda") else "cpu")
            self._target_device = self.device

        def predict(self, pairs):
            return [0.5 for _ in pairs]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = CrossEncoderReranker("fake-reranker", device="cuda")
    items = [RetrievalItem(chunk_id="c1", original_context_id="c1", text="text", score=0.1, chunk_unit=None, metadata={})]
    ranked = reranker.rerank("question", items, 1)

    assert captured["device"] == "cuda"
    assert str(reranker.requested_device) == "cuda"
    assert str(reranker.runtime_device).startswith("cuda")
    assert ranked[0].chunk_id == "c1"


def test_cuda_requested_but_cpu_runtime_warns(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            self.device = torch.device("cpu")
            self._target_device = self.device

        def encode(self, texts, convert_to_tensor=False, show_progress_bar=False, batch_size=None, normalize_embeddings=False):
            if convert_to_tensor:
                return type("FakeTensor", (), {"device": torch.device("cpu")})()
            return __import__("numpy").zeros((len(texts), 3), dtype="float32")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.warns(RuntimeWarning, match="requested device='cuda'"):
        encoder = BGEEncoder("fake-model", device="cuda", require_cuda=False)

    assert str(encoder.runtime_device) == "cpu"


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
  recursive: true
data:
  questions_path: data/raw/questions_fixed.jsonl
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
    assert cfg.data.documents_recursive is True


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
