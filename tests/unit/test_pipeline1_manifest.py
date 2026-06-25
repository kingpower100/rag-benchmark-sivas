import json
from pathlib import Path

import numpy as np

from src.pipeline1.generation.base import GenerationResult
from src.pipeline1.orchestrator import run_pipeline


class _FakeEmbedder:
    def encode_texts(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype="float32")

    def encode_query(self, text):
        return np.ones(2, dtype="float32")


class _FakeIndex:
    def build(self, embeddings):
        self.embeddings = embeddings

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake", encoding="utf-8")

    def load(self, path):
        pass

    def search(self, query_embedding, top_k):
        return np.array([1.0, 0.9], dtype="float32"), np.array([0, 1], dtype="int64")


class _FakeGenerator:
    def generate(self, prompt):
        if "fixed preprocessing model" in prompt:
            return GenerationResult(
                answer='{"cleaned_question":"Q?","detected_category":""}',
                input_tokens=5,
                output_tokens=5,
            )
        return GenerationResult(answer="1", input_tokens=5, output_tokens=1)


def test_pipeline1_manifest_contains_reproducibility_fields(tmp_path, monkeypatch):
    project_root = tmp_path
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "kb_documents_fixed.jsonl").write_text(
        (
            '{"doc_key":"doc-1","doc_name":"sivas_1.md","text":"alpha",'
            '"kategorie":"ERP","wissensart":"howto","titel":"Alpha","quellpfad":"docs/sivas_1.md"}\n'
            '{"doc_key":"doc-2","doc_name":"sivas_2.md","text":"beta",'
            '"kategorie":"ERP","wissensart":"howto","titel":"Beta","quellpfad":"docs/sivas_2.md"}\n'
        ),
        encoding="utf-8",
    )
    (data_dir / "questions_fixed.jsonl").write_text('{"question_id":"q1","frage":"Q?"}\n', encoding="utf-8")
    cfg_path = project_root / "config.yaml"
    cfg_path.write_text(
        """
experiment:
  experiment_id: "test_exp"
  random_seed: 42
  output_dir: "runs"
data:
  dataset_schema: "sivas"
  documents_path: "data/raw/kb_documents_fixed.jsonl"
  questions_path: "data/raw/questions_fixed.jsonl"
  question_id_field: "question_id"
  question_field: "frage"
  document_text_field: "text"
  allow_document_text_fallback: false
  allow_unsafe_query_fields: false
chunking:
  strategy: "fixed_word"
  chunk_size: 10
  chunk_overlap: 0
  allow_word_fallback: false
embedding:
  provider: "sentence_transformers"
  model_name: "fake"
  normalize_embeddings: true
  batch_size: 2
  device: "cpu"
index:
  type: "faiss"
  metric: "cosine"
retrieval:
  retriever_type: "dense"
  top_k: 1
  fetch_k: 2
reranker:
  enabled: false
generation:
  provider: "ollama"
  model_name: "fake"
  base_url: "http://localhost:11434"
  temperature: 0.0
  max_tokens: 8
  timeout_s: 10
  system_prompt: "Use context."
telemetry:
  estimate_cost: true
runtime:
  save_csv: true
  log_level: "INFO"
  resume: false
  overwrite: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")
    monkeypatch.setattr("src.pipeline1.orchestrator._project_root", lambda: project_root)
    monkeypatch.setattr("src.pipeline1.orchestrator.build_embedder", lambda config: _FakeEmbedder())
    monkeypatch.setattr("src.pipeline1.orchestrator.build_index", lambda config: _FakeIndex())
    monkeypatch.setattr("src.pipeline1.orchestrator.build_generator", lambda config: _FakeGenerator())

    run_dir = run_pipeline(str(cfg_path))

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    alias_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert alias_manifest["run_id"] == "test_exp"
    assert manifest["run_id"] == "test_exp"
    assert manifest["config_path"] == str(cfg_path.resolve())
    assert manifest["config_hash"]
    assert manifest["machine"]["hostname"] is not None
    assert manifest["machine"]["python_version"]
    assert manifest["data_hashes"]["documents_sha256"]
    assert manifest["resolved_config"]["experiment"]["experiment_id"] == "test_exp"
    assert manifest["models"]["embedding_model"] == "fake"
    assert manifest["models"]["index_type"] == "faiss"
    assert manifest["models"]["reranker_enabled"] is False
    assert manifest["models"]["generator_model"] == "fake"
    assert manifest["cache_artifact_paths"]["chunks"].endswith(".jsonl")
    assert manifest["cache_artifact_paths"]["embeddings"].endswith(".npy")
    assert manifest["cache_artifact_paths"]["index"].endswith(".faiss")
    assert manifest["chunker_versions"]["chunker_implementation"]
    assert manifest["chunk_units"]["word"] == 2
    assert manifest["output_row_counts"]["results.jsonl"] == 1
    assert manifest["run_stats"]["n_chunks"] == 2
    assert manifest["run_stats"]["failed_questions"] == 0
    assert manifest["artifacts"]["results.jsonl"]["sha256"]
    assert manifest["artifacts"]["events.jsonl"]["sha256"]
    assert manifest["start_timestamp_utc"]
    assert manifest["end_timestamp_utc"]
    assert events[0]["event_type"] == "pipeline_start"
    assert events[-1]["event_type"] == "pipeline_end"
    assert any(event["event_type"] == "retrieval_end" and event["question_id"] == "q1" for event in events)
    assert any(event["event_type"] == "generation_end" and event["question_id"] == "q1" for event in events)


def test_pipeline1_manifest_records_txt_folder_input(tmp_path, monkeypatch):
    project_root = tmp_path
    data_dir = project_root / "data" / "raw"
    transformed_dir = data_dir / "transformed"
    transformed_dir.mkdir(parents=True)
    (transformed_dir / "sivas_manual_01.txt").write_text("alpha", encoding="utf-8")
    (transformed_dir / "sivas_manual_02.txt").write_text("beta", encoding="utf-8")
    (data_dir / "questions_fixed.jsonl").write_text('{"question_id":"q1","frage":"Q?"}\n', encoding="utf-8")
    cfg_path = project_root / "config.yaml"
    cfg_path.write_text(
        """
experiment:
  experiment_id: "test_txt_exp"
  random_seed: 42
  output_dir: "runs"
documents:
  path: "data/raw/transformed"
  source_type: "txt_folder"
  text_field: "cleaned_context"
  file_glob: "*.txt"
data:
  dataset_schema: "sivas"
  questions_path: "data/raw/questions_fixed.jsonl"
  question_id_field: "question_id"
  question_field: "frage"
chunking:
  strategy: "fixed_word"
  chunk_size: 10
  chunk_overlap: 0
embedding:
  provider: "sentence_transformers"
  model_name: "fake"
  normalize_embeddings: true
  batch_size: 2
  device: "cpu"
index:
  type: "faiss"
  metric: "cosine"
retrieval:
  retriever_type: "dense"
  top_k: 1
  fetch_k: 2
reranker:
  enabled: false
generation:
  provider: "ollama"
  model_name: "fake"
  system_prompt: "Use context."
telemetry:
  estimate_cost: false
runtime:
  save_csv: true
  log_level: "INFO"
  resume: false
  overwrite: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")
    monkeypatch.setattr("src.pipeline1.orchestrator._project_root", lambda: project_root)
    monkeypatch.setattr("src.pipeline1.orchestrator.build_embedder", lambda config: _FakeEmbedder())
    monkeypatch.setattr("src.pipeline1.orchestrator.build_index", lambda config: _FakeIndex())
    monkeypatch.setattr("src.pipeline1.orchestrator.build_generator", lambda config: _FakeGenerator())

    run_dir = run_pipeline(str(cfg_path))

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert (run_dir / "manifest.json").exists()
    assert manifest["data_hashes"]["documents_sha256"] is None
    assert manifest["data_hashes"]["documents_source_type"] == "txt_folder"
    assert manifest["data_hashes"]["txt_files_loaded"] == 2
    assert manifest["document_input"]["folder_path"] == str(transformed_dir)
    assert manifest["document_input"]["txt_files_loaded"] == 2
    assert manifest["run_stats"]["n_documents"] == 2
