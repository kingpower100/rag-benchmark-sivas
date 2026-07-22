from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.pipeline1.generation.base import GenerationResult
from src.pipeline1.orchestrator import run_pipeline
from src.pipeline1.schemas.config_schema import OrchestrationConfig, PipelineConfig
from src.pipeline1.preflight import run_preflight_checks
from src.pipeline2.orchestrator import _category_routing_executed_for_row


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeEmbedder:
    def encode_texts(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype="float32")

    def encode_query(self, text):
        return np.ones(2, dtype="float32")


class _FakeIndex:
    metric = "cosine"

    def build(self, embeddings):
        self.embeddings = embeddings

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake", encoding="utf-8")

    def load(self, path):
        pass

    def search(self, query_embedding, top_k):
        count = min(top_k, len(getattr(self, "embeddings", [])) or 1)
        return np.ones(count, dtype="float32"), np.arange(count, dtype="int64")

    def search_hits(self, query_embedding, top_k):
        count = min(top_k, len(getattr(self, "embeddings", [])) or 1)
        return [
            {
                "_id": f"doc-{idx}:chunk:0000",
                "_score": 1.0 - (idx * 0.01),
                "_source": {
                    "chunk_id": f"doc-{idx}:chunk:0000",
                    "document_id": f"doc-{idx}",
                    "original_context_id": f"doc-{idx}",
                    "text": "alpha",
                    "metadata": {"kategorie": "ERP", "doc_key": f"doc-{idx}"},
                },
            }
            for idx in range(count)
        ]


class _GenerationOnlyGenerator:
    def generate(self, prompt):
        return GenerationResult(answer="answer", input_tokens=2, output_tokens=1)


class _OrchestrationGenerator:
    def generate(self, prompt):
        return GenerationResult(
            answer='{"cleaned_question":"Q?","detected_category":"ERP"}',
            input_tokens=2,
            output_tokens=2,
        )


def test_global_faiss_does_not_instantiate_orchestration_generator(tmp_path, monkeypatch):
    calls = _run_synthetic_pipeline(tmp_path, monkeypatch, index_type="faiss", retriever_type="dense")

    assert calls["orchestration"] == 0
    assert calls["generation"] == 1


def test_global_pgvector_does_not_instantiate_orchestration_generator(tmp_path, monkeypatch):
    calls = _run_synthetic_pipeline(tmp_path, monkeypatch, index_type="pgvector", retriever_type="dense")

    assert calls["orchestration"] == 0
    assert calls["generation"] == 1


def test_global_elasticsearch_does_not_instantiate_orchestration_generator(tmp_path, monkeypatch):
    calls = _run_synthetic_pipeline(
        tmp_path,
        monkeypatch,
        index_type="elasticsearch",
        retriever_type="elasticsearch_dense",
    )

    assert calls["orchestration"] == 0
    assert calls["generation"] == 1


def test_category_aware_retrieval_still_instantiates_orchestration_generator(tmp_path, monkeypatch):
    calls = _run_synthetic_pipeline(
        tmp_path,
        monkeypatch,
        index_type="faiss",
        retriever_type="category_aware_dense",
        orchestration_enabled=True,
    )

    assert calls["orchestration"] == 1
    assert calls["generation"] == 1


def test_category_aware_retrieval_rejects_orchestration_disabled():
    payload = _config_payload(index_type="faiss", retriever_type="category_aware_dense", orchestration_enabled=False)

    with pytest.raises(ValidationError, match="requires orchestration.enabled=true"):
        PipelineConfig.model_validate(payload)


def test_preflight_ignores_orchestration_prompt_and_model_when_disabled(tmp_path, monkeypatch):
    project_root = _write_inputs(tmp_path)
    payload = _config_payload(index_type="faiss", retriever_type="dense", orchestration_enabled=False)
    payload["orchestration"]["prompt_path"] = "missing-orchestration-prompt.txt"
    payload["orchestration"]["prompt_version"] = None
    payload["orchestration"]["model_name"] = "llama3.1:8b"
    payload["generation"]["model_name"] = "generation-model"
    cfg = PipelineConfig.model_validate(payload)

    monkeypatch.delenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", raising=False)
    monkeypatch.setattr("src.pipeline1.preflight._ollama_list_models", lambda: {"generation-model"})
    monkeypatch.setattr("src.pipeline1.preflight.requests.get", lambda *args, **kwargs: _FakeOllamaResponse())

    errors = run_preflight_checks(cfg, project_root)

    assert not any("orchestration.prompt_path" in error for error in errors)
    assert not any("llama3.1:8b" in error for error in errors)
    assert errors == []


def test_manifest_and_outputs_record_orchestration_disabled(tmp_path, monkeypatch):
    _run_synthetic_pipeline(tmp_path, monkeypatch, index_type="faiss", retriever_type="dense")
    run_dir = tmp_path / "runs" / "synthetic_dense"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert manifest["orchestration_enabled"] is False
    assert manifest["orchestration_status"] == "disabled"
    assert manifest["models"]["orchestration_prompt_sha256"] is None
    assert row["cleaned_question"] == "Q?"
    assert row["detected_category"] is None
    assert row["category_validated"] is False
    assert row["category_validation_reason"] == "orchestration_disabled"
    assert row["orchestration_error"] is None
    assert row["category_filter_applied"] is False
    assert row["category_fallback_used"] is False
    assert row["retrieval_diagnostics"]["retrieval_scope"] == "global"
    assert row["retrieval_diagnostics"]["fallback_used"] is False
    assert row["retrieval_diagnostics"]["orchestration_status"] == "disabled"


def test_pipeline2_suppresses_routing_metrics_for_pure_global_row():
    assert _category_routing_executed_for_row(
        {
            "retriever_type": "dense",
            "detected_category": None,
            "category_validated": False,
            "retrieval_diagnostics": {
                "retriever_type": "dense",
                "orchestration_status": "disabled",
            },
        }
    ) is False


def _run_synthetic_pipeline(
    tmp_path,
    monkeypatch,
    *,
    index_type: str,
    retriever_type: str,
    orchestration_enabled: bool = False,
) -> dict[str, int]:
    project_root = _write_inputs(tmp_path)
    cfg_path = project_root / "config.yaml"
    cfg_path.write_text(_config_yaml(index_type, retriever_type, orchestration_enabled), encoding="utf-8")
    calls = {"orchestration": 0, "generation": 0}

    def _build_generator(config):
        if isinstance(config, OrchestrationConfig):
            calls["orchestration"] += 1
            return _OrchestrationGenerator()
        calls["generation"] += 1
        return _GenerationOnlyGenerator()

    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")
    monkeypatch.setattr("src.pipeline1.orchestrator._project_root", lambda: project_root)
    monkeypatch.setattr("src.pipeline1.orchestrator.build_embedder", lambda config: _FakeEmbedder())
    monkeypatch.setattr("src.pipeline1.orchestrator.build_index", lambda config: _FakeIndex())
    monkeypatch.setattr("src.pipeline1.orchestrator.build_generator", _build_generator)
    if retriever_type == "category_aware_dense":
        monkeypatch.setattr(
            "src.pipeline1.retrieval.category_aware_dense_retriever.CategoryAwareDenseRetriever._build_category_retrievers",
            lambda self, embeddings, index_metric: None,
        )

    run_pipeline(str(cfg_path))
    return calls


def _write_inputs(project_root: Path) -> Path:
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "kb_documents_fixed.jsonl").write_text(
        '{"doc_key":"doc-0","doc_name":"doc.md","text":"alpha","kategorie":"ERP"}\n',
        encoding="utf-8",
    )
    (data_dir / "questions_fixed.jsonl").write_text('{"question_id":"q1","frage":"Q?"}\n', encoding="utf-8")
    return project_root


def _config_yaml(index_type: str, retriever_type: str, orchestration_enabled: bool) -> str:
    payload = _config_payload(index_type, retriever_type, orchestration_enabled)
    import yaml

    return yaml.safe_dump(payload, sort_keys=False)


def _config_payload(index_type: str, retriever_type: str, orchestration_enabled: bool = False) -> dict:
    payload = {
        "experiment": {"experiment_id": "synthetic_dense", "random_seed": 42, "output_dir": "runs"},
        "data": {
            "dataset_schema": "sivas",
            "documents_path": "data/raw/kb_documents_fixed.jsonl",
            "questions_path": "data/raw/questions_fixed.jsonl",
            "question_id_field": "question_id",
            "question_field": "frage",
            "document_text_field": "text",
            "allow_document_text_fallback": False,
            "allow_unsafe_query_fields": False,
        },
        "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
        "embedding": {
            "provider": "sentence_transformers",
            "model_name": "fake",
            "normalize_embeddings": True,
            "batch_size": 2,
            "device": "cpu",
        },
        "index": {"type": index_type, "metric": "cosine", "dense_dim": 2},
        "retrieval": {"retriever_type": retriever_type, "top_k": 1, "fetch_k": 1},
        "reranker": {"enabled": False},
        "orchestration": {
            "enabled": orchestration_enabled,
            "provider": "ollama",
            "fixed": True,
            "model_name": "llama3.1:8b",
            "prompt_path": str((PROJECT_ROOT / "src/pipeline1/prompts/orchestration_promptV4.txt").resolve()),
            "prompt_version": "orchestration_promptV4",
        },
        "generation": {
            "provider": "ollama",
            "model_name": "fake",
            "base_url": "http://localhost:11434",
            "system_prompt": "Use context.",
        },
        "telemetry": {"estimate_cost": False},
        "runtime": {"save_csv": False, "resume": False, "overwrite": True},
    }
    if index_type == "pgvector":
        payload["index"]["pgvector"] = {"dsn_env": "PGVECTOR_DSN"}
    return payload


class _FakeOllamaResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"name": "generation-model"}]}
