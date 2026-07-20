from pathlib import Path

import pytest
from pydantic import ValidationError

from src.pipeline1.embedding.factory import build_embedder
from src.pipeline1.orchestrator import _embedding_artifact_identity
from src.pipeline1.schemas.config_schema import EmbeddingConfig, PipelineConfig


def test_embedding_normalize_legacy_alias_true():
    cfg = EmbeddingConfig(provider="sentence_transformers", model_name="intfloat/multilingual-e5-small", normalize=True)

    assert cfg.normalize_embeddings is True
    assert "normalize" not in cfg.model_dump()


def test_embedding_normalize_legacy_alias_false():
    cfg = EmbeddingConfig(provider="sentence_transformers", model_name="intfloat/multilingual-e5-small", normalize=False)

    assert cfg.normalize_embeddings is False


def test_embedding_normalize_omitted_defaults_to_true():
    cfg = EmbeddingConfig(provider="sentence_transformers", model_name="intfloat/multilingual-e5-small")

    assert cfg.normalize_embeddings is True


def test_embedding_normalize_conflicting_aliases_fail():
    with pytest.raises(ValidationError, match="legacy alias"):
        EmbeddingConfig.model_validate(
            {
                "provider": "sentence_transformers",
                "model_name": "intfloat/multilingual-e5-small",
                "normalize": False,
                "normalize_embeddings": True,
            }
        )


def test_b00_config_loads_with_embedding_normalization_default():
    cfg = PipelineConfig.from_yaml("configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml")

    assert cfg.experiment.experiment_id == "B00_sivas_pgvector_reference"
    assert cfg.embedding.provider == "mistral"
    assert cfg.embedding.model_name == "mistral-embed"
    assert cfg.embedding.normalize_embeddings is True


def test_sentence_transformer_factory_receives_normalize_embeddings(monkeypatch):
    captured = {}

    class FakeBGEEncoder:
        def __init__(self, model_name, normalize_embeddings, batch_size, device, require_cuda, cache_dir):
            captured.update(
                {
                    "model_name": model_name,
                    "normalize_embeddings": normalize_embeddings,
                    "batch_size": batch_size,
                    "device": device,
                    "require_cuda": require_cuda,
                    "cache_dir": cache_dir,
                }
            )

    monkeypatch.setattr("src.pipeline1.embedding.factory.BGEEncoder", FakeBGEEncoder)
    cfg = EmbeddingConfig(
        provider="sentence_transformers",
        model_name="intfloat/multilingual-e5-small",
        normalize=False,
        batch_size=7,
        device="cpu",
        require_cuda=False,
        cache_dir="cache/path",
    )

    build_embedder(cfg)

    assert captured == {
        "model_name": "intfloat/multilingual-e5-small",
        "normalize_embeddings": False,
        "batch_size": 7,
        "device": "cpu",
        "require_cuda": False,
        "cache_dir": "cache/path",
    }


def test_orchestrator_embedding_artifact_identity_uses_schema_field():
    config_path = "configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml"
    cfg = PipelineConfig.from_yaml(config_path)

    identity = _embedding_artifact_identity(
        cfg,
        documents_fingerprint="documents-fingerprint",
        chunks_key="chunks-key",
        config_path=config_path,
        project_root=Path.cwd(),
    )

    assert identity["embedding_model_name"] == "mistral-embed"
    assert identity["embedding_normalization"] is True
