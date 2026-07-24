from pathlib import Path
import types
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from src.pipeline1.embedding.bge_encoder import BGEEncoder
from src.pipeline1.embedding.factory import build_embedder
from src.pipeline1.orchestrator import _embedding_artifact_identity
from src.pipeline1.schemas.config_schema import EmbeddingConfig, PipelineConfig
from src.pipeline1.stages.embedding_stage import EmbeddingStage


def test_embedding_normalize_legacy_alias_true():
    cfg = EmbeddingConfig(provider="sentence_transformers", model_name="fake-model", normalize=True)

    assert cfg.normalize_embeddings is True
    assert "normalize" not in cfg.model_dump()


def test_embedding_normalize_legacy_alias_false():
    cfg = EmbeddingConfig(provider="sentence_transformers", model_name="fake-model", normalize=False)

    assert cfg.normalize_embeddings is False


def test_embedding_normalize_omitted_defaults_to_true():
    cfg = EmbeddingConfig(provider="sentence_transformers", model_name="fake-model")

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
    assert cfg.embedding.expected_dimension == 1024


def test_sentence_transformer_factory_receives_normalize_embeddings(monkeypatch):
    captured = {}

    class FakeBGEEncoder:
        def __init__(
            self,
            model_name,
            normalize_embeddings,
            batch_size,
            device,
            require_cuda,
            cache_dir,
            query_prefix,
            document_prefix,
            dense_output_only,
            pooling,
            max_seq_length,
            expected_dimension,
        ):
            captured.update(
                {
                    "model_name": model_name,
                    "normalize_embeddings": normalize_embeddings,
                    "batch_size": batch_size,
                    "device": device,
                    "require_cuda": require_cuda,
                    "cache_dir": cache_dir,
                    "query_prefix": query_prefix,
                    "document_prefix": document_prefix,
                    "dense_output_only": dense_output_only,
                    "pooling": pooling,
                    "max_seq_length": max_seq_length,
                    "expected_dimension": expected_dimension,
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
        query_prefix="query: ",
        document_prefix="passage: ",
        expected_dimension=384,
    )

    build_embedder(cfg)

    assert captured == {
        "model_name": "intfloat/multilingual-e5-small",
        "normalize_embeddings": False,
        "batch_size": 7,
        "device": "cpu",
        "require_cuda": False,
        "cache_dir": "cache/path",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "dense_output_only": True,
        "pooling": "sentence_transformers_default",
        "max_seq_length": None,
        "expected_dimension": 384,
    }


def test_mistral_factory_receives_api_runtime_fields(monkeypatch):
    captured = {}

    class FakeMistralEmbedder:
        def __init__(
            self,
            model_name,
            batch_size,
            normalize_embeddings,
            timeout_s,
            max_retries,
            retry_backoff_base_s,
            api_base_url,
            expected_dimension,
        ):
            captured.update(
                {
                    "model_name": model_name,
                    "batch_size": batch_size,
                    "normalize_embeddings": normalize_embeddings,
                    "timeout_s": timeout_s,
                    "max_retries": max_retries,
                    "retry_backoff_base_s": retry_backoff_base_s,
                    "api_base_url": api_base_url,
                    "expected_dimension": expected_dimension,
                }
            )

    monkeypatch.setattr("src.pipeline1.embedding.mistral_embedder.MistralEmbedder", FakeMistralEmbedder)
    cfg = EmbeddingConfig(
        provider="mistral",
        model_name="mistral-embed",
        normalize_embeddings=True,
        batch_size=9,
        expected_dimension=1024,
        api_base_url="https://api.mistral.ai/v1/embeddings",
        timeout_s=12,
        max_retries=4,
        retry_backoff_base_s=0.25,
    )

    build_embedder(cfg)

    assert captured == {
        "model_name": "mistral-embed",
        "batch_size": 9,
        "normalize_embeddings": True,
        "timeout_s": 12,
        "max_retries": 4,
        "retry_backoff_base_s": 0.25,
        "api_base_url": "https://api.mistral.ai/v1/embeddings",
        "expected_dimension": 1024,
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


def test_e5_configs_require_query_and_passage_prefixes():
    with pytest.raises(ValidationError, match="E5 embedding configs"):
        EmbeddingConfig(
            provider="sentence_transformers",
            model_name="intfloat/multilingual-e5-small",
            query_prefix="",
            document_prefix="",
        )


def test_bge_m3_config_requires_dense_output_only():
    with pytest.raises(ValidationError, match="dense_output_only=true"):
        EmbeddingConfig(
            provider="sentence_transformers",
            model_name="BAAI/bge-m3",
            dense_output_only=False,
        )


def test_mistral_embedding_config_requires_supported_model_and_dimension():
    cfg = EmbeddingConfig(
        provider="mistral",
        model_name="mistral-embed",
        expected_dimension=1024,
    )

    assert cfg.model_name == "mistral-embed"
    assert cfg.expected_dimension == 1024

    with pytest.raises(ValidationError, match="mistral-embed"):
        EmbeddingConfig(provider="mistral", model_name="made-up-model", expected_dimension=1024)
    with pytest.raises(ValidationError, match="expected_dimension=1024"):
        EmbeddingConfig(provider="mistral", model_name="mistral-embed", expected_dimension=768)


def test_encoder_applies_distinct_query_and_document_prefixes(monkeypatch):
    encoded_inputs = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, device="cpu", cache_folder=None):
            self.model_name = model_name
            self.device = device

        def encode(self, texts, **kwargs):
            encoded_inputs.extend(texts)
            return np.ones((len(texts), 3), dtype="float32")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = BGEEncoder(
        "intfloat/multilingual-e5-small",
        device="cpu",
        query_prefix="query: ",
        document_prefix="passage: ",
        expected_dimension=3,
    )

    encoder.encode_texts(["doc text"])
    encoder.encode_query("question")

    assert "passage: doc text" in encoded_inputs
    assert "query: question" in encoded_inputs


def test_encoder_rejects_unexpected_dimension(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name, device="cpu", cache_folder=None):
            self.device = device

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = BGEEncoder("BAAI/bge-m3", device="cpu", expected_dimension=3)

    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        encoder.encode_texts(["doc text"])


def test_embedding_cache_keys_are_isolated_by_model_and_dimension():
    c02 = PipelineConfig.from_yaml("configs/pipeline1/final_experiments/C02_sentence1024.yaml")
    e00 = PipelineConfig.from_yaml("configs/pipeline1/final_experiments/E00_multilingual_e5_small.yaml")
    e01 = PipelineConfig.from_yaml("configs/pipeline1/final_experiments/E01_bge_m3.yaml")
    e02 = PipelineConfig.from_yaml("configs/pipeline1/final_experiments/E02_multilingual_e5_base.yaml")
    e03 = PipelineConfig.from_yaml("configs/pipeline1/final_experiments/E03_mistral_api_embedding.yaml")

    chunks_key = "c02-chunks"
    keys = {
        EmbeddingStage._cache_key_for(chunks_key, cfg.embedding.model_dump())
        for cfg in (c02, e00, e01, e02, e03)
    }

    assert len(keys) == 4
    assert c02.embedding.model_dump() == e00.embedding.model_dump()
    assert e01.embedding.expected_dimension == e01.index.dense_dim == 1024
    assert e02.embedding.expected_dimension == e02.index.dense_dim == 768
    assert e03.embedding.provider == "mistral"
    assert e03.embedding.expected_dimension == e03.index.dense_dim == 1024
