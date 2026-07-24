"""Tests for Phase 4 vector-backend configurations: V00 (FAISS), V01 (pgvector), V02 (Elasticsearch).

Verifies:
- YAML parses through the real loader and Pydantic schema.
- C02 chunking is inherited exactly.
- E03 embedding is inherited exactly.
- Only backend fields differ.
- Artifact isolation: unique experiment IDs and index names.
- Similarity semantics: cosine for all three.
- Embedding dimension: 1024 for all three.
- fetch_k and top_k are identical.
- Reranker and orchestration are disabled.
- Official experiment mapping includes V00, V01, V02.
- Pipeline 2 and Pipeline 3 configs load and reference correct paths.
- Chunk-level ground truth is the same C02 package for all three.
- No credentials appear in YAML fields.
"""
from __future__ import annotations

import yaml
import pytest

from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.config_loader import load_eval_config_payload
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

P1_PATHS = {
    "V00": "configs/pipeline1/final_experiments/V00_faiss.yaml",
    "V01": "configs/pipeline1/final_experiments/V01_pgvector.yaml",
    "V02": "configs/pipeline1/final_experiments/V02_elasticsearch_dense.yaml",
}
P2_PATHS = {
    "V00": "configs/pipeline2/final_experiments/V00_faiss_eval.yaml",
    "V01": "configs/pipeline2/final_experiments/V01_pgvector_eval.yaml",
    "V02": "configs/pipeline2/final_experiments/V02_elasticsearch_dense_eval.yaml",
}
P3_PATHS = {
    "V00": "configs/pipeline3/final_experiments/V00_faiss_eval.yaml",
    "V01": "configs/pipeline3/final_experiments/V01_pgvector_eval.yaml",
    "V02": "configs/pipeline3/final_experiments/V02_elasticsearch_dense_eval.yaml",
}

C02_GROUND_TRUTH = (
    "data/ground_truth/chunk_level/C02_sentence1024_overlap400/"
    "gold_chunk_annotations_C02_sentence1024_overlap400.jsonl"
)


def _load_p1(exp_id: str) -> PipelineConfig:
    return PipelineConfig.from_yaml(P1_PATHS[exp_id])


def _load_p2(exp_id: str) -> EvalConfig:
    return EvalConfig.from_yaml(P2_PATHS[exp_id])


def _load_p3(exp_id: str) -> Pipeline3Config:
    return Pipeline3Config.from_yaml(P3_PATHS[exp_id])


# ---------------------------------------------------------------------------
# 1. YAML parses through the real loader
# ---------------------------------------------------------------------------

def test_v00_p1_yaml_parses():
    cfg = _load_p1("V00")
    assert cfg.experiment.experiment_id == "V00"


def test_v01_p1_yaml_parses():
    cfg = _load_p1("V01")
    assert cfg.experiment.experiment_id == "V01"


def test_v02_p1_yaml_parses():
    cfg = _load_p1("V02")
    assert cfg.experiment.experiment_id == "V02"


# ---------------------------------------------------------------------------
# 2. C02 chunking is inherited exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_c02_chunking_inherited(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.chunking.strategy == "sentence"
    assert cfg.chunking.chunk_size == 1024
    assert cfg.chunking.chunk_overlap == 400
    assert cfg.chunking.chunk_size_unit == "tokens"
    assert cfg.chunking.chunk_overlap_unit == "tokens"
    assert cfg.chunking.tokenizer_name == "cl100k_base"


# ---------------------------------------------------------------------------
# 3. E03 embedding is inherited exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_e03_embedding_inherited(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.embedding.provider == "mistral"
    assert cfg.embedding.model_name == "mistral-embed"
    assert cfg.embedding.normalize_embeddings is True
    assert cfg.embedding.expected_dimension == 1024
    assert cfg.embedding.batch_size == 32


# ---------------------------------------------------------------------------
# 4. Only backend fields differ between V00, V01, V02
# ---------------------------------------------------------------------------

def test_only_backend_fields_differ():
    v00 = _load_p1("V00")
    v01 = _load_p1("V01")
    v02 = _load_p1("V02")

    # Chunking is identical
    assert v00.chunking.model_dump() == v01.chunking.model_dump() == v02.chunking.model_dump()

    # Embedding is identical
    assert v00.embedding.model_dump() == v01.embedding.model_dump() == v02.embedding.model_dump()

    # Generation is identical
    assert v00.generation.model_dump() == v01.generation.model_dump() == v02.generation.model_dump()

    # Reranker is identical (all disabled)
    assert v00.reranker.model_dump() == v01.reranker.model_dump() == v02.reranker.model_dump()

    # top_k and fetch_k are identical
    assert v00.retrieval.top_k == v01.retrieval.top_k == v02.retrieval.top_k == 5
    assert v00.retrieval.fetch_k == v01.retrieval.fetch_k == v02.retrieval.fetch_k == 20

    # Index type is the only substantive difference
    assert v00.index.type == "faiss"
    assert v01.index.type == "pgvector"
    assert v02.index.type == "elasticsearch"


# ---------------------------------------------------------------------------
# 5. Artifact isolation: unique experiment IDs and index names
# ---------------------------------------------------------------------------

def test_experiment_ids_are_unique():
    ids = {_load_p1(k).experiment.experiment_id for k in ["V00", "V01", "V02"]}
    assert ids == {"V00", "V01", "V02"}


def test_index_names_are_unique():
    names = {_load_p1(k).index.index_name for k in ["V00", "V01", "V02"]}
    assert len(names) == 3


def test_v01_table_name_is_unique():
    cfg = _load_p1("V01")
    assert cfg.index.pgvector is not None
    # Must not share a table with B00 (chunk_embeddings)
    assert cfg.index.pgvector.table_name != "chunk_embeddings"
    assert cfg.index.pgvector.table_name == "v01_phase4_mistral_embed"


# ---------------------------------------------------------------------------
# 6. Similarity semantics: cosine for all three
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_cosine_metric(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.index.metric == "cosine"


def test_v02_similarity_is_cosine():
    cfg = _load_p1("V02")
    assert cfg.index.similarity == "cosine"


def test_v02_retrieval_mode_is_script_score():
    cfg = _load_p1("V02")
    assert cfg.index.retrieval_mode == "script_score"


# ---------------------------------------------------------------------------
# 7. Embedding dimension is 1024 for all three
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_dense_dim_is_1024(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.index.dense_dim == 1024


# ---------------------------------------------------------------------------
# 8. fetch_k and top_k are enforced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_fetch_k_and_top_k(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.fetch_k == 20


# ---------------------------------------------------------------------------
# 9. Reranker disabled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_reranker_disabled(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.reranker.enabled is False


# ---------------------------------------------------------------------------
# 10. Orchestration disabled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_orchestration_disabled(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.orchestration.enabled is False


# ---------------------------------------------------------------------------
# 11. Backend-specific retriever type
# ---------------------------------------------------------------------------

def test_v00_retriever_type_is_dense():
    cfg = _load_p1("V00")
    assert cfg.retrieval.retriever_type == "dense"


def test_v01_retriever_type_is_dense():
    cfg = _load_p1("V01")
    assert cfg.retrieval.retriever_type == "dense"


def test_v02_retriever_type_is_elasticsearch_dense():
    cfg = _load_p1("V02")
    assert cfg.retrieval.retriever_type == "elasticsearch_dense"


# ---------------------------------------------------------------------------
# 12. pgvector config block is present and correct for V01
# ---------------------------------------------------------------------------

def test_v01_pgvector_block():
    cfg = _load_p1("V01")
    assert cfg.index.pgvector is not None
    assert cfg.index.pgvector.dsn_env == "PGVECTOR_DSN"
    assert cfg.index.pgvector.schema_name == "rag"
    assert cfg.index.pgvector.index_type == "exact"
    assert cfg.index.pgvector.rebuild_index is False


def test_v00_no_pgvector_block():
    cfg = _load_p1("V00")
    assert cfg.index.pgvector is None


def test_v02_no_pgvector_block():
    cfg = _load_p1("V02")
    assert cfg.index.pgvector is None


# ---------------------------------------------------------------------------
# 13. No credentials in YAML
# ---------------------------------------------------------------------------

def test_v01_no_credentials_in_yaml():
    cfg = _load_p1("V01")
    assert cfg.index.username is None
    assert cfg.index.password is None
    assert cfg.index.api_key is None


def test_v02_no_credentials_in_yaml():
    cfg = _load_p1("V02")
    assert cfg.index.username is None
    assert cfg.index.password is None
    assert cfg.index.api_key is None


# ---------------------------------------------------------------------------
# 14. No BM25, no hybrid, no filters in V02
# ---------------------------------------------------------------------------

def test_v02_no_bm25_or_hybrid():
    cfg = _load_p1("V02")
    assert cfg.retrieval.retriever_type == "elasticsearch_dense"
    assert cfg.retrieval.metadata_boosting.enabled is False
    assert cfg.retrieval.metadata_filtering.enabled is False


# ---------------------------------------------------------------------------
# 15. Official experiment mapping contains V00, V01, V02
# ---------------------------------------------------------------------------

def test_official_mapping_contains_v_experiments():
    with open("configs/official_experiment_mapping.yaml", "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)["official_experiment_mapping"]

    assert "V00" in mapping
    assert "V01" in mapping
    assert "V02" in mapping

    assert mapping["V00"]["pipeline1"] == "configs/pipeline1/final_experiments/V00_faiss.yaml"
    assert mapping["V01"]["pipeline1"] == "configs/pipeline1/final_experiments/V01_pgvector.yaml"
    assert mapping["V02"]["pipeline1"] == "configs/pipeline1/final_experiments/V02_elasticsearch_dense.yaml"


def test_mapping_v_entries_have_all_three_pipelines():
    with open("configs/official_experiment_mapping.yaml", "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)["official_experiment_mapping"]

    for exp_id in ["V00", "V01", "V02"]:
        entry = mapping[exp_id]
        assert "pipeline1" in entry
        assert "pipeline2" in entry
        assert "pipeline3" in entry


# ---------------------------------------------------------------------------
# 16. Pipeline 2 configs load and reference correct paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_p2_config_loads(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg is not None


@pytest.mark.parametrize("exp_id,suffix", [
    ("V00", "V00_faiss_eval"),
    ("V01", "V01_pgvector_eval"),
    ("V02", "V02_elasticsearch_dense_eval"),
])
def test_p2_eval_run_id(exp_id, suffix):
    cfg = _load_p2(exp_id)
    assert cfg.evaluation.eval_run_id == suffix


@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_p2_strict_failure_threshold(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg.evaluation.strict_failure_threshold is True
    assert float(cfg.evaluation.max_generation_failure_rate) == 0.0


@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_p2_pipeline1_results_path(exp_id):
    cfg = _load_p2(exp_id)
    expected = f"data/runs/pipeline1/{exp_id}/results.jsonl"
    assert cfg.inputs.pipeline1_results_path == expected
    assert cfg.inputs.rag_outputs == [expected]


# ---------------------------------------------------------------------------
# 17. Chunk-level ground truth is the same C02 package for all three
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_p2_chunk_level_ground_truth(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg.retrieval_evaluation is not None
    assert cfg.retrieval_evaluation.chunk_level.enabled is True
    assert cfg.retrieval_evaluation.chunk_level.ground_truth_path == C02_GROUND_TRUTH


# ---------------------------------------------------------------------------
# 18. Pipeline 3 configs load and reference correct paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_p3_config_loads(exp_id):
    cfg = _load_p3(exp_id)
    assert cfg is not None


@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_p3_run_id(exp_id):
    cfg = _load_p3(exp_id)
    assert cfg.pipeline3.run_id == exp_id


@pytest.mark.parametrize("exp_id", ["V00", "V01", "V02"])
def test_p3_pipeline1_results_path(exp_id):
    cfg = _load_p3(exp_id)
    expected = f"data/runs/pipeline1/{exp_id}/results.jsonl"
    assert cfg.inputs.pipeline1_results_path == expected


# ---------------------------------------------------------------------------
# 19. Embedding cache reuse: all V experiments produce the same embeddings key
# ---------------------------------------------------------------------------

def test_embedding_cache_key_is_identical_across_v_experiments():
    from src.pipeline1.stages.embedding_stage import EmbeddingStage
    from src.pipeline1.utils.hashing import stable_hash_dict

    cfgs = {k: _load_p1(k) for k in ["V00", "V01", "V02"]}
    embedding_configs = {k: cfg.embedding.model_dump() for k, cfg in cfgs.items()}

    # All embedding configs must be identical
    assert embedding_configs["V00"] == embedding_configs["V01"] == embedding_configs["V02"]

    # Identical embedding config + identical chunks_key → identical cache key → same artifact
    sentinel_chunks_key = "sentinel"
    keys = {
        k: EmbeddingStage._cache_key_for(sentinel_chunks_key, embedding_configs[k])
        for k in ["V00", "V01", "V02"]
    }
    assert keys["V00"] == keys["V01"] == keys["V02"]


# ---------------------------------------------------------------------------
# 20. Full mapping round-trip: mapping → files → schema validate
# ---------------------------------------------------------------------------

def test_full_mapping_roundtrip_v_experiments():
    with open("configs/official_experiment_mapping.yaml", "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)["official_experiment_mapping"]

    for exp_id in ["V00", "V01", "V02"]:
        paths = mapping[exp_id]

        p1 = PipelineConfig.from_yaml(paths["pipeline1"])
        p2 = EvalConfig.from_yaml(paths["pipeline2"])
        p3 = Pipeline3Config.from_yaml(paths["pipeline3"])

        assert p1.experiment.experiment_id == exp_id
        assert p2.inputs.pipeline1_results_path == f"data/runs/pipeline1/{exp_id}/results.jsonl"
        assert p3.pipeline3.run_id == exp_id
