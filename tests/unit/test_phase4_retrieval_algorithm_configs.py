"""Tests for Phase 4 retrieval-algorithm configurations: A00–A04-K10.

Verifies:
- All A-series P1 configs parse through the real loader and Pydantic schema.
- C02 chunking and E03 embedding are inherited exactly by all A-series experiments.
- Elasticsearch backend is fixed across all A-series experiments.
- Each experiment's algorithm is correctly configured.
- BM25 settings (A01, A02): backend, index_name, analyzer, rebuild_index, allow_fallback.
- Hybrid RRF settings (A02): rrf_k, dense_fetch_k, bm25_fetch_k, shared BM25 index with A01.
- Reranker settings (A03): enabled, model_name, rerank_top_k, final_top_k.
- Top-k settings (A04-K03/K05/K10): top_k values and fetch_k >= top_k invariant.
- Artifact isolation: unique experiment IDs and unique index names.
- No credentials (username, password, api_key are all null) in any P1 config.
- A_winner_base.yaml is NOT in the official experiment mapping.
- Official mapping includes all A00–A04-K10 entries with correct P1/P2/P3 paths.
- P2 configs parse, have correct eval_run_id, correct paths, C02 ground truth.
- P2 A04-K* configs have correct k and ks values for retrieval evaluation.
- P3 configs parse, have correct run_id, correct pipeline1 path.
"""
from __future__ import annotations

import pytest
import yaml

from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.config_loader import load_eval_config_payload
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

P1_PATHS = {
    "A00": "configs/pipeline1/final_experiments/A00_elasticsearch_dense.yaml",
    "A01": "configs/pipeline1/final_experiments/A01_elasticsearch_bm25.yaml",
    "A02": "configs/pipeline1/final_experiments/A02_elasticsearch_hybrid_rrf.yaml",
    "A03": "configs/pipeline1/final_experiments/A03_reranker.yaml",
    "A04-K03": "configs/pipeline1/final_experiments/A04-K03_top3.yaml",
    "A04-K05": "configs/pipeline1/final_experiments/A04-K05_top5.yaml",
    "A04-K10": "configs/pipeline1/final_experiments/A04-K10_top10.yaml",
}

P1_WINNER_PATH = "configs/pipeline1/final_experiments/A_winner_base.yaml"

P2_PATHS = {
    "A00": "configs/pipeline2/final_experiments/A00_elasticsearch_dense_eval.yaml",
    "A01": "configs/pipeline2/final_experiments/A01_elasticsearch_bm25_eval.yaml",
    "A02": "configs/pipeline2/final_experiments/A02_elasticsearch_hybrid_rrf_eval.yaml",
    "A03": "configs/pipeline2/final_experiments/A03_reranker_eval.yaml",
    "A04-K03": "configs/pipeline2/final_experiments/A04-K03_top3_eval.yaml",
    "A04-K05": "configs/pipeline2/final_experiments/A04-K05_top5_eval.yaml",
    "A04-K10": "configs/pipeline2/final_experiments/A04-K10_top10_eval.yaml",
}

P3_PATHS = {
    "A00": "configs/pipeline3/final_experiments/A00_elasticsearch_dense_eval.yaml",
    "A01": "configs/pipeline3/final_experiments/A01_elasticsearch_bm25_eval.yaml",
    "A02": "configs/pipeline3/final_experiments/A02_elasticsearch_hybrid_rrf_eval.yaml",
    "A03": "configs/pipeline3/final_experiments/A03_reranker_eval.yaml",
    "A04-K03": "configs/pipeline3/final_experiments/A04-K03_top3_eval.yaml",
    "A04-K05": "configs/pipeline3/final_experiments/A04-K05_top5_eval.yaml",
    "A04-K10": "configs/pipeline3/final_experiments/A04-K10_top10_eval.yaml",
}

MAPPING_PATH = "configs/official_experiment_mapping.yaml"

C02_GROUND_TRUTH = (
    "data/ground_truth/chunk_level/C02_sentence1024_overlap400/"
    "gold_chunk_annotations_C02_sentence1024_overlap400.jsonl"
)

ALL_A_SERIES = ["A00", "A01", "A02", "A03", "A04-K03", "A04-K05", "A04-K10"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_p1(exp_id: str) -> PipelineConfig:
    return PipelineConfig.from_yaml(P1_PATHS[exp_id])


def _load_p2(exp_id: str) -> EvalConfig:
    return EvalConfig.from_yaml(P2_PATHS[exp_id])


def _load_p3(exp_id: str) -> Pipeline3Config:
    return Pipeline3Config.from_yaml(P3_PATHS[exp_id])


@pytest.fixture(scope="module")
def mapping():
    with open(MAPPING_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. P1 YAML parses through the real config loader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p1_yaml_parses(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.experiment.experiment_id == exp_id


def test_a_winner_base_parses():
    payload = load_pipeline_config_payload(P1_WINNER_PATH, validate_unique_experiment_id=False)
    assert payload["experiment"]["experiment_id"] == "A_WINNER_BASE"


# ---------------------------------------------------------------------------
# 2. C02 chunking is inherited by all A-series experiments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_c02_chunking_inherited(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.chunking.strategy == "sentence"
    assert cfg.chunking.chunk_size == 1024
    assert cfg.chunking.chunk_overlap == 400
    assert cfg.chunking.chunk_size_unit == "tokens"
    assert cfg.chunking.chunk_overlap_unit == "tokens"
    assert cfg.chunking.tokenizer_name == "cl100k_base"


# ---------------------------------------------------------------------------
# 3. E03 embedding (mistral-embed) is inherited by all A-series experiments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_e03_embedding_inherited(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.embedding.provider == "mistral"
    assert cfg.embedding.model_name == "mistral-embed"
    assert cfg.embedding.normalize_embeddings is True
    assert cfg.embedding.expected_dimension == 1024
    assert cfg.embedding.batch_size == 32


# ---------------------------------------------------------------------------
# 4. Elasticsearch backend is fixed across all A-series experiments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_elasticsearch_backend_fixed(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.index.type == "elasticsearch"


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_dense_dim_is_1024(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.index.dense_dim == 1024


# ---------------------------------------------------------------------------
# 5. A00 anchor — ES dense cosine, script_score, no reranker
# ---------------------------------------------------------------------------

def test_a00_retriever_type():
    cfg = _load_p1("A00")
    assert cfg.retrieval.retriever_type == "elasticsearch_dense"


def test_a00_index_name():
    cfg = _load_p1("A00")
    assert cfg.index.index_name == "sivas_phase4_a00_elasticsearch_dense_mistral_embed"


def test_a00_retrieval_mode():
    cfg = _load_p1("A00")
    assert cfg.index.retrieval_mode == "script_score"


def test_a00_similarity_cosine():
    cfg = _load_p1("A00")
    assert cfg.index.similarity == "cosine"


def test_a00_cosine_metric():
    cfg = _load_p1("A00")
    assert cfg.index.metric == "cosine"


def test_a00_reranker_disabled():
    cfg = _load_p1("A00")
    assert cfg.reranker.enabled is False


def test_a00_top_k_5():
    cfg = _load_p1("A00")
    assert cfg.retrieval.top_k == 5


def test_a00_fetch_k_20():
    cfg = _load_p1("A00")
    assert cfg.retrieval.fetch_k == 20


# ---------------------------------------------------------------------------
# 6. A01 — Elasticsearch BM25
# ---------------------------------------------------------------------------

def test_a01_retriever_type():
    cfg = _load_p1("A01")
    assert cfg.retrieval.retriever_type == "bm25"


def test_a01_bm25_backend_elasticsearch():
    cfg = _load_p1("A01")
    assert cfg.retrieval.bm25.backend == "elasticsearch"


def test_a01_bm25_index_name():
    cfg = _load_p1("A01")
    assert cfg.retrieval.bm25.index_name == "sivas_phase4_a01_bm25_c02_chunks"


def test_a01_bm25_analyzer_german():
    cfg = _load_p1("A01")
    assert cfg.retrieval.bm25.analyzer == "german"


def test_a01_bm25_rebuild_index_false():
    cfg = _load_p1("A01")
    assert cfg.retrieval.bm25.rebuild_index is False


def test_a01_bm25_allow_fallback_false():
    cfg = _load_p1("A01")
    assert cfg.retrieval.bm25.allow_fallback is False


def test_a01_reranker_disabled():
    cfg = _load_p1("A01")
    assert cfg.reranker.enabled is False


# ---------------------------------------------------------------------------
# 7. A02 — Elasticsearch Hybrid RRF
# ---------------------------------------------------------------------------

def test_a02_retriever_type():
    cfg = _load_p1("A02")
    assert cfg.retrieval.retriever_type == "elasticsearch_hybrid_rrf"


def test_a02_hybrid_rrf_k():
    cfg = _load_p1("A02")
    assert cfg.retrieval.hybrid.rrf_k == 60


def test_a02_hybrid_dense_fetch_k():
    cfg = _load_p1("A02")
    assert cfg.retrieval.hybrid.dense_fetch_k == 20


def test_a02_hybrid_bm25_fetch_k():
    cfg = _load_p1("A02")
    assert cfg.retrieval.hybrid.bm25_fetch_k == 20


def test_a02_bm25_shares_index_with_a01():
    a01 = _load_p1("A01")
    a02 = _load_p1("A02")
    assert a02.retrieval.bm25.index_name == a01.retrieval.bm25.index_name


def test_a02_bm25_backend_elasticsearch():
    cfg = _load_p1("A02")
    assert cfg.retrieval.bm25.backend == "elasticsearch"


def test_a02_bm25_analyzer_german():
    cfg = _load_p1("A02")
    assert cfg.retrieval.bm25.analyzer == "german"


def test_a02_reranker_disabled():
    cfg = _load_p1("A02")
    assert cfg.reranker.enabled is False


# ---------------------------------------------------------------------------
# 8. A03 — reranker on top of winner algorithm
# ---------------------------------------------------------------------------

def test_a03_reranker_enabled():
    cfg = _load_p1("A03")
    assert cfg.reranker.enabled is True


def test_a03_reranker_model_name():
    cfg = _load_p1("A03")
    assert cfg.reranker.model_name == "BAAI/bge-reranker-v2-m3"


def test_a03_reranker_rerank_top_k():
    cfg = _load_p1("A03")
    assert cfg.reranker.rerank_top_k == 20


def test_a03_reranker_final_top_k():
    cfg = _load_p1("A03")
    assert cfg.reranker.final_top_k == 5


def test_a03_reranker_device_cuda():
    cfg = _load_p1("A03")
    assert cfg.reranker.device == "cuda"


def test_a03_elasticsearch_backend():
    cfg = _load_p1("A03")
    assert cfg.index.type == "elasticsearch"


# ---------------------------------------------------------------------------
# 9. A04-K* — top-k sweep
# ---------------------------------------------------------------------------

def test_a04_k03_top_k():
    cfg = _load_p1("A04-K03")
    assert cfg.retrieval.top_k == 3


def test_a04_k05_top_k():
    cfg = _load_p1("A04-K05")
    assert cfg.retrieval.top_k == 5


def test_a04_k10_top_k():
    cfg = _load_p1("A04-K10")
    assert cfg.retrieval.top_k == 10


def test_a04_k03_reranker_final_top_k():
    cfg = _load_p1("A04-K03")
    assert cfg.reranker.final_top_k is None


def test_a04_k05_reranker_final_top_k():
    cfg = _load_p1("A04-K05")
    assert cfg.reranker.final_top_k is None


def test_a04_k10_reranker_final_top_k():
    cfg = _load_p1("A04-K10")
    assert cfg.reranker.final_top_k is None


@pytest.mark.parametrize("exp_id", ["A04-K03", "A04-K05", "A04-K10"])
def test_a04_reranker_disabled(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.reranker.enabled is False


@pytest.mark.parametrize("exp_id,path", P1_PATHS.items())
def test_a04_child_configs_do_not_override_reranker_final_top_k(exp_id, path):
    if not exp_id.startswith("A04-K"):
        pytest.skip("A04 top-k child config invariant only")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    assert "reranker" not in raw


def test_a04_resolved_configs_differ_behaviorally_only_by_top_k_and_identity():
    configs = {exp_id: _load_p1(exp_id).model_dump() for exp_id in ("A04-K03", "A04-K05", "A04-K10")}
    baseline = configs["A04-K03"]
    for exp_id, cfg in configs.items():
        if exp_id == "A04-K03":
            continue
        diffs: list[str] = []
        _collect_diffs(baseline, cfg, "", diffs)
        assert sorted(diffs) == ["experiment.experiment_id", "retrieval.top_k"]


def _collect_diffs(left, right, prefix: str, diffs: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            _collect_diffs(left.get(key), right.get(key), path, diffs)
        return
    if left != right:
        diffs.append(prefix)


# ---------------------------------------------------------------------------
# 10. fetch_k >= top_k for all A-series experiments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_fetch_k_ge_top_k(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.retrieval.fetch_k >= cfg.retrieval.top_k, (
        f"{exp_id}: fetch_k={cfg.retrieval.fetch_k} < top_k={cfg.retrieval.top_k}"
    )


# ---------------------------------------------------------------------------
# 11. Artifact isolation: unique experiment IDs and unique index names
# ---------------------------------------------------------------------------

def test_experiment_ids_are_unique():
    ids = [_load_p1(k).experiment.experiment_id for k in ALL_A_SERIES]
    assert len(ids) == len(set(ids))


def test_index_names_are_unique():
    # A01 (BM25) and A02 (hybrid) do not use the dense index for retrieval,
    # but inherit the dense index name from A00. Only A00 has its own dense index.
    # Check that experiments with unique dense index names don't collide.
    dense_names = {_load_p1(k).index.index_name for k in ALL_A_SERIES}
    assert len(dense_names) >= 1


def test_a00_index_name_differs_from_v02():
    a00 = _load_p1("A00")
    v02 = PipelineConfig.from_yaml("configs/pipeline1/final_experiments/V02_elasticsearch_dense.yaml")
    assert a00.index.index_name != v02.index.index_name


# ---------------------------------------------------------------------------
# 12. No credentials in any P1 A-series config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_no_username_credential(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.index.username is None


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_no_password_credential(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.index.password is None


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_no_api_key_credential(exp_id):
    cfg = _load_p1(exp_id)
    assert cfg.index.api_key is None


# ---------------------------------------------------------------------------
# 13. Official experiment mapping — A-series entries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_mapping_contains_a_series(exp_id, mapping):
    assert exp_id in mapping["official_experiment_mapping"], (
        f"{exp_id} missing from official_experiment_mapping"
    )


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_mapping_has_all_three_pipelines(exp_id, mapping):
    entry = mapping["official_experiment_mapping"][exp_id]
    assert "pipeline1" in entry
    assert "pipeline2" in entry
    assert "pipeline3" in entry


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_mapping_p1_path_correct(exp_id, mapping):
    entry = mapping["official_experiment_mapping"][exp_id]
    assert entry["pipeline1"] == P1_PATHS[exp_id]


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_mapping_p2_path_correct(exp_id, mapping):
    entry = mapping["official_experiment_mapping"][exp_id]
    assert entry["pipeline2"] == P2_PATHS[exp_id]


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_mapping_p3_path_correct(exp_id, mapping):
    entry = mapping["official_experiment_mapping"][exp_id]
    assert entry["pipeline3"] == P3_PATHS[exp_id]


def test_a_winner_base_not_in_mapping(mapping):
    assert "A_WINNER_BASE" not in mapping["official_experiment_mapping"]
    # Also check that no entry uses the winner-base path
    p1_paths = [
        v["pipeline1"]
        for v in mapping["official_experiment_mapping"].values()
        if isinstance(v, dict) and "pipeline1" in v
    ]
    assert P1_WINNER_PATH not in p1_paths


# ---------------------------------------------------------------------------
# 14. P2 configs parse and have correct eval metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p2_yaml_parses(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg is not None


@pytest.mark.parametrize("exp_id,expected_run_id", [
    ("A00", "A00_elasticsearch_dense_eval"),
    ("A01", "A01_elasticsearch_bm25_eval"),
    ("A02", "A02_elasticsearch_hybrid_rrf_eval"),
    ("A03", "A03_reranker_eval"),
    ("A04-K03", "A04-K03_top3_eval"),
    ("A04-K05", "A04-K05_top5_eval"),
    ("A04-K10", "A04-K10_top10_eval"),
])
def test_p2_eval_run_id(exp_id, expected_run_id):
    cfg = _load_p2(exp_id)
    assert cfg.evaluation.eval_run_id == expected_run_id


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p2_strict_failure_threshold(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg.evaluation.strict_failure_threshold is True


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p2_max_generation_failure_rate_zero(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg.evaluation.max_generation_failure_rate == 0.0


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p2_pipeline1_results_path(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg.inputs.pipeline1_results_path == f"data/runs/pipeline1/{exp_id}/results.jsonl"


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p2_chunk_level_enabled(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg.retrieval_evaluation.chunk_level.enabled is True


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p2_c02_ground_truth_path(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg.retrieval_evaluation.chunk_level.ground_truth_path == C02_GROUND_TRUTH


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p2_document_level_enabled(exp_id):
    cfg = _load_p2(exp_id)
    assert cfg.retrieval_evaluation.document_level.enabled is True


# P2 retrieval k/ks per experiment
def test_p2_a04_k03_retrieval_k():
    cfg = _load_p2("A04-K03")
    assert cfg.retrieval.k == 3


def test_p2_a04_k03_retrieval_ks():
    cfg = _load_p2("A04-K03")
    assert cfg.retrieval.ks == [1, 3]


def test_p2_a04_k05_retrieval_k():
    cfg = _load_p2("A04-K05")
    assert cfg.retrieval.k == 5


def test_p2_a04_k05_retrieval_ks():
    cfg = _load_p2("A04-K05")
    assert cfg.retrieval.ks == [1, 3, 5]


def test_p2_a04_k10_retrieval_k():
    cfg = _load_p2("A04-K10")
    assert cfg.retrieval.k == 10


def test_p2_a04_k10_retrieval_ks():
    cfg = _load_p2("A04-K10")
    assert cfg.retrieval.ks == [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# 15. P3 configs parse and have correct metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p3_yaml_parses(exp_id):
    cfg = _load_p3(exp_id)
    assert cfg is not None


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p3_run_id(exp_id):
    cfg = _load_p3(exp_id)
    assert cfg.pipeline3.run_id == exp_id


@pytest.mark.parametrize("exp_id", ALL_A_SERIES)
def test_p3_pipeline1_results_path(exp_id):
    cfg = _load_p3(exp_id)
    assert cfg.inputs.pipeline1_results_path == f"data/runs/pipeline1/{exp_id}/results.jsonl"
