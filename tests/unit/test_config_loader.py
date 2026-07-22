import pytest
from pydantic import ValidationError

from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.config_loader import load_eval_config_payload
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.config_utils import official_config_files


def _minimal_pipeline1_payload(orchestration_model: str) -> dict:
    return {
        "experiment": {"experiment_id": "exp", "output_dir": "runs"},
        "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
        "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
        "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
        "index": {"type": "faiss", "metric": "cosine"},
        "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
        "reranker": {"enabled": False},
        "orchestration": {"model_name": orchestration_model},
        "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
        "telemetry": {"estimate_cost": False},
        "runtime": {"resume": False, "overwrite": True},
    }


def test_pipeline1_sivas_baseline_config_loads():
    cfg = PipelineConfig.from_yaml("configs/pipeline1/experiments Orchestration LLM/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml")

    assert cfg.experiment.experiment_id == "91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0"
    assert cfg.data.dataset_schema == "sivas"
    assert cfg.data.documents_path == "data/raw/kb_documents_fixed.jsonl"
    assert cfg.data.questions_path == "data/raw/questions_fixed.jsonl"
    assert cfg.data.document_text_field == "text"
    assert cfg.data.question_id_field == "question_id"
    assert cfg.data.question_field == "frage"
    assert cfg.retrieval.retriever_type == "category_aware_dense"
    assert cfg.index.type == "faiss"
    assert cfg.generation.model_name == "qwen2.5:7b"
    assert cfg.orchestration.prompt_version == "v0"
    assert cfg.orchestration.prompt_path == "src/pipeline1/prompts/orchestration_prompt.txt"
    assert cfg.runtime.resume is True
    assert cfg.runtime.overwrite is False


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    cfg_path = tmp_path / "dup.yaml"
    cfg_path.write_text(
        """
experiment:
  experiment_id: "one"
experiment:
  experiment_id: "two"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate YAML key 'experiment'"):
        load_pipeline_config_payload(str(cfg_path), validate_unique_experiment_id=False)


def test_official_pipeline2_configs_enable_chunk_evaluation():
    payload = load_eval_config_payload("configs/pipeline2/final_experiments/E00-G_global_dense_baseline_eval.yaml")

    chunk_level = payload["retrieval_evaluation"]["chunk_level"]
    assert chunk_level["enabled"] is True
    assert chunk_level["missing_question_policy"] == "error"
    assert chunk_level["ground_truth_path"].endswith(
        "gold_chunk_annotations_E00-G_sentence512_overlap200.jsonl"
    )


def test_official_pipeline2_configs_enforce_zero_failure_threshold():
    for cfg_path in official_config_files("pipeline2"):
        payload = load_eval_config_payload(str(cfg_path))
        evaluation = payload["evaluation"]
        assert evaluation["strict_failure_threshold"] is True, cfg_path
        assert float(evaluation["max_generation_failure_rate"]) == 0.0, cfg_path


def test_pipeline1_base_uses_sivas_defaults_and_safe_run_defaults():
    cfg = PipelineConfig.from_yaml("configs/pipeline1/base.yaml")

    assert cfg.data.dataset_schema == "sivas"
    assert cfg.data.documents_path == "data/raw/kb_documents_fixed.jsonl"
    assert cfg.data.questions_path == "data/raw/questions_fixed.jsonl"
    assert cfg.data.document_text_field == "text"
    assert cfg.data.allow_unsafe_query_fields is False
    assert cfg.runtime.resume is False
    assert cfg.runtime.overwrite is True


def test_pipeline2_base_uses_sivas_defaults():
    cfg = EvalConfig.from_yaml("configs/pipeline2/base_eval.yaml")

    assert cfg.evaluation.eval_run_id == "91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval"
    assert cfg.inputs.qa_path == "data/raw/qa_ground_truth_fixed.jsonl"
    assert cfg.inputs.questions_path == "data/raw/questions_fixed.jsonl"
    assert (
        cfg.inputs.pipeline1_results_path
        == "data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl"
    )
    assert cfg.inputs.rag_outputs == [
        "data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl"
    ]


def test_pipeline1_unknown_config_fields_fail():
    payload = {
        "experiment": {"experiment_id": "exp", "output_dir": "runs"},
        "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
        "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0, "fake_knob": True},
        "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
        "index": {"type": "faiss", "metric": "cosine"},
        "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
        "reranker": {"enabled": False},
        "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
        "telemetry": {"estimate_cost": False},
        "runtime": {"resume": False, "overwrite": True},
    }

    with pytest.raises(ValidationError, match="fake_knob"):
        PipelineConfig.model_validate(payload)


@pytest.mark.parametrize("model_name", ["mistral-small", "qwen2.5:7b", "llama3.1:8b"])
def test_pipeline1_orchestration_model_allowlist_accepts_phase2_models(model_name):
    cfg = PipelineConfig.model_validate(_minimal_pipeline1_payload(model_name))

    assert cfg.orchestration.model_name == model_name


def test_pipeline1_orchestration_model_allowlist_rejects_invalid_model():
    with pytest.raises(ValidationError, match="Unsupported orchestration.model_name 'invalid-model'"):
        PipelineConfig.model_validate(_minimal_pipeline1_payload("invalid-model"))


def test_reranker_enabled_requires_model_name():
    payload = _minimal_pipeline1_payload("llama3.1:8b")
    payload["reranker"] = {"enabled": True}

    with pytest.raises(ValidationError, match="reranker.model_name is required when reranker.enabled=true"):
        PipelineConfig.model_validate(payload)


@pytest.mark.parametrize("model_name", [None, "", "   "])
def test_reranker_enabled_rejects_missing_or_blank_model_name(model_name):
    payload = _minimal_pipeline1_payload("llama3.1:8b")
    payload["reranker"] = {"enabled": True, "model_name": model_name}

    with pytest.raises(ValidationError, match="reranker.model_name"):
        PipelineConfig.model_validate(payload)


def test_reranker_disabled_allows_missing_model_name():
    payload = _minimal_pipeline1_payload("llama3.1:8b")
    payload["reranker"] = {"enabled": False}

    cfg = PipelineConfig.model_validate(payload)

    assert cfg.reranker.enabled is False
    assert cfg.reranker.model_name is None


def test_pipeline2_unknown_config_fields_fail():
    payload = {
        "evaluation": {"eval_run_id": "eval"},
        "inputs": {"rag_outputs": []},
        "retrieval": {"ks": [1, 3, 5], "unused_metric": True},
    }

    with pytest.raises(ValidationError, match="unused_metric"):
        EvalConfig.model_validate(payload)
