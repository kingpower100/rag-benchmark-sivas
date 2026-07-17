import pytest
from pydantic import ValidationError

from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.schemas.eval_config_schema import EvalConfig


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
    assert cfg.runtime.resume is True
    assert cfg.runtime.overwrite is False


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


def test_pipeline2_unknown_config_fields_fail():
    payload = {
        "evaluation": {"eval_run_id": "eval"},
        "inputs": {"rag_outputs": []},
        "retrieval": {"ks": [1, 3, 5], "unused_metric": True},
    }

    with pytest.raises(ValidationError, match="unused_metric"):
        EvalConfig.model_validate(payload)
