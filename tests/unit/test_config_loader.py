from pathlib import Path

import pytest
from pydantic import ValidationError

from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def test_pipeline_configs_load_current_examples():
    p1 = PipelineConfig.from_yaml(
        "configs/pipeline1/experiments/"
        "01_officeqa_treasury_tableaware512_bge_small_dense_rerank3_qwen25_7b_ctxbudget.yaml"
    )
    p2 = EvalConfig.from_yaml(
        "configs/pipeline2/experiments/"
        "01_eval_officeqa_treasury_tableaware512_bge_small_dense_rerank3_qwen25_7b_fileeval_ks1_3_5_10_numacc.yaml"
    )

    assert p1.experiment.experiment_id == "01_officeqa_treasury_tableaware512_bge_small_dense_rerank3_qwen25_7b_ctxbudget"
    assert p1.retrieval.top_k == 5
    assert p1.generation.model_name == "qwen2.5:7b"
    assert p1.runtime.overwrite is False
    assert p1.embedding.device == "cuda"
    assert p1.embedding.require_cuda is True
    assert p1.reranker.device == "cuda"
    assert (
        p2.evaluation.eval_run_id
        == "01_eval_officeqa_treasury_tableaware512_bge_small_dense_rerank3_qwen25_7b_fileeval_ks1_3_5_10_numacc"
    )
    assert p2.evaluation.retrieval_eval_field == "retrieved_file_names"
    assert p2.evaluation.max_generation_failure_rate == 0.05
    assert p2.evaluation.strict_failure_threshold is False
    assert p2.retrieval.k == 5
    assert p2.runtime.overwrite is False


def test_pipeline1_base_uses_question_only_and_safe_run_defaults():
    p1 = PipelineConfig.from_yaml("configs/pipeline1/base.yaml")

    assert p1.data.questions_path == "data/raw/questions_only.jsonl"
    assert p1.data.document_text_field == "cleaned_context"
    assert p1.data.allow_unsafe_query_fields is False
    assert p1.runtime.resume is False
    assert p1.runtime.overwrite is True


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


def test_pipeline2_unknown_config_fields_fail():
    payload = {
        "evaluation": {"eval_run_id": "eval"},
        "inputs": {"rag_outputs": []},
        "retrieval": {"ks": [1, 3, 5], "unused_metric": True},
    }

    with pytest.raises(ValidationError, match="unused_metric"):
        EvalConfig.model_validate(payload)
