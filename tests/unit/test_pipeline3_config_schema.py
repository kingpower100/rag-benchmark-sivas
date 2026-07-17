from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline3.schemas.pipeline3_config_schema import (
    P3JudgeConfig,
    P3LLMJudgeConfig,
    P3WeightsConfig,
    Pipeline3Config,
)

_VALID_CONFIG = {
    "pipeline3": {"run_id": "test_run"},
    "inputs": {
        "pipeline1_results_path": "data/runs/pipeline1/test/results.jsonl"
    },
}


def test_valid_config_loads():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.pipeline3.run_id == "test_run"
    assert cfg.judge.model == "qwen2.5:14b"
    assert cfg.judge.temperature == 0.0


def test_extra_fields_rejected():
    bad = {**_VALID_CONFIG, "unknown_key": "value"}
    with pytest.raises(Exception):
        Pipeline3Config.model_validate(bad)


def test_weights_must_sum_to_one():
    with pytest.raises(Exception):
        P3WeightsConfig(
            correctness=0.5,
            faithfulness=0.5,
            completeness=0.5,
            hallucination=0.5,
            context_relevance=0.5,
        )


def test_weights_summing_to_one_accepted():
    w = P3WeightsConfig(
        correctness=0.30,
        faithfulness=0.25,
        completeness=0.20,
        hallucination=0.15,
        context_relevance=0.10,
    )
    total = (
        w.correctness
        + w.faithfulness
        + w.completeness
        + w.hallucination
        + w.context_relevance
    )
    assert abs(total - 1.0) < 0.01


def test_judge_temperature_must_be_non_negative():
    with pytest.raises(Exception):
        P3JudgeConfig(temperature=-0.1)


def test_default_ragas_metrics_faithfulness_and_answer_relevancy_enabled():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.ragas.metrics.faithfulness is True
    assert cfg.ragas.metrics.answer_relevancy is True
    assert not hasattr(cfg.ragas.metrics, "context_precision")


def test_default_context_recall_disabled():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.ragas.metrics.context_recall is False


def test_default_ragas_requires_explicit_cuda_and_fail_on_error():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.ragas.embeddings_device == "cuda"
    assert cfg.ragas.require_cuda is True
    assert cfg.ragas.fail_on_ragas_error is True


def test_default_output_dir():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.pipeline3.output_dir == "data/eval/runs/pipeline3"


def test_default_judge_max_retries():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.judge.max_retries == 3


def test_judge_max_retries_must_be_positive():
    with pytest.raises(Exception):
        P3JudgeConfig(max_retries=0)


def test_llm_judge_max_context_chars_default_is_backward_compatible():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.llm_judge.max_context_chars == 6000


def test_llm_judge_max_context_chars_accepts_configured_value():
    payload = {
        **_VALID_CONFIG,
        "llm_judge": {
            "enabled": True,
            "max_context_chars": 10000,
        },
    }
    cfg = Pipeline3Config.model_validate(payload)
    assert cfg.llm_judge.max_context_chars == 10000


def test_llm_judge_max_context_chars_must_be_positive():
    with pytest.raises(Exception):
        P3LLMJudgeConfig(max_context_chars=0)


def test_existing_pipeline3_configs_load_successfully():
    config_dir = Path("configs/pipeline3")
    config_paths = sorted(config_dir.rglob("*.yaml"))
    assert config_paths
    for config_path in config_paths:
        Pipeline3Config.from_yaml(str(config_path))
