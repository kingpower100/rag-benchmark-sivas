from __future__ import annotations

import pytest

from src.pipeline3.schemas.pipeline3_config_schema import (
    P3JudgeConfig,
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
            relevancy=0.5,
            completeness=0.5,
            hallucination=0.5,
            context_relevance=0.5,
        )


def test_weights_summing_to_one_accepted():
    w = P3WeightsConfig(
        correctness=0.25,
        faithfulness=0.20,
        relevancy=0.20,
        completeness=0.15,
        hallucination=0.10,
        context_relevance=0.10,
    )
    total = (
        w.correctness
        + w.faithfulness
        + w.relevancy
        + w.completeness
        + w.hallucination
        + w.context_relevance
    )
    assert abs(total - 1.0) < 0.01


def test_judge_temperature_must_be_non_negative():
    with pytest.raises(Exception):
        P3JudgeConfig(temperature=-0.1)


def test_default_ragas_metrics_all_enabled():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.ragas.metrics.faithfulness is True
    assert cfg.ragas.metrics.answer_relevancy is True
    assert cfg.ragas.metrics.context_precision is True
    assert cfg.ragas.metrics.context_recall is True


def test_default_output_dir():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.pipeline3.output_dir == "data/eval/runs/pipeline3"


def test_default_judge_max_retries():
    cfg = Pipeline3Config.model_validate(_VALID_CONFIG)
    assert cfg.judge.max_retries == 3


def test_judge_max_retries_must_be_positive():
    with pytest.raises(Exception):
        P3JudgeConfig(max_retries=0)
