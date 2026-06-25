import pytest

from src.pipeline1.generation.base import GenerationResult
from src.pipeline1.orchestration.parser import parse_orchestration_response
from src.pipeline1.orchestration.prompt import build_orchestration_prompt
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.orchestration_stage import OrchestrationStage


def test_orchestration_stage_cleans_question_and_detects_sivas_category():
    cfg = _cfg()
    chunks = [_chunk("c1", "Einkauf"), _chunk("c2", "Finanzen")]

    output = OrchestrationStage(
        cfg,
        chunks,
        generator_factory=lambda config: _FakeOrchestrationGenerator(),
    ).run(StageInput({"queries": [QueryRecord(question_id="Q001", question="  Wie bestelle ich?  ")]}))

    query = output.queries[0]
    assert query.cleaned_question == "Wie bestelle ich?"
    assert query.detected_category == "Einkauf"
    assert query.category_validated is True
    assert query.category_validation_reason is None


def test_orchestration_malformed_json_falls_back_without_crashing(monkeypatch):
    monkeypatch.setattr("src.pipeline1.stages.orchestration_stage.time.sleep", lambda seconds: None)
    cfg = _cfg()
    chunks = [_chunk("c1", "Einkauf")]

    output = OrchestrationStage(
        cfg,
        chunks,
        generator_factory=lambda config: _MalformedJsonGenerator(),
    ).run(
        StageInput(
            {
                "queries": [
                    QueryRecord(question_id="Q001", question="Original question?"),
                    QueryRecord(question_id="Q002", question="Second question?"),
                ]
            }
        )
    )

    first = output.queries[0]
    second = output.queries[1]
    assert first.cleaned_question == "Original question?"
    assert first.detected_category is None
    assert first.category_validated is False
    assert first.category_validation_reason == "orchestration failed before category validation"
    assert first.orchestration_error
    assert second.cleaned_question == "Second question?"


def test_orchestration_model_exception_falls_back_without_crashing(monkeypatch):
    monkeypatch.setattr("src.pipeline1.stages.orchestration_stage.time.sleep", lambda seconds: None)
    cfg = _cfg()

    output = OrchestrationStage(
        cfg,
        [_chunk("c1", "Einkauf")],
        generator_factory=lambda config: _FailingOrchestrationGenerator(),
    ).run(StageInput({"queries": [QueryRecord(question_id="Q001", question="Original question?")]}))

    query = output.queries[0]
    assert query.cleaned_question == "Original question?"
    assert query.detected_category is None
    assert query.category_validated is False
    assert query.category_validation_reason == "orchestration failed before category validation"
    assert query.orchestration_error == "orchestration down"


def test_orchestration_parser_rejects_answer_fields():
    with pytest.raises(ValueError, match="must not contain answer fields"):
        parse_orchestration_response(
            '{"cleaned_question":"Q?","detected_category":"Einkauf","answer":"42"}',
            "Q?",
            ["Einkauf"],
        )


def test_orchestration_parser_does_not_require_category_confidence():
    parsed = parse_orchestration_response(
        '{"cleaned_question":"Q?","detected_category":"Einkauf"}',
        "Q?",
        ["Einkauf"],
    )

    assert parsed == {
        "cleaned_question": "Q?",
        "detected_category": "Einkauf",
        "category_validated": True,
        "category_validation_reason": None,
    }


def test_orchestration_prompt_renders_optional_request_fields_as_null():
    prompt = build_orchestration_prompt("Q?", ["Einkauf"])

    assert "{{module_json}}" not in prompt
    assert "{{program_json}}" not in prompt
    assert "{{role_json}}" not in prompt
    assert "{{role_description_json}}" not in prompt


def test_orchestration_model_is_fixed_across_experiments():
    payload = _cfg_payload()
    payload["orchestration"] = {"model_name": "other-model", "fixed": True}

    with pytest.raises(ValueError, match="Orchestration model is fixed"):
        PipelineConfig.model_validate(payload)


class _FakeOrchestrationGenerator:
    def generate(self, prompt):
        assert "CategoryNames" in prompt
        assert "Einkauf" in prompt
        return GenerationResult(
            answer='{"cleaned_question":"Wie bestelle ich?","detected_category":"Einkauf"}',
            input_tokens=10,
            output_tokens=8,
        )


class _MalformedJsonGenerator:
    def generate(self, prompt):
        return GenerationResult(answer="{not json", input_tokens=1, output_tokens=1)


class _FailingOrchestrationGenerator:
    def generate(self, prompt):
        raise RuntimeError("orchestration down")


def _chunk(chunk_id: str, category: str):
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        original_context_id=f"doc-{chunk_id}",
        text="text",
        chunk_start=0,
        chunk_end=1,
        metadata={"kategorie": category},
    )


def _cfg():
    return PipelineConfig.model_validate(_cfg_payload())


def _cfg_payload():
    return {
        "experiment": {"experiment_id": "exp", "output_dir": "runs"},
        "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
        "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
        "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
        "index": {"type": "faiss", "metric": "cosine"},
        "retrieval": {"retriever_type": "category_aware_dense", "top_k": 1, "fetch_k": 2},
        "reranker": {"enabled": False},
        "orchestration": {"fixed": True},
        "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
        "telemetry": {},
        "runtime": {},
    }
