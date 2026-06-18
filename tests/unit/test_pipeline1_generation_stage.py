import json

from src.pipeline1.generation.base import GenerationResult
from src.pipeline1.observability.events import EventWriter
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.generation_stage import GenerationStage
from src.pipeline1.stages.retrieval_stage import RetrievalRow


def test_generation_stage_produces_output_record_compatible_row():
    cfg = _cfg()
    output = GenerationStage(cfg, _Retriever(), generator_factory=lambda config: _Generator("42")).run(
        StageInput({"retrieval_rows": [_retrieval_row()], "final_top_k": 1})
    )

    record = output.generation_rows[0].output_record
    assert record.question_id == "q1"
    assert record.generated_answer == "42"
    assert record.retrieved_chunks == ["c1"]
    assert record.retrieved_chunk_ids == ["c1"]
    assert record.retrieved_original_context_ids == ["ctx1"]
    assert record.retrieved_documents == ["doc-key-1"]
    assert record.raw_retrieved_context_ids == ["c1"]
    assert record.input_tokens == 5
    assert record.output_tokens == 1
    assert record.total_tokens == 6
    assert record.error is None


def test_generation_stage_error_path_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline1.stages.generation_stage.time.sleep", lambda seconds: None)
    cfg = _cfg()
    events = EventWriter(tmp_path / "events.jsonl", experiment_id="exp")
    generator = _FailingGenerator()
    output = GenerationStage(cfg, _Retriever(), event_writer=events, generator_factory=lambda config: generator).run(
        StageInput({"retrieval_rows": [_retrieval_row()], "final_top_k": 1})
    )
    events.close()

    record = output.generation_rows[0].output_record
    event_rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert record.generated_answer == ""
    assert record.input_tokens == 0
    assert record.output_tokens == 0
    assert record.total_tokens == 0
    assert record.error == "boom"
    assert generator.calls == 3
    assert any(row["event_type"] == "generation_error" for row in event_rows)
    assert any(row["event_type"] == "generation_end" and row["metrics"]["generation_failed"] is True for row in event_rows)


def test_generation_stage_retries_and_continues_after_transient_failure(monkeypatch):
    monkeypatch.setattr("src.pipeline1.stages.generation_stage.time.sleep", lambda seconds: None)
    cfg = _cfg()
    generator = _FlakyGenerator(failures_before_success=2)

    output = GenerationStage(cfg, _Retriever(), generator_factory=lambda config: generator).run(
        StageInput({"retrieval_rows": [_retrieval_row()], "final_top_k": 1})
    )

    record = output.generation_rows[0].output_record
    assert generator.calls == 3
    assert record.generated_answer == "recovered"
    assert record.error is None
    assert record.input_tokens == 7
    assert record.output_tokens == 2


def test_generation_stage_failed_question_does_not_stop_remaining_rows(monkeypatch):
    monkeypatch.setattr("src.pipeline1.stages.generation_stage.time.sleep", lambda seconds: None)
    cfg = _cfg()
    generator = _FailFirstQuestionGenerator()

    output = GenerationStage(cfg, _Retriever(), generator_factory=lambda config: generator).run(
        StageInput(
            {
                "retrieval_rows": [
                    _retrieval_row(question_id="q1"),
                    _retrieval_row(question_id="q2", question="Second question?"),
                ],
                "final_top_k": 1,
            }
        )
    )

    first = output.generation_rows[0].output_record
    second = output.generation_rows[1].output_record
    assert first.question_id == "q1"
    assert first.generated_answer == ""
    assert first.error == "first failed"
    assert second.question_id == "q2"
    assert second.generated_answer == "second ok"
    assert second.error is None


def test_generation_stage_preserves_prompt_context_diagnostics():
    cfg = _cfg(max_context_chars=5)
    output = GenerationStage(cfg, _Retriever(), generator_factory=lambda config: _Generator("ok")).run(
        StageInput({"retrieval_rows": [_retrieval_row(text="alpha beta gamma")], "final_top_k": 1})
    )

    record = output.generation_rows[0].output_record
    assert record.prompt_stats["chunks_before"] == 1
    assert record.prompt_stats["chunks_after"] in {0, 1}
    assert record.context_chars_before > record.context_chars_after
    assert record.prompt_tokens is not None


class _Generator:
    def __init__(self, answer):
        self.answer = answer

    def generate(self, prompt):
        return GenerationResult(answer=self.answer, input_tokens=5, output_tokens=1)


class _FailingGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        raise RuntimeError("boom")


class _FlakyGenerator:
    def __init__(self, failures_before_success):
        self.failures_before_success = failures_before_success
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise RuntimeError(f"transient {self.calls}")
        return GenerationResult(answer="recovered", input_tokens=7, output_tokens=2)


class _FailFirstQuestionGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if "What is the answer?" in prompt:
            raise RuntimeError("first failed")
        return GenerationResult(answer="second ok", input_tokens=5, output_tokens=2)


class _Retriever:
    def extract_query_metadata(self, question):
        return None


def _retrieval_row(text="alpha", question_id="q1", question="What is the answer?"):
    item = RetrievalItem(
        chunk_id="c1",
        original_context_id="ctx1",
        text=text,
        score=1.0,
        dense_score=1.0,
        metadata={"doc_id": 1, "doc_key": "doc-key-1", "document_id": "doc1", "file_name": "source.txt"},
    )
    return RetrievalRow(
        query=QueryRecord(question_id=question_id, question=question),
        raw_retrieved=[item],
        raw_dense_retrieved=[item],
        raw_bm25_retrieved=[],
        fused_retrieved=[],
        retrieved=[item],
        retrieval_time_ms=2.0,
        reranker_used=False,
        retrieval_warnings=[],
        retrieval_diagnostics={"diagnostic": "ok"},
    )


def _cfg(max_context_chars=24000):
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
            "reranker": {"enabled": False},
            "generation": {
                "provider": "ollama",
                "model_name": "fake",
                "system_prompt": "Use context.",
                "max_context_chars": max_context_chars,
                "max_chunk_chars": 8000,
            },
            "telemetry": {},
            "runtime": {},
        }
    )
