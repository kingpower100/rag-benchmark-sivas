import json

import pytest

from src.pipeline1.io.result_writer import ResultWriter
from src.pipeline1.orchestrator import _prepare_run_dir, _validate_resume_compatible


def _payload() -> dict:
    return {
        "experiment_id": "exp",
        "config_hash": "cfg",
        "documents_fingerprint": "docs",
        "cache_keys": {"chunks": "c", "embeddings": "e", "index": "i"},
        "retrieval": {"retriever_type": "dense", "top_k": 5, "fetch_k": 20},
        "reranker": {"enabled": False, "model_name": None, "device": "cpu", "rerank_top_k": None, "final_top_k": None},
        "generation": {"model_name": "m"},
        "orchestration": {"model_name": "o"},
        "prompt_template_version": "p",
    }


def test_resume_with_same_manifest_succeeds(tmp_path):
    payload = _payload()
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resume_compatibility": payload}), encoding="utf-8")

    _validate_resume_compatible(tmp_path, payload)


def test_resume_with_changed_config_hash_fails(tmp_path):
    previous = {**_payload(), "config_hash": "old"}
    current = {**previous, "config_hash": "new"}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resume_compatibility": previous}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="config_hash"):
        _validate_resume_compatible(tmp_path, current)


def test_resume_with_changed_chunk_cache_key_fails(tmp_path):
    previous = {**_payload(), "cache_keys": {"chunks": "old", "embeddings": "e", "index": "i"}}
    current = {**previous, "cache_keys": {"chunks": "new", "embeddings": "e", "index": "i"}}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resume_compatibility": previous}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="cache_keys"):
        _validate_resume_compatible(tmp_path, current)


@pytest.mark.parametrize(
    ("field", "previous_update", "current_update"),
    [
        ("reranker", {"enabled": False, "model_name": None}, {"enabled": True, "model_name": "reranker-a"}),
        ("reranker", {"enabled": True, "model_name": "reranker-a"}, {"enabled": False, "model_name": None}),
        ("reranker", {"enabled": True, "model_name": "reranker-a"}, {"enabled": True, "model_name": "reranker-b"}),
        ("reranker", {"rerank_top_k": 10}, {"rerank_top_k": 20}),
        ("reranker", {"final_top_k": 5}, {"final_top_k": 3}),
        ("retrieval", {"fetch_k": 20}, {"fetch_k": 40}),
    ],
)
def test_resume_rejects_retrieval_or_reranker_changes(tmp_path, field, previous_update, current_update):
    previous = _payload()
    current = _payload()
    previous[field] = {**previous[field], **previous_update}
    current[field] = {**current[field], **current_update}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resume_compatibility": previous}), encoding="utf-8")

    with pytest.raises(RuntimeError, match=field):
        _validate_resume_compatible(tmp_path, current)


def test_resume_requires_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="run_manifest"):
        _validate_resume_compatible(tmp_path, {"experiment_id": "exp"})


def test_overwrite_true_allows_clean_rerun(tmp_path):
    for name in ("results.jsonl", "results.csv", "run_manifest.json", "logs.txt", "pipeline1.log"):
        (tmp_path / name).write_text("old", encoding="utf-8")

    _prepare_run_dir(tmp_path, resume=False, overwrite=True)

    assert not (tmp_path / "results.jsonl").exists()
    assert not (tmp_path / "run_manifest.json").exists()


def test_resume_false_existing_dir_fails_without_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        _prepare_run_dir(tmp_path, resume=False, overwrite=False)


# ── ResultWriter.load_existing_question_ids — validity-aware resume ───────────

def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_resume_skips_valid_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            {"question_id": "q1", "answer": "Good answer", "error": None},
            {"question_id": "q2", "answer": "Another answer", "error": None},
        ],
    )
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    assert ids == {"q1", "q2"}


def test_resume_reruns_empty_answer_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            {"question_id": "q1", "answer": "Good answer", "error": None},
            # empty answer, no error — the G03 failure pattern
            {"question_id": "q2", "answer": "", "error": None},
        ],
    )
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    assert "q1" in ids
    assert "q2" not in ids


def test_resume_reruns_whitespace_answer_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "results.jsonl",
        [{"question_id": "q1", "answer": "   \n\t  ", "error": None}],
    )
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    assert "q1" not in ids


def test_resume_reruns_error_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            {"question_id": "q1", "answer": "Good", "error": None},
            {"question_id": "q2", "answer": "", "error": "OpenAI returned an empty generated answer"},
        ],
    )
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    assert "q1" in ids
    assert "q2" not in ids


def test_resume_reruns_missing_answer_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "results.jsonl",
        [{"question_id": "q1", "error": None}],  # no "answer" key
    )
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    assert "q1" not in ids


def test_resume_uses_generated_answer_alias(tmp_path):
    """Rows written before the 'answer' key was canonicalised use generated_answer."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "results.jsonl",
        [{"question_id": "q1", "generated_answer": "A valid answer", "error": None}],
    )
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    assert "q1" in ids


def test_resume_no_duplicate_question_ids(tmp_path):
    """If by some chance the same ID appears twice (once valid, once not), it is
    counted only once — valid."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            {"question_id": "q1", "answer": "", "error": None},
            {"question_id": "q1", "answer": "Valid answer", "error": None},
        ],
    )
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    # The second (valid) row makes q1 eligible to skip.
    assert "q1" in ids
    assert len(ids) == 1


def test_resume_empty_file_returns_empty_set(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text("", encoding="utf-8")
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    assert ids == set()


def test_resume_missing_file_returns_empty_set(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    writer = ResultWriter(run_dir)
    ids = writer.load_existing_question_ids()
    assert ids == set()
