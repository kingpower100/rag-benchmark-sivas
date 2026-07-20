import json

import pytest

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
