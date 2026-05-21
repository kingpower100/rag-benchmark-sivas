import json

import pytest

from src.pipeline1.orchestrator import _prepare_run_dir, _validate_resume_compatible


def test_resume_with_same_manifest_succeeds(tmp_path):
    payload = {
        "experiment_id": "exp",
        "config_hash": "cfg",
        "documents_fingerprint": "docs",
        "cache_keys": {"chunks": "c", "embeddings": "e", "index": "i"},
        "generation": {"model_name": "m"},
        "prompt_template_version": "p",
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resume_compatibility": payload}), encoding="utf-8")

    _validate_resume_compatible(tmp_path, payload)


def test_resume_with_changed_config_hash_fails(tmp_path):
    previous = {
        "experiment_id": "exp",
        "config_hash": "old",
        "documents_fingerprint": "docs",
        "cache_keys": {"chunks": "c", "embeddings": "e", "index": "i"},
        "generation": {"model_name": "m"},
        "prompt_template_version": "p",
    }
    current = {**previous, "config_hash": "new"}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resume_compatibility": previous}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="config_hash"):
        _validate_resume_compatible(tmp_path, current)


def test_resume_with_changed_chunk_cache_key_fails(tmp_path):
    previous = {
        "experiment_id": "exp",
        "config_hash": "cfg",
        "documents_fingerprint": "docs",
        "cache_keys": {"chunks": "old", "embeddings": "e", "index": "i"},
        "generation": {"model_name": "m"},
        "prompt_template_version": "p",
    }
    current = {**previous, "cache_keys": {"chunks": "new", "embeddings": "e", "index": "i"}}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resume_compatibility": previous}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="cache_keys"):
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
