import json

import numpy as np
import pytest

from src.pipeline1.orchestrator import _validate_embedding_cache, _validate_index_cache


class FakeIndex:
    def __init__(self, ntotal: int, dim: int) -> None:
        self.ntotal = ntotal
        self.dim = dim


def test_stale_embeddings_with_wrong_row_count_fail(tmp_path):
    path = tmp_path / "embeddings.npy"
    path.write_bytes(b"x")
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps({"chunks_key": "chunks", "embedding": {"model_name": "m"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="row count mismatch"):
        _validate_embedding_cache(np.zeros((2, 3)), 3, path, "chunks", {"model_name": "m"})


def test_stale_embeddings_with_wrong_metadata_fail(tmp_path):
    path = tmp_path / "embeddings.npy"
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps({"chunks_key": "old", "embedding": {"model_name": "m"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="metadata mismatch"):
        _validate_embedding_cache(np.zeros((2, 3)), 2, path, "new", {"model_name": "m"})


def test_stale_index_with_wrong_ntotal_fails(tmp_path):
    with pytest.raises(RuntimeError, match="row count mismatch"):
        _validate_index_cache(FakeIndex(ntotal=2, dim=3), 3, np.zeros((3, 3)), tmp_path / "index.faiss")


def test_stale_index_with_wrong_dimension_fails(tmp_path):
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        _validate_index_cache(FakeIndex(ntotal=3, dim=4), 3, np.zeros((3, 3)), tmp_path / "index.faiss")


def test_valid_cache_validation_succeeds(tmp_path):
    path = tmp_path / "embeddings.npy"
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps({"chunks_key": "chunks", "embedding": {"model_name": "m"}}),
        encoding="utf-8",
    )

    _validate_embedding_cache(np.zeros((3, 3)), 3, path, "chunks", {"model_name": "m"})
    _validate_index_cache(FakeIndex(ntotal=3, dim=3), 3, np.zeros((3, 3)), tmp_path / "index.faiss")
