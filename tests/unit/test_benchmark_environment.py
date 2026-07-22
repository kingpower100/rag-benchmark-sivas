from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import check_benchmark_environment as envcheck
from src.pipeline1.indexing.faiss_index import FaissIndex


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_requirements_constrain_numpy_and_faiss():
    for filename in ("requirements.txt", "requirements-lock.txt"):
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        assert "-c constraints.txt" in text
        assert "numpy==1.26.4" in text
        assert "faiss-cpu==1.8.0.post1" in text
        assert "numpy>=1.20" not in text
        assert "faiss-cpu==1.8.0\n" not in text


def test_constraints_pin_faiss_numpy_abi_pair():
    text = (PROJECT_ROOT / "constraints.txt").read_text(encoding="utf-8")

    assert "numpy==1.26.4" in text
    assert "faiss-cpu==1.8.0.post1" in text


def test_environment_check_rejects_unsupported_numpy_version(monkeypatch):
    class FakeNumpy:
        __version__ = "2.4.6"

    monkeypatch.setattr(envcheck.importlib, "import_module", lambda name: FakeNumpy())

    result = envcheck._check_numpy()

    assert result.ok is False
    assert "unsupported" in result.detail


def test_faiss_runtime_indexflatip_operation():
    results = envcheck.run_checks()
    failures = [result for result in results if not result.ok]

    assert failures == []


def test_production_faiss_index_add_search():
    index = FaissIndex(metric="cosine")
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index.build(vectors)

    scores, indices = index.search(np.array([1.0, 0.0], dtype="float32"), 2)

    assert scores.shape == (2,)
    assert indices.shape == (2,)
    assert int(indices[0]) == 0


def test_environment_check_cli_exits_zero_in_supported_environment():
    result = subprocess.run(
        [sys.executable, "scripts/check_benchmark_environment.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] FAISS index operation" in result.stdout
