"""Unit tests for host resolution in scripts/index_elasticsearch.py.

Tests the three cases:
  1. host_env set + env var present → uses env var value
  2. host_env set + env var missing  → raises / exits with clear message
  3. host_env null                   → falls back to cfg.retrieval.bm25.host
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build minimal cfg stubs without touching the real schema
# ---------------------------------------------------------------------------

def _bm25_cfg(host="http://localhost:9200", host_env=None):
    return SimpleNamespace(
        host=host,
        host_env=host_env,
        index_name="rag_benchmark_chunks",
        k1=1.5,
        b=0.75,
        analyzer="german",
    )


def _cfg(bm25):
    return SimpleNamespace(
        data=SimpleNamespace(documents_path="data/raw/docs.jsonl"),
        retrieval=SimpleNamespace(bm25=bm25),
    )


# ---------------------------------------------------------------------------
# The resolution logic extracted verbatim from the script so we can unit-test
# it without importing the full script (which calls sys.path.insert at module
# level and pulls in pipeline1 source).
# ---------------------------------------------------------------------------

def _resolve_host(bm25_cfg) -> str:
    """Mirror of the host-resolution block in scripts/index_elasticsearch.py."""
    host_env = bm25_cfg.host_env
    if host_env:
        host = os.environ.get(host_env, "").strip()
        if not host:
            raise RuntimeError(
                f"bm25.host_env is set to '{host_env}' but the environment "
                f"variable is missing or empty."
            )
        return host
    return bm25_cfg.host


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_host_env_resolves_to_env_var_value():
    """host_env='ELASTICSEARCH_URL' → reads the env var (e.g. localhost:9201)."""
    bm25 = _bm25_cfg(host="http://localhost:9200", host_env="ELASTICSEARCH_URL")
    with patch.dict(os.environ, {"ELASTICSEARCH_URL": "http://localhost:9201"}):
        resolved = _resolve_host(bm25)
    assert resolved == "http://localhost:9201"


def test_host_env_missing_env_var_raises():
    """host_env set but env var not exported → RuntimeError naming the variable."""
    bm25 = _bm25_cfg(host_env="ELASTICSEARCH_URL")
    env_without_key = {k: v for k, v in os.environ.items() if k != "ELASTICSEARCH_URL"}
    with patch.dict(os.environ, env_without_key, clear=True):
        with pytest.raises(RuntimeError, match="ELASTICSEARCH_URL"):
            _resolve_host(bm25)


def test_host_env_empty_string_raises():
    """host_env set but env var is empty string → RuntimeError."""
    bm25 = _bm25_cfg(host_env="ELASTICSEARCH_URL")
    with patch.dict(os.environ, {"ELASTICSEARCH_URL": "   "}):
        with pytest.raises(RuntimeError, match="ELASTICSEARCH_URL"):
            _resolve_host(bm25)


def test_host_env_null_falls_back_to_explicit_host():
    """host_env=None → uses cfg.retrieval.bm25.host directly."""
    bm25 = _bm25_cfg(host="http://myserver:9200", host_env=None)
    # Ensure ELASTICSEARCH_URL is absent to confirm it is never consulted
    env_without_key = {k: v for k, v in os.environ.items() if k != "ELASTICSEARCH_URL"}
    with patch.dict(os.environ, env_without_key, clear=True):
        resolved = _resolve_host(bm25)
    assert resolved == "http://myserver:9200"


def test_no_localhost_9200_hardcoded_in_script():
    """The script source must not contain a bare 'localhost:9200' string
    outside of comments or the cfg.bm25.host default fallback."""
    import re
    script = (
        __file__.replace("tests/unit", "scripts")
                .replace("test_index_elasticsearch_host_resolution.py", "index_elasticsearch.py")
    )
    # Normalise path separators
    from pathlib import Path
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "index_elasticsearch.py"
    source = script_path.read_text(encoding="utf-8")

    # Allow the fallback `host = bm25_cfg.host` but reject any literal URL
    hardcoded = re.findall(r'["\'](http://localhost:9200)["\']', source)
    assert hardcoded == [], (
        f"Hardcoded localhost:9200 URLs found in script: {hardcoded}"
    )
