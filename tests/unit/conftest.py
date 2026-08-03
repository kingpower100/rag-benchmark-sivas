"""Unit test conftest.

Patches tiktoken.get_encoding so config-schema validation does not require
network access to openaipublic.blob.core.windows.net. The encoding name is
still validated (the mock raises for unknown names), but no BPE file download
is attempted. This restores test-suite functionality on networks with SSL
inspection or restricted egress.
"""
from unittest.mock import MagicMock, patch

import pytest


_KNOWN_TIKTOKEN_ENCODINGS = {
    "cl100k_base",
    "p50k_base",
    "p50k_edit",
    "r50k_base",
    "o200k_base",
    "gpt2",
}


def _mock_get_encoding(name: str) -> MagicMock:
    if name not in _KNOWN_TIKTOKEN_ENCODINGS:
        raise ValueError(f"Unknown encoding: {name!r}")
    return MagicMock(name=f"tiktoken-encoding-{name}")


@pytest.fixture(autouse=True, scope="session")
def patch_tiktoken_network():
    """Prevent tiktoken from attempting to download BPE files during tests."""
    with patch("tiktoken.get_encoding", side_effect=_mock_get_encoding):
        yield
