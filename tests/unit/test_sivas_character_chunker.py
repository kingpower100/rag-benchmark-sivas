from __future__ import annotations

import pytest

from src.pipeline1.chunking.sivas_character_chunker import (
    SIVAS_BOUNDARY_RE,
    SIVAS_CHARACTER_CHUNKER_VERSION,
    SivasCharacterChunker,
)
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.stages.chunking_stage import ChunkingStage
from src.pipeline1.utils.hashing import stable_hash_dict


def _doc(text: str, doc_id: str = "d1", metadata: dict | None = None) -> DocumentRecord:
    return DocumentRecord(
        document_id=doc_id,
        original_context_id=doc_id,
        text=text,
        metadata=metadata or {},
    )


def _chunks(text: str, max_chars: int = 2048):
    return SivasCharacterChunker(max_chars=max_chars).chunk_documents([_doc(text)])


def _assert_exact_spans(original_text: str, chunks) -> None:
    assert "".join(chunk.text for chunk in chunks) == original_text
    if not chunks:
        assert original_text == ""
        return
    assert chunks[0].chunk_start == 0
    assert chunks[-1].chunk_end == len(original_text)
    for chunk in chunks:
        assert chunk.text == original_text[chunk.chunk_start:chunk.chunk_end]
        assert chunk.metadata["start_char"] == chunk.chunk_start
        assert chunk.metadata["end_char"] == chunk.chunk_end
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.chunk_end == current.chunk_start


def _pipeline_config(max_chunk_tokens: int = 1800, chunk_overlap: int = 0) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {
                "strategy": "sivas_character",
                "chunk_size": 2048,
                "chunk_overlap": chunk_overlap,
                "max_chunk_chars": 2048,
                "max_chunk_tokens": max_chunk_tokens,
                "oversized_chunk_policy": "warn",
            },
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 2},
            "reranker": {"enabled": False},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {"estimate_cost": False},
            "runtime": {"resume": False, "overwrite": True},
        }
    )


def test_boundary_regex_is_exact_requested_pattern():
    assert SIVAS_BOUNDARY_RE.pattern == r"(?<=[.!?;:])\s+|\n\n|\n(?=#{1,6}\s)|\n(?=-\s)"


def test_exact_reconstruction_substrings_and_no_gaps_or_overlap():
    original_text = (
        "  Intro sentence.  Next sentence!\n\n"
        "Before heading\n# Heading\n"
        "Lead-in:\n- bullet one\n- bullet two\n"
        "German: Größe, Straße, Übermaß?  Done.   "
    )
    chunks = _chunks(original_text, max_chars=45)
    assert len(chunks) > 1
    _assert_exact_spans(original_text, chunks)


def test_whitespace_preservation():
    original_text = (
        "  Leading whitespace.  Multiple spaces after punctuation.\n\n"
        "Line before heading\n# Heading\n"
        "Line before bullet\n- bullet\n"
        "Trailing whitespace.   "
    )
    chunks = _chunks(original_text, max_chars=80)
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == original_text
    assert "\n\n" in reconstructed
    assert "\n# Heading" in reconstructed
    assert "\n- bullet" in reconstructed
    assert ".  Multiple" in reconstructed
    assert reconstructed.startswith("  Leading")
    assert reconstructed.endswith("   ")


def test_exact_2048_character_ceiling_is_accepted():
    original_text = ("A" * 2046) + ". "
    chunks = _chunks(original_text, max_chars=2048)
    assert len(chunks) == 1
    assert len(chunks[0].text) == 2048
    _assert_exact_spans(original_text, chunks)


def test_next_span_starts_new_chunk_at_2049_characters():
    original_text = ("A" * 2046) + ". B"
    chunks = _chunks(original_text, max_chars=2048)
    assert [len(chunk.text) for chunk in chunks] == [2048, 1]
    assert chunks[1].text == "B"
    _assert_exact_spans(original_text, chunks)


def test_unicode_counts_python_characters_not_utf8_bytes():
    first = ("Ä" * 2046) + ". "
    second = "ö"
    original_text = first + second
    chunks = _chunks(original_text, max_chars=2048)
    assert len(first) == 2048
    assert len(first.encode("utf-8")) > 2048
    assert [len(chunk.text) for chunk in chunks] == [2048, 1]
    _assert_exact_spans(original_text, chunks)


def test_oversized_indivisible_segment_is_kept_whole_and_warns():
    original_text = "X" * 2049
    with pytest.warns(RuntimeWarning, match="oversized indivisible source span"):
        chunks = _chunks(original_text, max_chars=2048)
    assert len(chunks) == 1
    assert chunks[0].text == original_text
    assert chunks[0].metadata["oversized_chunk"] is True
    assert chunks[0].metadata["oversized_chunk_policy"] == "warn_keep_whole"
    _assert_exact_spans(original_text, chunks)


def test_empty_input_is_deterministic():
    assert _chunks("", max_chars=2048) == []


def test_max_chunk_tokens_does_not_change_sivas_texts_or_boundaries():
    original_text = ("A" * 2046) + ". B"
    cfg_a = _pipeline_config(max_chunk_tokens=1)
    cfg_b = _pipeline_config(max_chunk_tokens=9999)
    chunker_a = ChunkingStage(cfg_a, None, None, None).build_chunker()
    chunker_b = ChunkingStage(cfg_b, None, None, None).build_chunker()

    chunks_a = chunker_a.chunk_documents([_doc(original_text)])
    chunks_b = chunker_b.chunk_documents([_doc(original_text)])

    assert [(c.text, c.chunk_start, c.chunk_end) for c in chunks_a] == [
        (c.text, c.chunk_start, c.chunk_end) for c in chunks_b
    ]


def test_chunk_overlap_config_is_not_applied_to_sivas_character():
    original_text = ("A" * 2046) + ". B"
    cfg = _pipeline_config(chunk_overlap=64)
    chunks = ChunkingStage(cfg, None, None, None).build_chunker().chunk_documents([_doc(original_text)])
    assert cfg.chunking.chunk_overlap == 64
    _assert_exact_spans(original_text, chunks)


def test_sivas_character_rejects_split_oversized_policy():
    payload = _pipeline_config().model_dump()
    payload["chunking"]["oversized_chunk_policy"] = "split"
    with pytest.raises(ValueError, match="only supports chunking.oversized_chunk_policy='warn'"):
        PipelineConfig.model_validate(payload)


def test_chunker_version_changes_chunk_cache_key_from_previous_implementation():
    cfg = _pipeline_config()
    current = stable_hash_dict(
        {
            "documents_fingerprint": "docs",
            "chunking": cfg.chunking.model_dump(),
            "chunker_versions": {"chunker_implementation": SIVAS_CHARACTER_CHUNKER_VERSION},
        }
    )
    previous = stable_hash_dict(
        {
            "documents_fingerprint": "docs",
            "chunking": cfg.chunking.model_dump(),
            "chunker_versions": {"chunker_implementation": "sivas_character_v1"},
        }
    )
    assert current != previous
