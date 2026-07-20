"""Integration tests for parent-context: config loading, P2/P3 compatibility, store round-trip."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.pipeline1.parent_context.markdown_parser import parse_markdown_sections
from src.pipeline1.parent_context.parent_store import ChildMappingEntry, ParentStore
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline3.stages.validation_stage import _extract_context_texts


# ---------------------------------------------------------------------------
# C00 and C03 config parsing
# ---------------------------------------------------------------------------

def test_c00_config_parses():
    from src.pipeline1.config_loader import load_pipeline_config_payload
    from src.pipeline1.schemas.config_schema import PipelineConfig

    project_root = Path(__file__).resolve().parents[2]
    c00_path = project_root / "configs" / "pipeline1" / "final_experiments" / "C00_sentence512_baseline.yaml"
    if not c00_path.exists():
        pytest.skip("C00 config not found")

    payload = load_pipeline_config_payload(str(c00_path), validate_unique_experiment_id=False)
    cfg = PipelineConfig.model_validate(payload)
    assert cfg.experiment.experiment_id == "C00"
    assert cfg.parent_context.enabled is False


def test_c03_config_parses():
    from src.pipeline1.config_loader import load_pipeline_config_payload
    from src.pipeline1.schemas.config_schema import PipelineConfig

    project_root = Path(__file__).resolve().parents[2]
    c03_path = project_root / "configs" / "pipeline1" / "final_experiments" / "C03_parent_context.yaml"
    if not c03_path.exists():
        pytest.skip("C03 config not found")

    payload = load_pipeline_config_payload(str(c03_path), validate_unique_experiment_id=False)
    cfg = PipelineConfig.model_validate(payload)
    assert cfg.experiment.experiment_id == "C03"
    assert cfg.parent_context.enabled is True
    assert cfg.parent_context.parent_unit == "markdown_section"
    assert cfg.parent_context.unique_parent_top_k == 5
    # C03 must share chunking config with C00
    assert cfg.chunking.strategy == "sentence"
    assert cfg.chunking.chunk_size == 512
    assert cfg.chunking.chunk_overlap == 200


def test_c03_and_c00_share_chunking_config():
    from src.pipeline1.config_loader import load_pipeline_config_payload
    from src.pipeline1.schemas.config_schema import PipelineConfig

    project_root = Path(__file__).resolve().parents[2]
    c00_path = project_root / "configs" / "pipeline1" / "final_experiments" / "C00_sentence512_baseline.yaml"
    c03_path = project_root / "configs" / "pipeline1" / "final_experiments" / "C03_parent_context.yaml"
    if not c00_path.exists() or not c03_path.exists():
        pytest.skip("C00/C03 configs not found")

    c00 = PipelineConfig.model_validate(
        load_pipeline_config_payload(str(c00_path), validate_unique_experiment_id=False)
    )
    c03 = PipelineConfig.model_validate(
        load_pipeline_config_payload(str(c03_path), validate_unique_experiment_id=False)
    )
    assert c00.chunking.model_dump() == c03.chunking.model_dump()
    assert c00.embedding.model_dump() == c03.embedding.model_dump()
    assert c00.retrieval.model_dump() == c03.retrieval.model_dump()


# ---------------------------------------------------------------------------
# ParentStore round-trip (save/load)
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, text: str) -> DocumentRecord:
    return DocumentRecord(
        document_id=doc_id,
        original_context_id=doc_id,
        text=text,
        metadata={"doc_key": doc_id},
    )


def _make_chunk(chunk_id: str, doc_id: str, text: str, chunk_start: int = 0) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=doc_id,
        original_context_id=doc_id,
        text=text,
        chunk_start=chunk_start,
        chunk_end=chunk_start + 1,
        metadata={"doc_id": doc_id},
    )


def test_parent_store_save_load_round_trip():
    from src.pipeline1.parent_context.markdown_parser import MarkdownSection

    sections = [
        MarkdownSection(
            parent_id="abc123",
            document_id="doc1",
            original_context_id="doc1",
            parent_title="Section",
            heading_level=1,
            section_index=0,
            start_char=0,
            end_char=100,
            parent_text="Section text.",
            metadata={},
        )
    ]
    mapping = {
        "chunk1": ChildMappingEntry("chunk1", "abc123", "text_position", False)
    }
    store = ParentStore(sections, mapping)

    with tempfile.TemporaryDirectory() as tmpdir:
        store_dir = Path(tmpdir) / "store"
        store.save(store_dir)

        loaded = ParentStore.load(store_dir)
        assert "abc123" in loaded
        assert loaded.get("abc123").parent_title == "Section"
        assert loaded.resolve_parent_id("chunk1") == "abc123"


def test_parent_store_validate_passes_with_consistent_data():
    from src.pipeline1.parent_context.markdown_parser import MarkdownSection

    sections = [
        MarkdownSection(
            parent_id="p1",
            document_id="doc1",
            original_context_id="doc1",
            parent_title="Title",
            heading_level=1,
            section_index=0,
            start_char=0,
            end_char=10,
            parent_text="Text.",
            metadata={},
        )
    ]
    mapping = {"c1": ChildMappingEntry("c1", "p1", "text_position", False)}
    store = ParentStore(sections, mapping)
    errors = store.validate()
    assert errors == []


def test_parent_store_validate_detects_broken_reference():
    from src.pipeline1.parent_context.markdown_parser import MarkdownSection

    sections = []  # no sections
    mapping = {"c1": ChildMappingEntry("c1", "nonexistent_parent", "text_position", False)}
    store = ParentStore(sections, mapping)
    errors = store.validate()
    assert len(errors) == 1


def test_parent_store_build_no_headings_single_section():
    doc = _make_doc("d1", "Plain text without headings.")
    chunk = _make_chunk("c1", "d1", "Plain text without headings.")
    store = ParentStore.build([doc], [chunk])
    pid = store.resolve_parent_id("c1")
    assert pid is not None
    section = store.get(pid)
    assert section.parent_title == "[document]"


def test_parent_store_fingerprint_is_deterministic():
    fp1 = ParentStore.compute_fingerprint("doc_hash", "chunks_key", {"strategy": "sentence"})
    fp2 = ParentStore.compute_fingerprint("doc_hash", "chunks_key", {"strategy": "sentence"})
    assert fp1 == fp2


def test_parent_store_fingerprint_changes_with_chunks_key():
    fp1 = ParentStore.compute_fingerprint("doc_hash", "key_A", {"strategy": "sentence"})
    fp2 = ParentStore.compute_fingerprint("doc_hash", "key_B", {"strategy": "sentence"})
    assert fp1 != fp2


def test_new_fingerprint_differs_from_v1_fingerprint():
    """Old v1 parent store fingerprint must not match the new v2 fingerprint."""
    from src.pipeline1.utils.hashing import stable_hash_dict
    old_v1_fp = stable_hash_dict({
        "format_version": "1.0",
        "parser_version": "markdown_heading_v1",
        "boundary_policy_version": "largest_overlap_v1",
        "metadata_schema_version": "1.0",
        "parent_unit": "markdown_section",
        "documents_fingerprint": "doc_hash",
        "chunks_key": "chunks_key",
        "chunking": {"strategy": "sentence"},
    })
    new_fp = ParentStore.compute_fingerprint("doc_hash", "chunks_key", {"strategy": "sentence"})
    assert old_v1_fp != new_fp, "v1 parent store must be invalidated after mapping algorithm upgrade"


# ---------------------------------------------------------------------------
# Child-to-parent mapping: deterministic across repeated builds
# ---------------------------------------------------------------------------

def test_mapping_stable_across_repeated_builds():
    doc = _make_doc("d1", "# Section A\n\nContent A.\n\n# Section B\n\nContent B.")
    chunk = _make_chunk("c1", "d1", "Content A.", 0)
    store1 = ParentStore.build([doc], [chunk])
    store2 = ParentStore.build([doc], [chunk])
    assert store1.resolve_parent_id("c1") == store2.resolve_parent_id("c1")


def test_child_maps_to_correct_parent_section():
    text = "# Section A\n\nChunk A content.\n\n# Section B\n\nChunk B content."
    doc = _make_doc("d1", text)
    chunk_a = _make_chunk("ca", "d1", "Chunk A content.", 0)
    chunk_b = _make_chunk("cb", "d1", "Chunk B content.", 1)
    store = ParentStore.build([doc], [chunk_a, chunk_b])

    pid_a = store.resolve_parent_id("ca")
    pid_b = store.resolve_parent_id("cb")
    assert pid_a is not None
    assert pid_b is not None
    assert pid_a != pid_b

    assert "Chunk A" in store.get(pid_a).parent_text
    assert "Chunk B" in store.get(pid_b).parent_text


# ---------------------------------------------------------------------------
# Pipeline 3 context extraction: generation_context_texts takes priority
# ---------------------------------------------------------------------------

def test_p3_uses_generation_context_texts_when_present():
    row = {
        "generation_context_texts": ["Expanded parent section text."],
        "retrieved_context_texts": ["Child chunk text."],
    }
    texts = _extract_context_texts(row)
    assert texts == ["Expanded parent section text."]


def test_p3_falls_back_to_retrieved_context_texts_when_no_generation_contexts():
    row = {
        "retrieved_context_texts": ["Child chunk text."],
    }
    texts = _extract_context_texts(row)
    assert texts == ["Child chunk text."]


def test_p3_falls_back_when_generation_context_texts_empty():
    row = {
        "generation_context_texts": [],
        "retrieved_context_texts": ["Child chunk text."],
    }
    texts = _extract_context_texts(row)
    assert texts == ["Child chunk text."]


def test_p3_old_run_without_generation_context_texts_still_works():
    row = {
        "retrieved_chunk_texts": ["Old format text."],
    }
    texts = _extract_context_texts(row)
    assert texts == ["Old format text."]


# ---------------------------------------------------------------------------
# OutputRecord: generation_context_texts exported
# ---------------------------------------------------------------------------

def test_output_record_exports_generation_context_texts():
    from src.pipeline1.schemas.output_record import OutputRecord

    record = OutputRecord(
        experiment_id="C03",
        question_id="q1",
        question="Test?",
        generated_answer="Answer.",
        retrieved_chunk_ids=["c1"],
        retrieved_original_context_ids=["doc1"],
        retrieved_context_texts=["child text"],
        retrieval_scores=[0.9],
        top_k=5,
        chunking_strategy="sentence",
        chunk_size=512,
        chunk_overlap=200,
        embedding_model="test",
        retriever_type="dense",
        reranker_used=False,
        llm_model="qwen2.5:7b-instruct",
        retrieval_time_ms=10.0,
        generation_time_ms=100.0,
        total_latency_ms=110.0,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        generation_context_texts=["Full parent section text."],
        parent_context_enabled=True,
    )
    exported = record.to_export_record()
    assert exported["generation_context_texts"] == ["Full parent section text."]
    assert exported["parent_context_enabled"] is True
    # Retrieved child data still present for P2
    assert "retrieved_original_context_ids" in exported
    assert exported["retrieved_original_context_ids"] == ["doc1"]
