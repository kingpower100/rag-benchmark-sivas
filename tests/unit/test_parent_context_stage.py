"""Tests for ParentContextStage expansion and deduplication algorithm."""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field

from src.pipeline1.parent_context.markdown_parser import MarkdownSection
from src.pipeline1.parent_context.parent_store import ChildMappingEntry, ParentStore
from src.pipeline1.schemas.config_schema import ParentContextConfig, PipelineConfig
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.parent_context_stage import GenerationContext, ParentContextStage
from src.pipeline1.stages.retrieval_stage import RetrievalRow
from src.pipeline1.schemas.query import QueryRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retrieval_item(chunk_id: str, text: str, score: float = 0.9, doc_id: str = "doc1"):
    from src.pipeline1.schemas.retrieval import RetrievalItem
    return RetrievalItem(
        chunk_id=chunk_id,
        original_context_id=doc_id,
        text=text,
        score=score,
    )


def _make_row(raw_retrieved, retrieved=None):
    return RetrievalRow(
        query=QueryRecord(question_id="q1", question="Test?"),
        raw_retrieved=raw_retrieved,
        raw_dense_retrieved=[],
        raw_bm25_retrieved=[],
        fused_retrieved=[],
        retrieved=retrieved or raw_retrieved[:5],
        retrieval_time_ms=10.0,
        reranker_used=False,
        retrieval_warnings=[],
        retrieval_diagnostics={},
    )


def _make_section(parent_id: str, text: str, doc_id: str = "doc1") -> MarkdownSection:
    return MarkdownSection(
        parent_id=parent_id,
        document_id=doc_id,
        original_context_id=doc_id,
        parent_title=f"Section {parent_id}",
        heading_level=1,
        section_index=0,
        start_char=0,
        end_char=len(text),
        parent_text=text,
        metadata={},
    )


def _make_store(
    parent_ids: list[str],
    parent_texts: list[str],
    chunk_to_parent: dict[str, str | None],
) -> ParentStore:
    sections = [_make_section(pid, txt) for pid, txt in zip(parent_ids, parent_texts)]
    mapping = {
        cid: ChildMappingEntry(
            chunk_id=cid,
            parent_id=pid,
            policy="text_position",
            boundary_spanning=False,
        )
        for cid, pid in chunk_to_parent.items()
    }
    return ParentStore(sections, mapping)


def _cfg(
    enabled: bool = True,
    top_k: int = 5,
    missing_policy: str = "use_child",
    max_parent_tokens: int = 1800,
) -> PipelineConfig:
    return PipelineConfig.model_validate({
        "experiment": {"experiment_id": "test", "output_dir": "data/runs"},
        "data": {"documents_path": "docs.jsonl", "questions_path": "q.jsonl"},
        "chunking": {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 200},
        "embedding": {"provider": "sentence_transformers", "model_name": "intfloat/multilingual-e5-small"},
        "index": {"type": "faiss"},
        "retrieval": {"retriever_type": "dense", "top_k": 5, "fetch_k": 20},
        "reranker": {"enabled": False},
        "generation": {"provider": "ollama", "model_name": "qwen2.5:7b-instruct",
                       "system_prompt": "Q: {question}\nCtx: {context}"},
        "telemetry": {"estimate_cost": False},
        "runtime": {"save_csv": False},
        "parent_context": {
            "enabled": enabled,
            "parent_unit": "markdown_section",
            "deduplicate": True,
            "missing_parent_policy": missing_policy,
            "unique_parent_top_k": top_k,
            "max_parent_tokens": max_parent_tokens,
        },
    })


# ---------------------------------------------------------------------------
# Disabled path
# ---------------------------------------------------------------------------

def test_disabled_passes_through_unchanged():
    cfg = _cfg(enabled=False)
    row = _make_row([_make_retrieval_item("c1", "text")])
    stage = ParentContextStage(cfg, parent_store=None)
    out = stage.run(StageInput({"retrieval_rows": [row]}))
    assert out.retrieval_rows[0] is row
    assert out.diagnostics["parent_context_enabled"] is False


# ---------------------------------------------------------------------------
# Single child → single parent
# ---------------------------------------------------------------------------

def test_one_child_maps_to_one_parent():
    cfg = _cfg()
    store = _make_store(["par1"], ["Full parent text."], {"c1": "par1"})
    row = _make_row([_make_retrieval_item("c1", "child text", score=0.95)])
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [row]}))
    new_row = out.retrieval_rows[0]
    assert len(new_row.generation_contexts) == 1
    gc = new_row.generation_contexts[0]
    assert gc.text == "Full parent text."
    assert gc.parent_id == "par1"
    assert gc.trigger_child_id == "c1"
    assert gc.trigger_child_score == pytest.approx(0.95)
    assert gc.parent_context_expanded is True


# ---------------------------------------------------------------------------
# Deduplication: multiple children → same parent
# ---------------------------------------------------------------------------

def test_two_children_same_parent_deduplicates():
    cfg = _cfg()
    store = _make_store(["par1"], ["Parent text."], {"c1": "par1", "c2": "par1"})
    raw = [
        _make_retrieval_item("c1", "child 1", score=0.9),
        _make_retrieval_item("c2", "child 2", score=0.8),
    ]
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [_make_row(raw)]}))
    new_row = out.retrieval_rows[0]
    assert len(new_row.generation_contexts) == 1  # deduplicated
    diag = new_row.parent_context_diagnostics
    assert diag["duplicate_parent_count"] == 1


# ---------------------------------------------------------------------------
# Multiple parents preserve child ranking order
# ---------------------------------------------------------------------------

def test_multiple_parents_in_rank_order():
    cfg = _cfg()
    store = _make_store(
        ["par_a", "par_b"],
        ["Parent A text.", "Parent B text."],
        {"c1": "par_a", "c2": "par_b"},
    )
    raw = [
        _make_retrieval_item("c1", "child 1", score=0.9),
        _make_retrieval_item("c2", "child 2", score=0.7),
    ]
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [_make_row(raw)]}))
    new_row = out.retrieval_rows[0]
    assert len(new_row.generation_contexts) == 2
    assert new_row.generation_contexts[0].parent_id == "par_a"
    assert new_row.generation_contexts[1].parent_id == "par_b"


# ---------------------------------------------------------------------------
# Continues through 20 candidates until 5 unique parents found
# ---------------------------------------------------------------------------

def test_continues_until_unique_parent_top_k_reached():
    cfg = _cfg(top_k=3)
    # 6 children but only 3 unique parents: par1 (c1,c2), par2 (c3,c4), par3 (c5,c6)
    chunk_to_parent = {"c1": "par1", "c2": "par1", "c3": "par2", "c4": "par2", "c5": "par3", "c6": "par3"}
    store = _make_store(
        ["par1", "par2", "par3"],
        ["P1.", "P2.", "P3."],
        chunk_to_parent,
    )
    raw = [_make_retrieval_item(f"c{i}", f"child {i}", score=1.0 - i * 0.05) for i in range(1, 7)]
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [_make_row(raw)]}))
    new_row = out.retrieval_rows[0]
    assert len(new_row.generation_contexts) == 3
    parent_ids = [gc.parent_id for gc in new_row.generation_contexts]
    assert parent_ids == ["par1", "par2", "par3"]


# ---------------------------------------------------------------------------
# Fewer than unique_parent_top_k valid parents available
# ---------------------------------------------------------------------------

def test_fewer_than_top_k_parents_available():
    cfg = _cfg(top_k=5)
    store = _make_store(["par1", "par2"], ["P1.", "P2."], {"c1": "par1", "c2": "par2"})
    raw = [
        _make_retrieval_item("c1", "child 1"),
        _make_retrieval_item("c2", "child 2"),
    ]
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [_make_row(raw)]}))
    assert len(out.retrieval_rows[0].generation_contexts) == 2  # only 2 available


# ---------------------------------------------------------------------------
# Missing parent with use_child policy
# ---------------------------------------------------------------------------

def test_missing_parent_use_child_fallback():
    cfg = _cfg(missing_policy="use_child")
    store = _make_store([], [], {})  # empty store
    row = _make_row([_make_retrieval_item("c1", "child text")])
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [row]}))
    new_row = out.retrieval_rows[0]
    assert len(new_row.generation_contexts) == 1
    gc = new_row.generation_contexts[0]
    assert gc.text == "child text"
    assert gc.parent_context_expanded is False
    assert new_row.parent_context_diagnostics["parent_fallback_to_child_count"] == 1


# ---------------------------------------------------------------------------
# Missing parent with error policy
# ---------------------------------------------------------------------------

def test_missing_parent_error_policy_raises():
    cfg = _cfg(missing_policy="error")
    store = _make_store([], [], {})  # no mapping
    row = _make_row([_make_retrieval_item("c1", "text")])
    with pytest.raises(RuntimeError):
        ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [row]}))


# ---------------------------------------------------------------------------
# Trigger child score and rank preserved
# ---------------------------------------------------------------------------

def test_best_child_score_and_rank_preserved():
    cfg = _cfg()
    store = _make_store(["par1"], ["Parent."], {"c1": "par1"})
    raw = [_make_retrieval_item("c1", "child", score=0.88)]
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [_make_row(raw)]}))
    gc = out.retrieval_rows[0].generation_contexts[0]
    assert gc.trigger_child_score == pytest.approx(0.88)
    assert gc.trigger_child_rank == 1


# ---------------------------------------------------------------------------
# Diagnostics fields
# ---------------------------------------------------------------------------

def test_diagnostics_contain_required_fields():
    cfg = _cfg()
    store = _make_store(["par1"], ["Parent."], {"c1": "par1", "c2": "par1"})
    raw = [
        _make_retrieval_item("c1", "c1", score=0.9),
        _make_retrieval_item("c2", "c2", score=0.8),
    ]
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [_make_row(raw)]}))
    diag = out.retrieval_rows[0].parent_context_diagnostics
    assert "parent_context_enabled" in diag
    assert "retrieved_child_candidate_count" in diag
    assert "selected_unique_parent_count" in diag
    assert "expanded_parent_ids" in diag
    assert "child_to_parent" in diag
    assert "duplicate_parent_count" in diag
    assert "missing_parent_count" in diag
    assert "parent_fallback_to_child_count" in diag
    assert diag["oversized_parent_policy"] == "prefer_deeper_section"
    assert diag["child_provenance_preserved"] is True


# ---------------------------------------------------------------------------
# Retrieved (P2 eval) vs generation_contexts (P3 eval) separation
# ---------------------------------------------------------------------------

def test_retrieved_unchanged_after_expansion():
    cfg = _cfg()
    store = _make_store(["par1"], ["Big parent text."], {"c1": "par1"})
    child = _make_retrieval_item("c1", "small child text")
    row = _make_row([child], retrieved=[child])
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [row]}))
    new_row = out.retrieval_rows[0]
    # Retrieved (child) list unchanged
    assert new_row.retrieved == [child]
    # Generation context uses parent text
    assert new_row.generation_contexts[0].text == "Big parent text."


# ---------------------------------------------------------------------------
# Oversized parent: prefer_deeper_section uses candidate_parent_ids
# ---------------------------------------------------------------------------

def _make_store_with_candidates(
    sections: list,
    chunk_to_parent: dict,
    chunk_candidates: dict | None = None,
) -> ParentStore:
    """Build a ParentStore with explicit section objects and candidate lists."""
    from src.pipeline1.parent_context.parent_store import ParentStore as _PS
    mapping = {}
    for cid, pid in chunk_to_parent.items():
        candidates = (chunk_candidates or {}).get(cid, [])
        mapping[cid] = ChildMappingEntry(
            chunk_id=cid,
            parent_id=pid,
            policy="deepest_containing",
            boundary_spanning=False,
            mapping_type="fully_contained",
            candidate_parent_ids=candidates,
        )
    return _PS(sections, mapping)


def test_oversized_parent_falls_back_to_candidate():
    from src.pipeline1.parent_context.markdown_parser import MarkdownSection

    big_text = "A" * 9000   # ~2250 tokens — exceeds limit
    small_text = "B" * 100  # ~25 tokens — fits

    big_section = MarkdownSection(
        parent_id="par_h1", document_id="d", original_context_id="d",
        parent_title="H1", heading_level=1, section_index=0,
        start_char=0, end_char=len(big_text), parent_text=big_text, metadata={},
    )
    small_section = MarkdownSection(
        parent_id="par_h2", document_id="d", original_context_id="d",
        parent_title="H2", heading_level=2, section_index=1,
        start_char=0, end_char=len(small_text), parent_text=small_text, metadata={},
    )

    store = _make_store_with_candidates(
        [big_section, small_section],
        chunk_to_parent={"c1": "par_h1"},  # primary is H1 (oversized)
        chunk_candidates={"c1": ["par_h2"]},  # candidate is H2 (fits)
    )

    cfg = _cfg(max_parent_tokens=500)  # limit that big_text exceeds
    row = _make_row([_make_retrieval_item("c1", "child")])
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [row]}))
    gc = out.retrieval_rows[0].generation_contexts[0]
    # Should use H2 (fits), not H1 (oversized)
    assert gc.parent_id == "par_h2"
    assert gc.text == small_text
    assert gc.oversized_parent is False


def test_oversized_parent_kept_when_no_smaller_candidate():
    from src.pipeline1.parent_context.markdown_parser import MarkdownSection

    big_text = "A" * 9000  # ~2250 tokens

    big_section = MarkdownSection(
        parent_id="par_h1", document_id="d", original_context_id="d",
        parent_title="H1", heading_level=1, section_index=0,
        start_char=0, end_char=len(big_text), parent_text=big_text, metadata={},
    )

    store = _make_store_with_candidates(
        [big_section],
        chunk_to_parent={"c1": "par_h1"},
        chunk_candidates={"c1": []},  # no alternatives
    )

    cfg = _cfg(max_parent_tokens=500)
    row = _make_row([_make_retrieval_item("c1", "child")])
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [row]}))
    gc = out.retrieval_rows[0].generation_contexts[0]
    # Keeps H1 (all oversized), marks it as oversized
    assert gc.parent_id == "par_h1"
    assert gc.oversized_parent is True
    assert out.retrieval_rows[0].parent_context_diagnostics["oversized_parent_count"] == 1


def test_diagnostics_include_oversized_parent_count():
    cfg = _cfg()
    store = _make_store(["par1"], ["Parent."], {"c1": "par1"})
    raw = [_make_retrieval_item("c1", "child", score=0.9)]
    out = ParentContextStage(cfg, parent_store=store).run(StageInput({"retrieval_rows": [_make_row(raw)]}))
    diag = out.retrieval_rows[0].parent_context_diagnostics
    assert "oversized_parent_count" in diag
