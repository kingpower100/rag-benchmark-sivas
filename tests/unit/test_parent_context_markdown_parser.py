"""Tests for Markdown parent-section parser."""
from __future__ import annotations

import pytest

from src.pipeline1.parent_context.markdown_parser import (
    MAPPING_POLICY_VERSION,
    MARKDOWN_PARENT_PARSER_VERSION,
    MarkdownSection,
    compute_parent_id,
    parse_markdown_sections,
)


DOC_ID = "doc_001"
CTX_ID = "ctx_001"


# ---------------------------------------------------------------------------
# compute_parent_id determinism
# ---------------------------------------------------------------------------

def test_parent_id_is_deterministic():
    id1 = compute_parent_id(DOC_ID, 1, 0, "Introduction")
    id2 = compute_parent_id(DOC_ID, 1, 0, "Introduction")
    assert id1 == id2


def test_parent_id_differs_by_document():
    id1 = compute_parent_id("doc_A", 1, 0, "Intro")
    id2 = compute_parent_id("doc_B", 1, 0, "Intro")
    assert id1 != id2


def test_parent_id_differs_by_level():
    id1 = compute_parent_id(DOC_ID, 1, 0, "Section")
    id2 = compute_parent_id(DOC_ID, 2, 0, "Section")
    assert id1 != id2


def test_parent_id_differs_by_index():
    id1 = compute_parent_id(DOC_ID, 1, 0, "Same Title")
    id2 = compute_parent_id(DOC_ID, 1, 1, "Same Title")
    assert id1 != id2


def test_parent_id_is_hex_string():
    pid = compute_parent_id(DOC_ID, 1, 0, "Intro")
    assert len(pid) == 64
    int(pid, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# parse_markdown_sections — basic cases
# ---------------------------------------------------------------------------

def test_empty_text_returns_empty():
    assert parse_markdown_sections(DOC_ID, CTX_ID, "") == []


def test_no_headings_single_document_section():
    text = "Some plain text without any headings."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert len(sections) == 1
    assert sections[0].parent_title == "[document]"
    assert sections[0].heading_level == 0
    assert sections[0].parent_text.strip() == text.strip()


def test_single_heading():
    text = "# Overview\n\nThis is the overview section."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert len(sections) == 1
    assert sections[0].parent_title == "Overview"
    assert sections[0].heading_level == 1
    assert "overview section" in sections[0].parent_text


def test_two_level_1_headings():
    text = "# Section A\n\nContent A.\n\n# Section B\n\nContent B."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert len(sections) == 2
    assert sections[0].parent_title == "Section A"
    assert sections[1].parent_title == "Section B"
    assert "Content A" in sections[0].parent_text
    assert "Content B" in sections[1].parent_text


def test_nested_subsections_remain_inside_parent():
    text = (
        "# Parent\n\nParent intro.\n\n"
        "## Child\n\nChild content.\n\n"
        "# Next Parent\n\nNext content."
    )
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    # Should have: Parent (contains ## Child), Next Parent
    parent_section = next(s for s in sections if s.parent_title == "Parent")
    assert "Child" in parent_section.parent_text
    assert "Child content" in parent_section.parent_text


def test_content_before_first_heading_is_intro():
    text = "Introductory text.\n\n# First Heading\n\nHeading content."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    titles = [s.parent_title for s in sections]
    assert "[intro]" in titles
    intro = next(s for s in sections if s.parent_title == "[intro]")
    assert "Introductory text" in intro.parent_text


def test_empty_sections_are_skipped():
    text = "# Empty\n\n# Has Content\n\nSome text here."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    titles = [s.parent_title for s in sections]
    assert "Empty" not in titles
    assert "Has Content" in titles


def test_repeated_heading_titles_get_distinct_ids():
    text = "# Overview\n\nFirst.\n\n# Overview\n\nSecond."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert len(sections) == 2
    ids = {s.parent_id for s in sections}
    assert len(ids) == 2  # distinct IDs despite same title


def test_tables_remain_in_section():
    text = "# Data\n\n| Col1 | Col2 |\n|------|------|\n| A    | B    |"
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert len(sections) == 1
    assert "Col1" in sections[0].parent_text


def test_lists_remain_in_section():
    text = "# Steps\n\n- Step 1\n- Step 2\n- Step 3"
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert len(sections) == 1
    assert "Step 1" in sections[0].parent_text


def test_h2_does_not_close_h1_parent():
    text = "# H1\n\n## H2\n\nH2 content.\n\n# Next H1\n\nNext."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    # H1 should contain H2 content
    h1_section = next(s for s in sections if s.parent_title == "H1")
    assert "H2 content" in h1_section.parent_text


def test_h1_closes_previous_h1():
    text = "# First\n\nFirst content.\n\n# Second\n\nSecond content."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    first = next(s for s in sections if s.parent_title == "First")
    assert "Second content" not in first.parent_text


def test_section_indices_are_sequential():
    text = "# A\n\nContent A.\n\n# B\n\nContent B.\n\n# C\n\nContent C."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert [s.section_index for s in sections] == [0, 1, 2]


def test_document_id_and_context_id_stored():
    text = "# Section\n\nContent."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert sections[0].document_id == DOC_ID
    assert sections[0].original_context_id == CTX_ID


def test_inherited_metadata_preserved():
    text = "# Section\n\nContent."
    meta = {"kategorie": "ERP", "doc_key": "abc"}
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text, inherited_metadata=meta)
    assert sections[0].metadata.get("kategorie") == "ERP"


def test_malformed_heading_does_not_crash():
    # Text with unusual characters near heading markers
    text = "##\n\n# Valid Section\n\nContent."
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    assert any(s.parent_title == "Valid Section" for s in sections)


def test_deeply_nested_headings():
    text = (
        "# H1\n\n## H2\n\n### H3\n\nDeep content.\n\n"
        "## H2b\n\nH2b content.\n\n"
        "# H1b\n\nH1b content."
    )
    sections = parse_markdown_sections(DOC_ID, CTX_ID, text)
    h1 = next(s for s in sections if s.parent_title == "H1")
    # H1 should contain H2, H3, H2b
    assert "Deep content" in h1.parent_text
    assert "H2b content" in h1.parent_text


def test_parser_version_constant_exists():
    assert MARKDOWN_PARENT_PARSER_VERSION == "markdown_heading_v1"


def test_mapping_policy_version_constant_exists():
    assert MAPPING_POLICY_VERSION == "deepest_containing_then_overlap_v2"
