"""Unit tests for SivasCharacterChunker.

Covers:
- Exact SIVAS boundary regex behavior
- Semicolon and colon boundaries
- Blank-line (paragraph) boundaries
- Markdown heading boundaries
- Markdown bullet-item boundaries
- 2048-character accumulation (character count, not words/tokens)
- Boundary overflow (segment itself exceeds max_chars)
- Final chunk retention
- Zero overlap (each segment belongs to exactly one chunk)
- Metadata preservation
"""
from __future__ import annotations

import pytest

from src.pipeline1.chunking.sivas_character_chunker import (
    SIVAS_BOUNDARY_RE,
    SivasCharacterChunker,
)
from src.pipeline1.schemas.document import DocumentRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str, doc_id: str = "d1", metadata: dict | None = None) -> DocumentRecord:
    return DocumentRecord(
        document_id=doc_id,
        original_context_id=doc_id,
        text=text,
        metadata=metadata or {},
    )


def _chunks(text: str, max_chars: int = 2048, metadata: dict | None = None):
    chunker = SivasCharacterChunker(max_chars=max_chars)
    doc = _doc(text, metadata=metadata)
    return chunker.chunk_documents([doc])


# ---------------------------------------------------------------------------
# Test 1: Exact regex — sentence boundaries after . ! ? ; :
# ---------------------------------------------------------------------------

class TestExactRegexBehavior:
    def test_period_boundary(self):
        chunks = _chunks("Hello. World.", max_chars=2048)
        assert len(chunks) == 1
        assert "Hello." in chunks[0].text

    def test_exclamation_boundary_splits_at_limit(self):
        a = "A" * 1000 + "!"
        b = "B" * 1000
        chunks = _chunks(f"{a} {b}", max_chars=1500)
        assert len(chunks) == 2
        assert chunks[0].text.endswith("!")
        assert chunks[1].text == b

    def test_question_boundary(self):
        seg1 = "Is this correct?"
        seg2 = "Yes it is."
        text = f"{seg1} {seg2}"
        chunks = _chunks(text, max_chars=len(seg1) + 1)
        assert len(chunks) == 2

    def test_semicolon_boundary(self):
        seg1 = "First clause; "
        seg2 = "second clause."
        text = "First clause; second clause."
        chunks = _chunks(text, max_chars=len("First clause;") + 1)
        assert len(chunks) == 2

    def test_colon_boundary(self):
        seg1 = "Title:"
        seg2 = "body text here."
        text = "Title: body text here."
        chunks = _chunks(text, max_chars=len("Title:") + 1)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Test 2: Semicolon and colon as explicit boundary characters
# ---------------------------------------------------------------------------

class TestSemicolonAndColonBoundaries:
    def test_semicolon_splits_when_over_limit(self):
        long_a = "X" * 500 + ";"
        long_b = "Y" * 500
        text = f"{long_a} {long_b}"
        chunks = _chunks(text, max_chars=600)
        assert len(chunks) == 2
        assert chunks[0].text.endswith(";")

    def test_colon_splits_when_over_limit(self):
        long_a = "Z" * 500 + ":"
        long_b = "W" * 500
        text = f"{long_a} {long_b}"
        chunks = _chunks(text, max_chars=600)
        assert len(chunks) == 2
        assert chunks[0].text.endswith(":")

    def test_semicolon_and_colon_within_limit_stay_together(self):
        text = "Alpha: beta; gamma."
        chunks = _chunks(text, max_chars=2048)
        assert len(chunks) == 1
        assert "Alpha:" in chunks[0].text
        assert "beta;" in chunks[0].text
        assert "gamma." in chunks[0].text


# ---------------------------------------------------------------------------
# Test 3: Blank-line boundaries (\n\n)
# ---------------------------------------------------------------------------

class TestBlankLineBoundaries:
    def test_blank_line_splits(self):
        text = "Paragraph one.\n\nParagraph two."
        chunks = _chunks(text, max_chars=len("Paragraph one.") + 1)
        assert len(chunks) == 2
        assert "Paragraph one." in chunks[0].text
        assert "Paragraph two." in chunks[1].text

    def test_blank_line_within_limit_stays_together(self):
        text = "Short.\n\nAlso short."
        chunks = _chunks(text, max_chars=2048)
        assert len(chunks) == 1

    def test_multiple_blank_lines_produce_multiple_splits(self):
        parts = [f"Para {i}." for i in range(5)]
        text = "\n\n".join(parts)
        chunks = _chunks(text, max_chars=len(parts[0]) + 1)
        assert len(chunks) == 5


# ---------------------------------------------------------------------------
# Test 4: Markdown heading boundaries
# ---------------------------------------------------------------------------

class TestMarkdownHeadingBoundaries:
    def test_h1_boundary(self):
        text = "Intro text.\n# Main Heading\nBody text."
        chunks = _chunks(text, max_chars=len("Intro text.") + 5)
        heading_chunk = next(c for c in chunks if "# Main Heading" in c.text)
        assert heading_chunk is not None

    def test_h2_boundary(self):
        text = "Some text.\n## Section\nContent here."
        chunks = _chunks(text, max_chars=20)
        texts = [c.text for c in chunks]
        assert any("## Section" in t for t in texts)

    def test_heading_boundary_not_triggered_without_leading_newline(self):
        text = "No boundary before# this heading."
        chunks = _chunks(text, max_chars=2048)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Test 5: Markdown bullet-item boundaries
# ---------------------------------------------------------------------------

class TestMarkdownBulletBoundaries:
    def test_bullet_item_boundary(self):
        text = "Preamble text.\n- Item one\n- Item two"
        chunks = _chunks(text, max_chars=len("Preamble text.") + 5)
        bullet_chunks = [c for c in chunks if "Item" in c.text]
        assert len(bullet_chunks) >= 1

    def test_bullet_within_limit_stays_together(self):
        text = "Short.\n- item"
        chunks = _chunks(text, max_chars=2048)
        assert len(chunks) == 1

    def test_multiple_bullets_split_at_limit(self):
        seg1 = "A" * 100
        seg2 = "- " + "B" * 100
        text = f"{seg1}\n{seg2}"
        chunks = _chunks(text, max_chars=120)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Test 6: 2048-character accumulation (characters, not words/tokens)
# ---------------------------------------------------------------------------

class TestCharacterAccumulation:
    def test_accumulates_up_to_2048_chars(self):
        seg_size = 500
        seg1 = "A" * seg_size + "."
        seg2 = "B" * seg_size + "."
        seg3 = "C" * seg_size + "."
        seg4 = "D" * seg_size + "."
        # 4 * (501 + 1 sep) = ~2008 chars combined — should all fit in one chunk
        text = f"{seg1} {seg2} {seg3} {seg4}"
        chunks = _chunks(text, max_chars=2048)
        assert len(chunks) == 1

    def test_character_not_word_boundary(self):
        # 5 segments of 20 words each (~100 chars per segment, ~505 chars total).
        # Under a word-count limit of 20 each would be its own chunk;
        # under a 2048-character limit all five fit together.
        seg = ("word " * 20).strip() + "."  # ~100 chars, ends with "."
        # Use paragraph breaks (\n\n) as explicit segment boundaries.
        text = "\n\n".join([seg] * 5)
        # All 5 segments together are well under 2048 chars → single chunk.
        chunks_large = _chunks(text, max_chars=2048)
        assert len(chunks_large) == 1
        # With a tight character limit each segment becomes its own chunk.
        chunks_small = _chunks(text, max_chars=110)
        assert len(chunks_small) == 5

    def test_exactly_at_limit_stays_together(self):
        # Two segments that together equal exactly max_chars after join
        max_chars = 100
        # seg1 + " " + seg2 == 100
        seg1 = "A" * 50 + "."
        seg2 = "B" * 48  # 51 + 1 + 48 = 100
        text = f"{seg1} {seg2}"
        chunks = _chunks(text, max_chars=max_chars)
        assert len(chunks) == 1

    def test_one_over_limit_splits(self):
        max_chars = 100
        seg1 = "A" * 50 + "."
        seg2 = "B" * 49  # 51 + 1 + 49 = 101 — exceeds limit
        text = f"{seg1} {seg2}"
        chunks = _chunks(text, max_chars=max_chars)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Test 7: Boundary overflow (single segment exceeds max_chars)
# ---------------------------------------------------------------------------

class TestBoundaryOverflow:
    def test_oversized_segment_becomes_own_chunk(self):
        oversized = "X" * 3000
        chunks = _chunks(oversized, max_chars=2048)
        assert len(chunks) == 1
        assert len(chunks[0].text) == 3000

    def test_oversized_segment_after_normal_chunk(self):
        normal = "Normal text."
        oversized = "Y" * 3000
        text = f"{normal}\n\n{oversized}"
        chunks = _chunks(text, max_chars=2048)
        assert len(chunks) == 2
        assert chunks[0].text == normal
        assert len(chunks[1].text) == 3000

    def test_oversized_segment_before_normal_chunk(self):
        oversized = "Z" * 3000
        normal = "Normal."
        text = f"{oversized}\n\n{normal}"
        chunks = _chunks(text, max_chars=2048)
        assert len(chunks) == 2
        assert len(chunks[0].text) == 3000
        assert chunks[1].text == normal


# ---------------------------------------------------------------------------
# Test 8: Final chunk retention
# ---------------------------------------------------------------------------

class TestFinalChunkRetention:
    def test_final_chunk_is_retained(self):
        seg1 = "A" * 1500 + "."
        seg2 = "B" * 100
        text = f"{seg1} {seg2}"
        chunks = _chunks(text, max_chars=1600)
        last = chunks[-1]
        assert "B" * 100 in last.text

    def test_single_segment_document_produces_one_chunk(self):
        text = "Only one sentence here."
        chunks = _chunks(text, max_chars=2048)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_empty_document_produces_no_chunks(self):
        chunks = _chunks("", max_chars=2048)
        assert chunks == []

    def test_whitespace_only_document_produces_no_chunks(self):
        chunks = _chunks("   \n\n  \t  ", max_chars=2048)
        assert chunks == []


# ---------------------------------------------------------------------------
# Test 9: Zero overlap
# ---------------------------------------------------------------------------

class TestZeroOverlap:
    def test_no_segment_appears_in_two_chunks(self):
        segs = [f"Sentence {i}." for i in range(10)]
        text = " ".join(segs)
        chunks = _chunks(text, max_chars=30)
        seen_texts: set[str] = set()
        for chunk in chunks:
            assert chunk.text not in seen_texts, "Chunk text repeated — overlap detected"
            seen_texts.add(chunk.text)

    def test_all_text_covered(self):
        segs = ["Alpha.", "Beta!", "Gamma?", "Delta;", "Epsilon:"]
        text = " ".join(segs)
        chunker = SivasCharacterChunker(max_chars=10)
        doc = _doc(text)
        chunks = chunker.chunk_documents([doc])
        combined = " ".join(c.text for c in chunks)
        for seg in segs:
            assert seg in combined or seg.rstrip(".!?;:") in combined

    def test_chunk_count_independent_of_overlap_parameter(self):
        segs = ["Long sentence one.", "Long sentence two.", "Long sentence three."]
        text = " ".join(segs)
        chunker = SivasCharacterChunker(max_chars=25)
        doc = _doc(text)
        chunks = chunker.chunk_documents([doc])
        # No sentence should appear in two consecutive chunks
        for i in range(len(chunks) - 1):
            assert chunks[i].text != chunks[i + 1].text


# ---------------------------------------------------------------------------
# Test 10: Metadata preservation
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    def test_document_id_preserved(self):
        chunker = SivasCharacterChunker()
        doc = _doc("Text. More text.", doc_id="doc_42")
        chunks = chunker.chunk_documents([doc])
        for chunk in chunks:
            assert chunk.document_id == "doc_42"

    def test_original_context_id_preserved(self):
        chunker = SivasCharacterChunker()
        doc = _doc("Text. More text.", doc_id="doc_99")
        chunks = chunker.chunk_documents([doc])
        for chunk in chunks:
            assert chunk.original_context_id == "doc_99"

    def test_custom_metadata_field_preserved(self):
        meta = {"kategorie": "Einkauf", "doc_id": "abc", "wissensart": "FAQ"}
        chunker = SivasCharacterChunker()
        doc = _doc("Chunk A. Chunk B.", metadata=meta)
        chunks = chunker.chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata.get("kategorie") == "Einkauf"
            assert chunk.metadata.get("wissensart") == "FAQ"

    def test_chunk_strategy_in_metadata(self):
        chunks = _chunks("Text. More.")
        for chunk in chunks:
            assert chunk.metadata.get("chunk_strategy") == "sivas_character"

    def test_chunk_id_is_stable_and_unique(self):
        chunks = _chunks("Sentence one. Sentence two. Sentence three.", max_chars=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique within a document"

    def test_multiple_documents_metadata_not_mixed(self):
        chunker = SivasCharacterChunker()
        doc_a = _doc("Doc A text.", doc_id="A", metadata={"kategorie": "Cat_A"})
        doc_b = _doc("Doc B text.", doc_id="B", metadata={"kategorie": "Cat_B"})
        chunks = chunker.chunk_documents([doc_a, doc_b])
        a_chunks = [c for c in chunks if c.document_id == "A"]
        b_chunks = [c for c in chunks if c.document_id == "B"]
        for chunk in a_chunks:
            assert chunk.metadata.get("kategorie") == "Cat_A"
        for chunk in b_chunks:
            assert chunk.metadata.get("kategorie") == "Cat_B"
