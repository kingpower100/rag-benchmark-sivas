"""
Regression tests for the Unicode-aware BM25 tokenizer.

Before the fix the regex was [a-z0-9]+, which silently dropped German Umlauts
(ae, oe, ue, AE, OE, UE, ss), breaking BM25 matching for German ERP text.
After the fix the regex is Unicode word-chars minus underscore (re.UNICODE)
which preserves complete words.
"""
from __future__ import annotations

import pytest

from src.pipeline1.retrieval.bm25_retriever import BM25Retriever, _tokenize
from src.pipeline1.schemas.chunk import ChunkRecord


# ---------------------------------------------------------------------------
# Tokenizer unit tests
# ---------------------------------------------------------------------------

class TestGermanTokenizer:
    """_tokenize() must return each German word as a single, intact token."""

    def test_buchfuehrungsperiode_is_one_token(self):
        tokens = _tokenize("Buchführungsperiode")
        assert tokens == ["buchführungsperiode"], f"Got {tokens}"

    def test_auftraege_is_one_token(self):
        tokens = _tokenize("Aufträge")
        assert tokens == ["aufträge"], f"Got {tokens}"

    def test_uebernahme_is_one_token(self):
        tokens = _tokenize("Übernahme")
        assert tokens == ["übernahme"], f"Got {tokens}"

    def test_groesse_is_one_token(self):
        tokens = _tokenize("Größe")
        assert tokens == ["größe"], f"Got {tokens}"

    def test_lowercase_umlaut_preserved(self):
        tokens = _tokenize("größenordnung")
        assert tokens == ["größenordnung"], f"Got {tokens}"

    def test_sharp_s_preserved(self):
        # ß must stay in the token; it should not split the word
        tokens = _tokenize("Straße")
        assert tokens == ["straße"], f"Got {tokens}"

    def test_all_uppercase_umlauts_lowercased_into_single_tokens(self):
        tokens = _tokenize("Ä Ö Ü")
        assert tokens == ["ä", "ö", "ü"], f"Got {tokens}"

    def test_mixed_german_and_ascii_sentence(self):
        tokens = _tokenize("Die Übergabe der Aufträge")
        assert tokens == ["die", "übergabe", "der", "aufträge"], f"Got {tokens}"

    def test_german_compound_word_is_single_token(self):
        tokens = _tokenize("Rechnungsführung")
        assert tokens == ["rechnungsführung"]

    # ------------------------------------------------------------------
    # Backward-compatibility: ASCII words and numbers must still work
    # ------------------------------------------------------------------

    def test_plain_ascii_words_unchanged(self):
        tokens = _tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_numbers_are_kept(self):
        tokens = _tokenize("Version 3 patch 42")
        assert tokens == ["version", "3", "patch", "42"]

    def test_alphanumeric_token_kept(self):
        tokens = _tokenize("ERP11 version2")
        assert tokens == ["erp11", "version2"]

    def test_punctuation_splits_tokens(self):
        tokens = _tokenize("Auftrag, Bestellung.")
        assert tokens == ["auftrag", "bestellung"]

    def test_empty_string_returns_empty(self):
        assert _tokenize("") == []

    def test_none_like_empty_string_returns_empty(self):
        # _tokenize handles falsy input via `(text or "")`
        assert _tokenize("") == []

    def test_underscore_is_not_part_of_token(self):
        # Underscores used as separators must split tokens
        tokens = _tokenize("chunk_id_01")
        assert tokens == ["chunk", "id", "01"]


# ---------------------------------------------------------------------------
# BM25Retriever integration tests — German vocabulary matching
# ---------------------------------------------------------------------------

def _chunk(cid: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid,
        document_id=f"doc-{cid}",
        original_context_id=f"doc-{cid}",
        text=text,
        chunk_start=0,
        chunk_end=len(text),
        metadata={"document_id": f"doc-{cid}", "file_name": f"{cid}.txt"},
    )


class TestBM25GermanRetrieval:
    """BM25Retriever must score German-language queries against German documents."""

    def test_retrieves_document_with_umlaut_query_term(self):
        chunks = [
            _chunk("c1", "Die Buchführungsperiode beginnt im Januar."),
            _chunk("c2", "Standard English document about invoices."),
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("Buchführungsperiode", top_k=2)
        top_ids = [r.chunk_id for r in results]
        assert "c1" == top_ids[0], f"Expected c1 first, got {top_ids}"

    def test_retrieves_document_with_auftraege(self):
        chunks = [
            _chunk("c1", "Alle offenen Aufträge werden angezeigt."),
            _chunk("c2", "Das System speichert Rechnungen."),
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("Aufträge anzeigen", top_k=2)
        assert results[0].chunk_id == "c1"

    def test_retrieves_document_with_sharp_s(self):
        # Document uses "Straße" as a standalone token (not a compound word) so
        # that BM25 exact-token matching can find it.
        chunks = [
            _chunk("c1", "Die Straße führt zur Innenstadt."),
            _chunk("c2", "Nothing relevant here."),
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("Straße", top_k=2)
        assert results[0].chunk_id == "c1"

    def test_umlaut_query_does_not_match_unrelated_document(self):
        chunks = [
            _chunk("c1", "Completely unrelated English text about databases."),
            _chunk("c2", "Übernahme und Übergabe von Verantwortlichkeiten."),
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("Übernahme", top_k=2)
        assert results[0].chunk_id == "c2"

    def test_case_insensitive_umlaut_matching(self):
        # Document uses lowercase ü; query uses uppercase Ü — both tokenize to ü
        chunks = [
            _chunk("c1", "Die übergabe der Dokumente erfolgte gestern."),
            _chunk("c2", "Irrelevant chunk about something else entirely."),
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("Übergabe", top_k=2)
        assert results[0].chunk_id == "c1"

    def test_bm25_scores_positive_for_matching_umlaut_chunk(self):
        chunks = [_chunk("c1", "Die Rechnungsführung ist korrekt abgeschlossen.")]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("Rechnungsführung", top_k=1)
        assert len(results) == 1
        assert results[0].score > 0.0

    def test_ascii_retrieval_still_works_after_fix(self):
        chunks = [
            _chunk("c1", "invoice processing complete"),
            _chunk("c2", "unrelated text about something else"),
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("invoice", top_k=2)
        assert results[0].chunk_id == "c1"
