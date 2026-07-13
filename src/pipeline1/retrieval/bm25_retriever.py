from __future__ import annotations

import math
import re
import time
from collections import Counter

from src.pipeline1.retrieval.adapters import search_results_to_retrieval_items, strip_adapter_metadata
from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.retrieval.contracts import DedupePolicy, RetrievalTrace, SearchQuery, SearchResult
from src.pipeline1.retrieval.dedupe import dedupe_search_results
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.retrieval import RetrievalItem


# re.UNICODE is the default for str in Python 3, but stated explicitly so the
# intent is clear: German Umlauts (ä ö ü ß Ä Ö Ü) must be part of their token.
# [^\W_]+ matches Unicode word chars minus underscore — letters, digits, accented
# chars — across every script the ERP corpus may contain.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[ChunkRecord], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tokenized = [_tokenize(chunk.text) for chunk in chunks]
        self._doc_lens = [len(tokens) for tokens in self._tokenized]
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        self._term_freqs = [Counter(tokens) for tokens in self._tokenized]
        self._idf = self._build_idf()
        self.last_bm25_candidates: list[RetrievalItem] = []

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        results, trace = self.search(SearchQuery(question_id="", query_text=question, top_k=top_k, fetch_k=top_k))
        ranked = strip_adapter_metadata(search_results_to_retrieval_items(results))
        self.last_bm25_candidates = ranked
        self.last_retrieval_diagnostics = dict(trace.diagnostics)
        return ranked

    def search(self, query: SearchQuery) -> tuple[list[SearchResult], RetrievalTrace]:
        start = time.perf_counter()
        query_terms = _tokenize(query.query_text)
        if not query_terms or not self.chunks:
            trace = RetrievalTrace(
                question_id=query.question_id,
                backend="bm25",
                query_latency_ms=(time.perf_counter() - start) * 1000,
                raw_results_count=0,
                final_results_count=0,
                dedupe_policy=DedupePolicy.NONE.value,
                filters_applied=query.filters,
                diagnostics={"dedupe_policy": DedupePolicy.NONE.value},
            )
            return [], trace
        query_terms = list(dict.fromkeys(query_terms))
        scored = []
        for idx, chunk in enumerate(self.chunks):
            score = self._score(query_terms, idx)
            if score <= 0:
                continue
            scored.append((score, idx, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                original_context_id=chunk.original_context_id or chunk.document_id,
                text=chunk.text,
                score=float(score),
                retrieval_backend="bm25",
                metadata=dict(chunk.metadata),
                diagnostics={"bm25_score": float(score), "ranking_score_type": "bm25_score"},
            )
            for score, _, chunk in scored
        ]
        ranked, dedupe_diagnostics = dedupe_search_results(results, query.top_k, DedupePolicy.NONE)
        ranked = [
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                original_context_id=result.original_context_id,
                text=result.text,
                score=result.score,
                rank=index,
                retrieval_backend=result.retrieval_backend,
                metadata=result.metadata,
                diagnostics=result.diagnostics,
            )
            for index, result in enumerate(ranked, start=1)
        ]
        return ranked, RetrievalTrace(
            question_id=query.question_id,
            backend="bm25",
            query_latency_ms=(time.perf_counter() - start) * 1000,
            raw_results_count=len(results),
            final_results_count=len(ranked),
            dedupe_policy=DedupePolicy.NONE.value,
            filters_applied=query.filters,
            diagnostics=dedupe_diagnostics,
        )

    def extract_query_metadata(self, question: str):
        from src.pipeline1.retrieval.metadata import extract_query_metadata

        return extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))

    def _build_idf(self) -> dict[str, float]:
        n_docs = len(self._tokenized)
        doc_freqs: Counter[str] = Counter()
        for tokens in self._tokenized:
            doc_freqs.update(set(tokens))
        return {
            term: math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
            for term, df in doc_freqs.items()
        }

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        tf = self._term_freqs[doc_index]
        doc_len = self._doc_lens[doc_index]
        if doc_len == 0 or self._avgdl == 0:
            return 0.0
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = self._idf.get(term, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl)
            score += idf * (freq * (self.k1 + 1.0) / denom)
        return score


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())
