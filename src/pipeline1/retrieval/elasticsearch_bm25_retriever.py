from __future__ import annotations

from typing import Any

from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.retrieval import RetrievalItem


class ElasticsearchBM25Error(RuntimeError):
    pass


class ElasticsearchBM25Retriever(BaseRetriever):
    def __init__(
        self,
        chunks: list[ChunkRecord],
        host: str,
        index_name: str,
        k1: float = 1.5,
        b: float = 0.75,
        rebuild_index: bool = False,
        client: Any | None = None,
    ) -> None:
        self.chunks = chunks
        self.host = host
        self.index_name = index_name
        self.k1 = k1
        self.b = b
        self.client = client or self._build_client(host)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.last_bm25_candidates: list[RetrievalItem] = []
        self._ensure_available()
        self._ensure_index(rebuild_index)

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        try:
            response = self.client.search(
                index=self.index_name,
                size=top_k,
                query={"match": {"cleaned_context": {"query": question}}},
            )
        except Exception as ex:
            raise ElasticsearchBM25Error(
                f"Elasticsearch BM25 search failed for index '{self.index_name}' at {self.host}: {ex}"
            ) from ex
        hits = response.get("hits", {}).get("hits", [])
        rows = [self._hit_to_item(hit) for hit in hits]
        self.last_bm25_candidates = rows
        return rows

    def extract_query_metadata(self, question: str):
        from src.pipeline1.retrieval.metadata import extract_query_metadata

        return extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))

    def _build_client(self, host: str):
        try:
            from elasticsearch import Elasticsearch
        except Exception as ex:
            raise ElasticsearchBM25Error(
                "retrieval.bm25.backend='elasticsearch' requires the 'elasticsearch' package. "
                "Install project requirements before running this config."
            ) from ex
        return Elasticsearch(host)

    def _ensure_available(self) -> None:
        try:
            if hasattr(self.client, "info"):
                self.client.info()
            elif hasattr(self.client, "ping") and not self.client.ping():
                raise RuntimeError("ping returned false")
        except Exception as ex:
            raise ElasticsearchBM25Error(f"Elasticsearch is unavailable at {self.host}: {ex}") from ex

    def _ensure_index(self, rebuild_index: bool) -> None:
        try:
            exists = self.client.indices.exists(index=self.index_name)
            if exists and rebuild_index:
                self.client.indices.delete(index=self.index_name)
                exists = False
            if not exists:
                self.client.indices.create(index=self.index_name, body=self._index_body())
                self._bulk_index_chunks()
        except Exception as ex:
            raise ElasticsearchBM25Error(
                f"Failed to prepare Elasticsearch BM25 index '{self.index_name}' at {self.host}: {ex}"
            ) from ex

    def _index_body(self) -> dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "similarity": {
                        "default": {
                            "type": "BM25",
                            "k1": self.k1,
                            "b": self.b,
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "context_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "cleaned_context": {"type": "text"},
                    "file_name": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        }

    def _bulk_index_chunks(self) -> None:
        if not self.chunks:
            return

        batch_size = 1000

        for start in range(0, len(self.chunks), batch_size):
            batch = self.chunks[start:start + batch_size]

            operations = []

            for chunk in batch:
                operations.append({
                    "index": {
                    "_index": self.index_name,
                        "_id": chunk.chunk_id,
                }
                })

                operations.append({
                    "context_id": chunk.metadata.get("context_id") or chunk.metadata.get("original_context_id"),
                    "chunk_id": chunk.chunk_id,
                    "cleaned_context": chunk.text,
                    "file_name": chunk.metadata.get("file_name"),
                    "document_id": getattr(chunk, "document_id", None) or chunk.metadata.get("document_id"),
                    "metadata": chunk.metadata,
                })

            self.client.bulk(operations=operations, refresh=False)

        if hasattr(self.client.indices, "refresh"):
            self.client.indices.refresh(index=self.index_name)

    def _chunk_document(self, chunk: ChunkRecord) -> dict[str, Any]:
        metadata = dict(chunk.metadata)
        return {
            "context_id": chunk.original_context_id or chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "cleaned_context": chunk.text,
            "file_name": metadata.get("file_name") or metadata.get("source_file"),
            "document_id": metadata.get("doc_id") or metadata.get("document_id") or chunk.document_id,
            "metadata": metadata,
        }

    def _hit_to_item(self, hit: dict[str, Any]) -> RetrievalItem:
        source = hit.get("_source") or {}
        chunk_id = str(source.get("chunk_id") or hit.get("_id"))
        chunk = self.chunk_by_id.get(chunk_id)
        metadata = dict(source.get("metadata") or {})
        if source.get("file_name") and "file_name" not in metadata:
            metadata["file_name"] = source["file_name"]
        if source.get("document_id") and "document_id" not in metadata:
            metadata["document_id"] = source["document_id"]
        text = source.get("cleaned_context") or (chunk.text if chunk else "")
        original_context_id = source.get("context_id") or (chunk.original_context_id if chunk else None) or source.get("document_id") or chunk_id
        return RetrievalItem(
            chunk_id=chunk_id,
            original_context_id=str(original_context_id),
            text=str(text),
            score=float(hit.get("_score") or 0.0),
            dense_score=None,
            bm25_score=float(hit.get("_score") or 0.0),
            rrf_score=None,
            rerank_score=None,
            ranking_score_type="bm25_score",
            retrieval_source="elasticsearch_bm25",
            chunk_unit=metadata.get("chunk_unit"),
            metadata=metadata,
        )
