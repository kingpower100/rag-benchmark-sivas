from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.pipeline1.indexing.base import BaseVectorIndex
from src.pipeline1.schemas.chunk import ChunkRecord


class ElasticsearchIndexError(RuntimeError):
    pass


class ElasticsearchIndex(BaseVectorIndex):
    uses_external_storage = True

    def __init__(
        self,
        host: str,
        index_name: str,
        dense_dim: int,
        index_alias: str | None = None,
        index_version: str | None = None,
        vector_field: str = "embedding",
        text_field: str = "text",
        similarity: str = "cosine",
        recreate: bool = False,
        retrieval_mode: str = "script_score",
        num_candidates: int = 100,
        shards: int = 1,
        replicas: int = 0,
        refresh_after_index: bool = True,
        request_timeout: int = 60,
        verify_certs: bool = False,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.host = host
        self.base_index_name = index_name
        self.index_alias = index_alias
        self.index_version = index_version
        self.index_name = self._versioned_index_name(index_name, index_version)
        self.dense_dim = dense_dim
        self.vector_field = vector_field
        self.text_field = text_field
        self.similarity = similarity
        self.recreate = recreate
        self.retrieval_mode = retrieval_mode
        self.num_candidates = num_candidates
        self.shards = shards
        self.replicas = replicas
        self.refresh_after_index = refresh_after_index
        self.request_timeout = request_timeout
        self.verify_certs = verify_certs
        self.username = username
        self.password = password
        self.api_key = api_key
        self.client = client or self._build_client(host)
        self.chunks: list[ChunkRecord] = []
        self.logger = logging.getLogger("pipeline1")
        self.last_health: dict[str, Any] = {}
        self._ensure_available()

    def set_chunks(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks

    def build(self, embeddings: np.ndarray) -> None:
        if len(embeddings.shape) != 2:
            raise ElasticsearchIndexError("Elasticsearch dense index requires a 2D embeddings array.")
        if int(embeddings.shape[1]) != self.dense_dim:
            raise ElasticsearchIndexError(
                f"Embedding dimension mismatch for Elasticsearch index '{self.index_name}': "
                f"config.dense_dim={self.dense_dim} embeddings={int(embeddings.shape[1])}"
            )
        if self.chunks and len(self.chunks) != int(embeddings.shape[0]):
            raise ElasticsearchIndexError(
                f"Chunk/embedding row mismatch for Elasticsearch index '{self.index_name}': "
                f"chunks={len(self.chunks)} embeddings={int(embeddings.shape[0])}"
            )
        self._ensure_index()
        if self.chunks:
            self._bulk_index_chunks(embeddings)
        self._verify_index_exists()
        self.last_health = self.health()
        self.logger.info("Elasticsearch health index=%s health=%s", self.index_name, self.last_health)
        if self.index_alias:
            self._update_alias()

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f"external_backend=elasticsearch\nhost={self.host}\nindex_name={self.index_name}\n",
            encoding="utf-8",
        )

    def load(self, path: str) -> None:
        self._ensure_index()

    def search(self, query_embedding: np.ndarray, top_k: int):
        start = time.perf_counter()
        query_vector = np.asarray(query_embedding, dtype="float32").tolist()
        response = self._execute_search(query_vector, top_k)
        hits = response.get("hits", {}).get("hits", [])
        latency_ms = (time.perf_counter() - start) * 1000
        self.logger.info(
            "Elasticsearch dense query index=%s top_k=%s hits=%s latency_ms=%.2f",
            self.index_name,
            top_k,
            len(hits),
            latency_ms,
        )
        chunk_ids = []
        scores = []
        for hit in hits:
            source = hit.get("_source") or {}
            chunk_ids.append(str(source.get("chunk_id") or hit.get("_id")))
            scores.append(float(hit.get("_score") or 0.0) - 1.0)
        return chunk_ids, scores

    def search_hits(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        start = time.perf_counter()
        query_vector = np.asarray(query_embedding, dtype="float32").tolist()
        response = self._execute_search(query_vector, top_k)
        hits = response.get("hits", {}).get("hits", [])
        latency_ms = (time.perf_counter() - start) * 1000
        self.logger.info(
            "Elasticsearch dense query index=%s top_k=%s hits=%s latency_ms=%.2f",
            self.index_name,
            top_k,
            len(hits),
            latency_ms,
        )
        return hits

    def search_hits_filtered(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        category_field: str,
        category_value: str,
    ) -> list[dict[str, Any]]:
        """Dense search with a term filter on a metadata field (script_score mode only)."""
        start = time.perf_counter()
        query_vector = np.asarray(query_embedding, dtype="float32").tolist()
        response = self._script_score_search_filtered(query_vector, top_k, category_field, category_value)
        hits = response.get("hits", {}).get("hits", [])
        latency_ms = (time.perf_counter() - start) * 1000
        self.logger.info(
            "Elasticsearch dense category-filtered query index=%s field=%s top_k=%s hits=%s latency_ms=%.2f",
            self.index_name,
            category_field,
            top_k,
            len(hits),
            latency_ms,
        )
        return hits

    @property
    def ntotal(self) -> int:
        try:
            response = self.client.count(index=self.index_name)
            return int(response.get("count") or 0)
        except Exception:
            return 0

    @property
    def dim(self) -> int:
        return self.dense_dim

    def _build_client(self, host: str):
        try:
            from elasticsearch import Elasticsearch
        except Exception as ex:
            raise ElasticsearchIndexError(
                "index.type='elasticsearch' requires the 'elasticsearch' package. "
                "Install project requirements before running this config."
            ) from ex
        kwargs: dict[str, Any] = {
            "request_timeout": self.request_timeout,
            "verify_certs": self.verify_certs,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        elif self.username is not None or self.password is not None:
            kwargs["basic_auth"] = (self.username or "", self.password or "")
        return Elasticsearch(host, **kwargs)

    def _ensure_available(self) -> None:
        try:
            if hasattr(self.client, "info"):
                self.client.info()
            elif hasattr(self.client, "ping") and not self.client.ping():
                raise RuntimeError("ping returned false")
        except Exception as ex:
            raise ElasticsearchIndexError(f"Elasticsearch is unavailable at {self.host}: {ex}") from ex
        self.logger.info("Elasticsearch connected host=%s index=%s", self.host, self.index_name)

    def _ensure_index(self) -> None:
        try:
            exists = self.client.indices.exists(index=self.index_name)
            if exists and self.recreate:
                self.client.indices.delete(index=self.index_name)
                exists = False
            if not exists:
                self.client.indices.create(index=self.index_name, body=self._index_body())
                self.logger.info("Elasticsearch index created index=%s", self.index_name)
            else:
                self.logger.info("Elasticsearch index reused index=%s", self.index_name)
        except Exception as ex:
            raise ElasticsearchIndexError(
                f"Failed to prepare Elasticsearch index '{self.index_name}' at {self.host}: {ex}"
            ) from ex

    def _index_body(self) -> dict[str, Any]:
        return {
            "settings": {
                "number_of_shards": self.shards,
                "number_of_replicas": self.replicas,
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "original_context_id": {"type": "keyword"},
                    self.text_field: {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                    self.vector_field: self._vector_field_mapping(),
                }
            }
        }

    def _vector_field_mapping(self) -> dict[str, Any]:
        # script_score is a brute-force exhaustive scan; building an ANN/HNSW index
        # wastes ingestion time and memory without any benefit.
        # index=False disables HNSW while keeping the field queryable via script_score.
        if self.retrieval_mode == "script_score":
            return {"type": "dense_vector", "dims": self.dense_dim, "index": False}
        return {
            "type": "dense_vector",
            "dims": self.dense_dim,
            "index": True,
            "similarity": self.similarity,
        }

    def _bulk_index_chunks(self, embeddings: np.ndarray) -> None:
        if not self.chunks:
            return
        batch_size = 500
        indexed = 0
        for start in range(0, len(self.chunks), batch_size):
            operations = []
            batch = self.chunks[start:start + batch_size]
            batch_embeddings = embeddings[start:start + len(batch)]
            for chunk, embedding in zip(batch, batch_embeddings):
                operations.append({"index": {"_index": self.index_name, "_id": chunk.chunk_id}})
                operations.append(self._chunk_document(chunk, embedding))
            if operations:
                self.client.bulk(operations=operations, refresh=False)
                indexed += len(batch)
        if self.refresh_after_index and hasattr(self.client.indices, "refresh"):
            self.client.indices.refresh(index=self.index_name)
        self.logger.info("Elasticsearch chunks indexed index=%s count=%s", self.index_name, indexed)

    def _execute_search(self, query_vector: list[float], top_k: int) -> dict[str, Any]:
        if self.retrieval_mode == "knn":
            return self._knn_search(query_vector, top_k)
        return self._script_score_search(query_vector, top_k)

    def _script_score_search(self, query_vector: list[float], top_k: int) -> dict[str, Any]:
        return self.client.search(
            index=self.search_index_name,
            size=top_k,
            query={
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{self.vector_field}') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
            source=["chunk_id", "document_id", "original_context_id", self.text_field, "metadata"],
        )

    def _script_score_search_filtered(
        self,
        query_vector: list[float],
        top_k: int,
        category_field: str,
        category_value: str,
    ) -> dict[str, Any]:
        return self.client.search(
            index=self.search_index_name,
            size=top_k,
            query={
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": [{"term": {category_field: category_value}}]
                        }
                    },
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{self.vector_field}') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
            source=["chunk_id", "document_id", "original_context_id", self.text_field, "metadata"],
        )

    def _knn_search(self, query_vector: list[float], top_k: int) -> dict[str, Any]:
        response = self.client.search(
            index=self.search_index_name,
            size=top_k,
            knn={
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(self.num_candidates, top_k),
            },
            source=["chunk_id", "document_id", "original_context_id", self.text_field, "metadata"],
        )
        return self._normalize_knn_response(response)

    @staticmethod
    def _normalize_knn_response(response: dict[str, Any]) -> dict[str, Any]:
        # Downstream dense retriever subtracts 1.0 because script_score adds it for non-negative ES scores.
        # Normalize kNN hits into the same shape while preserving their native score after subtraction.
        hits = response.get("hits", {}).get("hits", [])
        for hit in hits:
            if "_score" in hit and hit["_score"] is not None:
                hit["_score"] = float(hit["_score"]) + 1.0
        return response

    @property
    def search_index_name(self) -> str:
        return self.index_alias or self.index_name

    def health(self) -> dict[str, Any]:
        health: dict[str, Any] = {"index": self.index_name, "alias": self.index_alias}
        try:
            if hasattr(self.client, "cluster") and hasattr(self.client.cluster, "health"):
                health["cluster"] = self.client.cluster.health(index=self.index_name)
        except Exception as ex:
            health["cluster_error"] = str(ex)
        try:
            health["index_exists"] = bool(self.client.indices.exists(index=self.index_name))
        except Exception as ex:
            health["index_exists_error"] = str(ex)
        return health

    def _verify_index_exists(self) -> None:
        try:
            exists = self.client.indices.exists(index=self.index_name)
        except Exception as ex:
            raise ElasticsearchIndexError(
                f"Failed to verify Elasticsearch index '{self.index_name}' at {self.host}: {ex}"
            ) from ex
        if not exists:
            raise ElasticsearchIndexError(f"Elasticsearch index '{self.index_name}' does not exist after build.")

    def _update_alias(self) -> None:
        body = {
            "actions": [
                {"remove": {"index": "*", "alias": self.index_alias, "ignore_unavailable": True}},
                {"add": {"index": self.index_name, "alias": self.index_alias}},
            ]
        }
        try:
            self.client.indices.update_aliases(body=body)
        except TypeError:
            self.client.indices.update_aliases(actions=body["actions"])
        self.logger.info("Elasticsearch alias updated alias=%s index=%s", self.index_alias, self.index_name)

    @staticmethod
    def _versioned_index_name(index_name: str, index_version: str | None) -> str:
        if not index_version:
            return index_name
        return f"{index_name}_{index_version}"

    def _chunk_document(self, chunk: ChunkRecord, embedding: np.ndarray) -> dict[str, Any]:
        metadata = dict(chunk.metadata)
        if "document_id" not in metadata:
            metadata["document_id"] = chunk.document_id
        if chunk.original_context_id and "original_context_id" not in metadata:
            metadata["original_context_id"] = chunk.original_context_id
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "original_context_id": chunk.original_context_id or chunk.document_id,
            self.text_field: chunk.text,
            "metadata": metadata,
            self.vector_field: np.asarray(embedding, dtype="float32").tolist(),
        }
