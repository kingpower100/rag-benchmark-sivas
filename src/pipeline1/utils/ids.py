import hashlib
import json


def make_chunk_id(document_id: str, start: int, end: int, text: str) -> str:
    payload = f"{document_id}:{start}:{end}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_configured_chunk_id(document_id: str, chunk_index: int, text: str, strategy_config: dict) -> str:
    config = json.dumps(strategy_config, sort_keys=True, separators=(",", ":"), default=str)
    payload = f"{document_id}:{chunk_index}:{text}:{config}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_sivas_chunk_id(doc_key: str, chunk_index: int) -> str:
    return f"{doc_key}:chunk:{chunk_index + 1:04d}"


def make_chunk_id_for_document(document_id: str, start: int, end: int, text: str, metadata: dict | None, chunk_index: int) -> str:
    doc_key = (metadata or {}).get("doc_key")
    if doc_key is not None and str(doc_key).strip():
        return make_sivas_chunk_id(str(doc_key).strip(), chunk_index)
    return make_chunk_id(document_id, start, end, text)


def make_configured_chunk_id_for_document(
    document_id: str,
    chunk_index: int,
    text: str,
    strategy_config: dict,
    metadata: dict | None,
) -> str:
    doc_key = (metadata or {}).get("doc_key")
    if doc_key is not None and str(doc_key).strip():
        return make_sivas_chunk_id(str(doc_key).strip(), chunk_index)
    return make_configured_chunk_id(document_id, chunk_index, text, strategy_config)


def stable_retrieved_document_id(metadata: dict | None, original_context_id: str | None) -> str | None:
    metadata = metadata or {}
    return (
        metadata.get("doc_key")
        or metadata.get("document_id")
        or metadata.get("doc_id")
        or original_context_id
    )
