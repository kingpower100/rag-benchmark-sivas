import hashlib
import json


def make_chunk_id(document_id: str, start: int, end: int, text: str) -> str:
    payload = f"{document_id}:{start}:{end}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_configured_chunk_id(document_id: str, chunk_index: int, text: str, strategy_config: dict) -> str:
    config = json.dumps(strategy_config, sort_keys=True, separators=(",", ":"), default=str)
    payload = f"{document_id}:{chunk_index}:{text}:{config}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
