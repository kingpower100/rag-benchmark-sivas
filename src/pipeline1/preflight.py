import os
import json
from pathlib import Path

import requests

from src.pipeline1.io.jsonl_reader import list_txt_files


def _resolve_path(base_dir: Path | None, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() or base_dir is None else (base_dir / path).resolve()


def run_preflight_checks(cfg, base_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    docs_path = _resolve_path(base_dir, cfg.data.documents_path)
    questions_path = _resolve_path(base_dir, cfg.data.questions_path)
    errors.extend(_validate_documents_input(cfg, docs_path))
    if not questions_path.exists() or not questions_path.is_file():
        errors.append(f"questions_path is missing or not a file: {questions_path}")
    elif questions_path.stat().st_size == 0:
        errors.append(f"questions_path is empty: {questions_path}")
    final_top_k = cfg.reranker.final_top_k or cfg.retrieval.top_k
    if cfg.reranker.enabled and cfg.retrieval.fetch_k <= final_top_k:
        errors.append(
            f"retrieval.fetch_k ({cfg.retrieval.fetch_k}) must be > final top_k ({final_top_k}) "
            "when reranker.enabled=true"
        )
    elif not cfg.reranker.enabled and cfg.retrieval.fetch_k < cfg.retrieval.top_k:
        errors.append(f"retrieval.fetch_k ({cfg.retrieval.fetch_k}) must be >= retrieval.top_k ({cfg.retrieval.top_k})")
    elif cfg.retrieval.fetch_k == cfg.retrieval.top_k:
        print(
            f"WARNING: retrieval.fetch_k equals retrieval.top_k ({cfg.retrieval.top_k}); no extra candidates are fetched."
        )
    if cfg.chunking.chunk_overlap >= cfg.chunking.chunk_size:
        errors.append(f"chunking.chunk_overlap ({cfg.chunking.chunk_overlap}) must be < chunking.chunk_size ({cfg.chunking.chunk_size})")
    if cfg.index.metric == "cosine" and not cfg.embedding.normalize_embeddings:
        errors.append("embedding.normalize_embeddings must be true when index.metric is cosine")
    if cfg.embedding.require_cuda or cfg.embedding.device == "cuda":
        try:
            import torch

            if cfg.embedding.require_cuda and cfg.embedding.device != "cuda":
                errors.append("embedding.require_cuda=true requires embedding.device to be set to cuda")
            if not torch.cuda.is_available():
                errors.append("embedding.device is cuda or embedding.require_cuda=true but CUDA is not available to torch")
            elif cfg.embedding.require_cuda and torch.cuda.device_count() == 0:
                errors.append("embedding.require_cuda=true but torch reports zero CUDA devices")
        except Exception as ex:
            errors.append(f"embedding.cuda requirements could not be checked: {ex}")
    if cfg.reranker.enabled and cfg.reranker.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append("reranker.device is cuda but CUDA is not available to torch")
        except Exception as ex:
            errors.append(f"reranker.device is cuda but torch/CUDA could not be checked: {ex}")
    if questions_path.exists() and questions_path.is_file():
        errors.extend(_validate_question_ids(questions_path, cfg.data.question_id_field))
        errors.extend(_validate_safe_query_file(questions_path, cfg.data.allow_unsafe_query_fields))
    if os.getenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "0") != "1":
        base_url = os.getenv("OLLAMA_BASE_URL", cfg.generation.base_url).rstrip("/")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=min(cfg.generation.timeout_s, 10))
            response.raise_for_status()
            available_models = _ollama_model_names(response.json())
            if cfg.generation.model_name not in available_models:
                available = ", ".join(sorted(available_models)) or "<none>"
                errors.append(f"Ollama model '{cfg.generation.model_name}' not found at {base_url}/api/tags. Available: {available}")
        except requests.RequestException as ex:
            errors.append(f"Unable to reach Ollama at {base_url}/api/tags: {ex}")
    return errors


def _validate_documents_input(cfg, docs_path: Path) -> list[str]:
    errors: list[str] = []
    source_type = cfg.data.documents_source_type
    if source_type == "jsonl":
        if not docs_path.exists() or not docs_path.is_file():
            errors.append(f"documents_path is missing or not a file: {docs_path}")
        elif docs_path.stat().st_size == 0:
            errors.append(f"documents_path is empty: {docs_path}")
        return errors
    if source_type == "txt_folder":
        if not docs_path.exists() or not docs_path.is_dir():
            errors.append(f"documents_path is missing or not a folder for txt_folder source_type: {docs_path}")
            return errors
        files = list_txt_files(docs_path, cfg.data.documents_file_glob, cfg.data.documents_recursive)
        if not files:
            errors.append(
                f"documents_path has no files matching documents_file_glob={cfg.data.documents_file_glob!r}: {docs_path}"
            )
        return errors
    errors.append(f"Unsupported documents_source_type: {source_type}")
    return errors


def _validate_question_ids(path: Path, question_id_field: str) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as ex:
                errors.append(f"questions file has invalid JSON on line {line_number}: {ex}")
                continue
            question_id = row.get(question_id_field)
            if question_id is None:
                continue
            question_id = str(question_id)
            if question_id in seen:
                duplicates.add(question_id)
            seen.add(question_id)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        errors.append(f"questions file contains duplicate question IDs in field '{question_id_field}': {sample}")
    return errors


def _validate_safe_query_file(path: Path, allow_unsafe_fields: bool) -> list[str]:
    if allow_unsafe_fields:
        return []
    forbidden_fields = {
        "program_answer",
        "original_answer",
        "answer",
        "ground_truth_answer",
        "expected_answer",
        "context_id",
        "gold_context_id",
        "gold_context_ids",
    }
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            unsafe = forbidden_fields & set(row)
            if unsafe:
                fields = ", ".join(sorted(unsafe))
                errors.append(
                    f"Pipeline 1 query file must be questions_only.jsonl-style and may not contain "
                    f"answer/gold-bearing fields. Found on line {line_number}: {fields}"
                )
                break
    return errors


def _ollama_model_names(payload: dict) -> set[str]:
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        for key in ("name", "model"):
            value = model.get(key)
            if value:
                names.add(str(value))
    return names
