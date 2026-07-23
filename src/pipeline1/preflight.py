import json
import os
import subprocess
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
    orchestration_enabled = _orchestration_enabled(cfg)
    if orchestration_enabled:
        errors.extend(_validate_orchestration_prompt(cfg, base_dir))
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
    if cfg.embedding.require_cuda or str(cfg.embedding.device).startswith("cuda"):
        try:
            import torch

            if cfg.embedding.require_cuda and not str(cfg.embedding.device).startswith("cuda"):
                errors.append("embedding.require_cuda=true requires embedding.device to be set to cuda or cuda:N")
            if not torch.cuda.is_available():
                errors.append("embedding.device is cuda or embedding.require_cuda=true but CUDA is not available to torch")
            elif cfg.embedding.require_cuda and torch.cuda.device_count() == 0:
                errors.append("embedding.require_cuda=true but torch reports zero CUDA devices")
        except Exception as ex:
            errors.append(f"embedding.cuda requirements could not be checked: {ex}")
    reranker_cuda_index = _parse_cuda_device(cfg.reranker.device)
    if cfg.reranker.enabled and reranker_cuda_index is not None:
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append(f"Reranker requested CUDA device {cfg.reranker.device}, but CUDA is unavailable.")
            else:
                device_count = torch.cuda.device_count()
                if reranker_cuda_index >= device_count:
                    errors.append(
                        f"Reranker requested CUDA device {cfg.reranker.device}, but only {device_count} CUDA device(s) are available."
                    )
        except Exception as ex:
            errors.append(f"reranker.device is cuda but torch/CUDA could not be checked: {ex}")
    if questions_path.exists() and questions_path.is_file():
        question_field = "frage" if cfg.data.dataset_schema == "sivas" else cfg.data.question_field
        errors.extend(_validate_questions_file(questions_path, cfg.data.question_id_field, question_field))
        errors.extend(_validate_safe_query_file(questions_path, cfg.data.allow_unsafe_query_fields))
    required_ollama_models = set()
    if cfg.generation.provider == "ollama":
        required_ollama_models.add(cfg.generation.model_name)
    if orchestration_enabled and cfg.orchestration.provider == "ollama":
        required_ollama_models.add(cfg.orchestration.model_name)
    if required_ollama_models and os.getenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "0") != "1":
        default_base_url = (
            cfg.generation.base_url
            if cfg.generation.provider == "ollama"
            else cfg.orchestration.base_url
        )
        base_url = os.getenv("OLLAMA_BASE_URL", default_base_url).rstrip("/")
        cli_models = _ollama_list_models()
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=min(cfg.generation.timeout_s, 10))
            response.raise_for_status()
            available_models = _ollama_model_names(response.json()) | cli_models
            missing_models = sorted(
                model for model in required_ollama_models if not _ollama_model_available(model, available_models)
            )
            if missing_models:
                available = ", ".join(sorted(available_models)) or "<none>"
                errors.append(
                    f"Ollama model(s) not found via `ollama list` or {base_url}/api/tags: {', '.join(missing_models)}. "
                    f"Available: {available}. Install with: "
                    + " ; ".join(f"ollama pull {model}" for model in missing_models)
                )
        except requests.RequestException as ex:
            errors.append(f"Unable to reach Ollama at {base_url}/api/tags: {ex}")
    return errors


def _orchestration_enabled(cfg) -> bool:
    return bool(
        cfg.orchestration.enabled
        and cfg.retrieval.retriever_type in {"category_aware_dense", "adaptive_category_aware_dense"}
    )


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


def _validate_questions_file(path: Path, question_id_field: str, question_field: str) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    loaded = 0
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
            question = row.get(question_field)
            if question_id is None or str(question_id).strip() == "":
                errors.append(f"questions file row {line_number} is missing non-empty field '{question_id_field}'")
                continue
            if question is None or str(question).strip() == "":
                errors.append(f"questions file row {line_number} is missing non-empty field '{question_field}'")
                continue
            question_id = str(question_id)
            if question_id in seen:
                duplicates.add(question_id)
            seen.add(question_id)
            loaded += 1
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        errors.append(f"questions file contains duplicate question IDs in field '{question_id_field}': {sample}")
    if loaded == 0 and not errors:
        errors.append(f"no questions loaded from questions_path: {path}")
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
                    f"Pipeline 1 query file must contain questions only and may not contain "
                    f"answer/gold-bearing fields. Found on line {line_number}: {fields}"
                )
                break
    return errors


def _parse_cuda_device(device: str) -> int | None:
    text = str(device or "").strip().lower()
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _ollama_model_available(required: str, available: set[str]) -> bool:
    if required in available:
        return True
    # "model" matches "model:latest" but "model:tag" only matches "model:tag"
    if ":" not in required and f"{required}:latest" in available:
        return True
    return False


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


def _validate_orchestration_prompt(cfg, base_dir: Path | None) -> list[str]:
    errors: list[str] = []
    prompt_path = getattr(cfg.orchestration, "prompt_path", None)
    if prompt_path is None:
        return errors
    resolved = Path(prompt_path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = (base_dir / resolved).resolve()
    if not resolved.exists() or not resolved.is_file():
        errors.append(f"orchestration.prompt_path is missing or not a file: {resolved}")
    elif resolved.stat().st_size == 0:
        errors.append(f"orchestration.prompt_path is empty: {resolved}")
    return errors


def _ollama_list_models() -> set[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
    except Exception:
        return set()
    names: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        columns = line.split()
        if columns:
            names.add(columns[0])
    return names
