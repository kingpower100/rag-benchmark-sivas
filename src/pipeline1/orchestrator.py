from __future__ import annotations

import json
import os
import platform
import shutil
import time
import subprocess
from pathlib import Path

from src.pipeline1.chunking.fixed_token_chunker import FIXED_TOKEN_CHUNKER_VERSION, FixedTokenChunker
from src.pipeline1.chunking.fixed_word_chunker import FIXED_WORD_CHUNKER_VERSION, FixedWordChunker
from src.pipeline1.chunking.sentence_chunker import SENTENCE_CHUNKER_VERSION, SENTENCE_SPLITTER_VERSION, SentenceChunker
from src.pipeline1.chunking.table_aware_chunker import TABLE_AWARE_CHUNKER_VERSION, TableAwareChunker
from src.pipeline1.embedding.cache import EmbeddingCache
from src.pipeline1.embedding.factory import build_embedder
from src.pipeline1.generation.factory import build_generator
from src.pipeline1.generation.prompt_builder import PROMPT_TEMPLATE_VERSION
from src.pipeline1.indexing.factory import build_index
from src.pipeline1.io.jsonl_reader import JsonlReader, list_txt_files
from src.pipeline1.io.manifest_writer import write_manifest
from src.pipeline1.io.result_writer import ResultWriter
from src.pipeline1.metadata import METADATA_SCHEMA_VERSION
from src.pipeline1.observability.events import EventType, EventWriter
from src.pipeline1.preflight import run_preflight_checks
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.output_record import OutputRecord
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.chunking_stage import ChunkingStage
from src.pipeline1.stages.document_stage import DocumentStage
from src.pipeline1.stages.embedding_stage import EmbeddingStage
from src.pipeline1.stages.generation_stage import GenerationStage
from src.pipeline1.orchestration.prompt import DEFAULT_ORCHESTRATION_PROMPT_PATH, ORCHESTRATION_PROMPT_VERSION
from src.pipeline1.stages.orchestration_stage import OrchestrationStage
from src.pipeline1.stages.retrieval_stage import (
    RetrievalStage,
    dedupe_retrieval_by_chunk_id,
    json_safe,
    last_candidates,
    retrieval_diagnostics_from,
    retrieve_top_k_unique_contexts,
)
from src.pipeline1.parent_context.parent_store import ParentStore
from src.pipeline1.stages.parent_context_stage import ParentContextStage
from src.pipeline1.stages.run_writer_stage import RunWriterStage
from src.pipeline1.telemetry.logger import build_logger
from src.pipeline1.utils.hashing import file_sha256, stable_hash_dict
from src.pipeline1.utils.seed import set_seed


def run_pipeline(config_path: str) -> Path:
    start_time = time.time()
    print("[1/10] Loading config")
    cfg = PipelineConfig.from_yaml(config_path)
    set_seed(cfg.experiment.random_seed)
    project_root = _project_root()
    run_dir = project_root / cfg.experiment.output_dir / cfg.experiment.experiment_id
    _print_cuda_startup_state(cfg)

    _prepare_run_dir(run_dir, cfg.runtime.resume, cfg.runtime.overwrite)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(run_dir / "logs" / "pipeline1.log", cfg.runtime.log_level, extra_log_paths=[run_dir / "logs.txt"])
    event_writer = EventWriter(run_dir / "events.jsonl", cfg.experiment.experiment_id, run_id=cfg.experiment.experiment_id)
    event_writer.write(
        stage="pipeline",
        event_type=EventType.PIPELINE_START,
        message="Pipeline 1 run started.",
        metrics={"config_path": str(Path(config_path).resolve())},
    )

    print("[2/10] Running preflight checks")
    preflight_errors = run_preflight_checks(cfg, project_root)
    if preflight_errors:
        event_writer.write(
            stage="pipeline",
            event_type=EventType.PIPELINE_ERROR,
            message="Pipeline 1 preflight failed.",
            diagnostics={"errors": preflight_errors},
        )
        event_writer.close()
        raise RuntimeError("; ".join(preflight_errors))
    docs_path = project_root / cfg.data.documents_path
    questions_path = project_root / cfg.data.questions_path
    print("[3/10] Loading documents")
    document_load_start = time.perf_counter()
    event_writer.write(
        stage="documents",
        event_type=EventType.DOCUMENT_LOAD_START,
        message="Document loading started.",
        metrics={"documents_path": str(docs_path), "documents_source_type": cfg.data.documents_source_type},
    )
    document_output = DocumentStage(cfg, docs_path).run()
    docs = document_output.documents
    document_input_info = document_output.document_input_info
    event_writer.write(
        stage="documents",
        event_type=EventType.DOCUMENT_LOAD_END,
        message="Document loading completed.",
        duration_ms=(time.perf_counter() - document_load_start) * 1000,
        metrics={"document_count": len(docs), **document_input_info},
    )
    logger.info("document_input=%s", document_input_info)

    cache_dir = project_root / "data" / "processed"
    print("[4/10] Chunking documents")
    chunking_start = time.perf_counter()
    event_writer.write(
        stage="chunking",
        event_type=EventType.CHUNKING_START,
        message="Chunking started.",
        metrics={"chunking_strategy": cfg.chunking.strategy, "chunk_size": cfg.chunking.chunk_size, "chunk_overlap": cfg.chunking.chunk_overlap},
    )
    chunking_output = ChunkingStage(cfg, project_root, cache_dir, docs_path, logger=logger).run(
        StageInput({"documents": docs})
    )
    chunks = chunking_output.chunks
    chunks_key = chunking_output.chunks_key
    chunks_path = chunking_output.chunks_path
    chunker_versions = chunking_output.chunker_versions
    documents_fingerprint = chunking_output.documents_fingerprint
    chunk_diagnostics = chunking_output.chunk_diagnostics
    chunk_cache_status = chunking_output.cache_status
    event_writer.write(
        stage="chunking",
        event_type=EventType.CHUNKING_END,
        message="Chunking completed.",
        duration_ms=(time.perf_counter() - chunking_start) * 1000,
        metrics={"chunk_count": len(chunks), "cache_status": chunk_cache_status, "chunks_path": str(chunks_path)},
        diagnostics=chunk_diagnostics,
    )
    if chunk_diagnostics["empty_chunks"]:
        raise RuntimeError(f"Chunk validation failed: empty_chunks={chunk_diagnostics['empty_chunks']}")
    if chunk_diagnostics["over_max_chunk_chars"] or chunk_diagnostics["over_max_chunk_tokens"]:
        message = (
            "Chunk validation found oversized chunks: "
            f"over_max_chunk_chars={chunk_diagnostics['over_max_chunk_chars']} "
            f"over_max_chunk_tokens={chunk_diagnostics['over_max_chunk_tokens']}"
        )
        if cfg.chunking.oversized_chunk_policy == "raise":
            raise RuntimeError(message)
        logger.warning(message)

    # Build / load parent store (C03 and other parent-context experiments only).
    parent_store: ParentStore | None = None
    parent_store_key: str | None = None
    parent_store_path: Path | None = None
    parent_store_cache_status: str = "disabled"
    if cfg.parent_context.enabled:
        parent_store_key = ParentStore.compute_fingerprint(
            documents_fingerprint=documents_fingerprint,
            chunks_key=chunks_key,
            chunking_config=cfg.chunking.model_dump(),
        )
        parent_store_path = cache_dir / "parent_stores" / parent_store_key
        if parent_store_path.exists() and (parent_store_path / "sections.jsonl").exists():
            parent_store = ParentStore.load(parent_store_path)
            parent_store_cache_status = "loaded"
            logger.info("Loaded parent store: %s", parent_store_path)
        else:
            parent_store = ParentStore.build(docs, chunks)
            errors = parent_store.validate()
            if errors:
                raise RuntimeError("Parent store validation failed: " + "; ".join(errors))
            parent_store.save(parent_store_path)
            parent_store_cache_status = "built"
            logger.info("Built parent store: %s", parent_store_path)

    print("[5/10] Generating embeddings")
    embedding_start = time.perf_counter()
    event_writer.write(
        stage="embedding",
        event_type=EventType.EMBEDDING_START,
        message="Embedding stage started.",
        metrics={"embedding_model": cfg.embedding.model_name, "chunk_count": len(chunks)},
    )
    cache_validation = {"embeddings": "not_loaded", "index": "not_loaded"}
    embedding_output = EmbeddingStage(cfg, cache_dir, embedder_factory=build_embedder, logger=logger).run(
        StageInput({"chunks": chunks, "chunks_key": chunks_key})
    )
    embedder = embedding_output.embedder
    embeddings = embedding_output.embeddings
    embeddings_key = embedding_output.embeddings_key
    embeddings_path = embedding_output.embeddings_path
    cache_validation["embeddings"] = embedding_output.cache_status
    _validate_configured_dense_dim(cfg, embeddings)
    _print_embedding_runtime_state(cfg, embedder)
    embedding_duration_s = time.perf_counter() - embedding_start
    event_writer.write(
        stage="embedding",
        event_type=EventType.EMBEDDING_END,
        message="Embedding stage completed.",
        duration_ms=embedding_duration_s * 1000,
        metrics={
            "embedding_rows": int(embeddings.shape[0]) if len(embeddings.shape) > 0 else 0,
            "embedding_dim": int(embeddings.shape[1]) if len(embeddings.shape) > 1 else None,
            "cache_status": cache_validation["embeddings"],
        },
    )

    index_config_for_cache = (
        {"type": cfg.index.type, "metric": cfg.index.metric, "index_name": cfg.index.index_name}
        if cfg.index.type == "faiss"
        else cfg.index.model_dump()
    )
    index_key = stable_hash_dict(
        {
            "embeddings_key": embeddings_key,
            "index": index_config_for_cache,
        }
    )
    compatibility_payload = _run_compatibility_payload(
        config_path,
        cfg,
        documents_fingerprint,
        chunks_key,
        embeddings_key,
        index_key,
    )
    if run_dir.exists() and cfg.runtime.resume and not cfg.runtime.overwrite and (run_dir / "results.jsonl").exists():
        _validate_resume_compatible(run_dir, compatibility_payload)
    index = build_index(cfg.index)
    if hasattr(index, "set_chunks"):
        index.set_chunks(chunks)
    if hasattr(index, "set_artifact_identity"):
        index.set_artifact_identity(
            {
                "dataset_fingerprint": documents_fingerprint,
                "source_document_fingerprint": documents_fingerprint,
                "chunk_store_fingerprint": chunks_key,
                "chunking_configuration_fingerprint": stable_hash_dict(cfg.chunking.model_dump()),
                "embedding_model_name": cfg.embedding.model_name,
                "embedding_normalization": cfg.embedding.normalize,
                "framework_config_hash": file_sha256(config_path),
                "framework_code_version": _git_commit(project_root),
            }
        )
    index_suffix = f".{cfg.index.type}" if cfg.index.type != "faiss" else ".faiss"
    index_path = cache_dir / "indexes" / f"{index_key}{index_suffix}"
    print(f"[6/10] Building/loading {cfg.index.type} index")
    index_start = time.perf_counter()
    event_writer.write(
        stage="indexing",
        event_type=EventType.INDEX_BUILD_START,
        message="Index stage started.",
        metrics={"index_type": cfg.index.type, "index_path": str(index_path), "chunk_count": len(chunks)},
    )
    if getattr(index, "uses_external_storage", False):
        index.build(embeddings)
        index.save(str(index_path))
        cache_validation["index"] = "built_or_reused"
        logger.info("Prepared %s index: %s", cfg.index.type, index_path)
    else:
        if index_path.exists():
            index.load(str(index_path))
            try:
                _validate_index_cache(index, len(chunks), embeddings, index_path)
                cache_validation["index"] = "validated"
            except RuntimeError:
                if cfg.runtime.cache_mismatch_policy != "rebuild":
                    raise
                index_path.unlink(missing_ok=True)
                index.build(embeddings)
                index.save(str(index_path))
                cache_validation["index"] = "rebuilt_after_mismatch"
            logger.info("Loaded %s index: %s", cfg.index.type, index_path)
        else:
            index.build(embeddings)
            index.save(str(index_path))
            cache_validation["index"] = "built"
            logger.info("Built %s index: %s", cfg.index.type, index_path)
    event_writer.write(
        stage="indexing",
        event_type=EventType.INDEX_BUILD_END,
        message="Index stage completed.",
        duration_ms=(time.perf_counter() - index_start) * 1000,
        metrics={"index_type": cfg.index.type, "cache_status": cache_validation["index"], "index_path": str(index_path)},
        diagnostics={"health": getattr(index, "last_health", {})},
    )

    print("[7/10] Loading questions")
    queries = list(
        JsonlReader.iter_queries(
            str(questions_path),
            cfg.data.question_id_field,
            cfg.data.question_field,
            logger,
            cfg.data.allow_unsafe_query_fields,
            dataset_schema=cfg.data.dataset_schema,
        )
    )
    _log_run_info(logger, cfg, docs_count=len(docs), chunk_count=len(chunks), question_count=len(queries), questions_path=questions_path)

    run_writer_stage = RunWriterStage(run_dir, save_csv=cfg.runtime.save_csv, logger=logger, resume=cfg.runtime.resume)
    run_writer_output = run_writer_stage.run()
    writer = run_writer_output.writer
    existing_ids = run_writer_output.existing_question_ids
    pending_queries = [query for query in queries if query.question_id not in existing_ids]

    print("[8/11] Processing questions with incremental checkpointing")
    orchestration_stage = OrchestrationStage(
        cfg,
        chunks,
        event_writer=event_writer,
        logger=logger,
        generator_factory=build_generator,
    )

    attempted = 0
    written = 0
    retrieval_total_ms = 0.0
    generation_total_ms = 0.0
    try:
        for row_index, query in enumerate(pending_queries, start=1):
            logger.info(
                "checkpoint_row_start question_id=%s row=%s/%s existing_completed=%s",
                query.question_id,
                row_index,
                len(pending_queries),
                len(existing_ids),
            )
            attempted += 1
            try:
                orchestration_output = orchestration_stage.run(StageInput({"queries": [query]}))
                orchestrated_query = orchestration_output.queries[0]

                print(f"[9/11] Retrieving contexts for {row_index}/{len(pending_queries)}")
                retrieval_output = RetrievalStage(
                    cfg,
                    embedder,
                    index,
                    chunks,
                    event_writer=event_writer,
                    logger=logger,
                    embeddings=embeddings,
                ).run(StageInput({"queries": [orchestrated_query]}))
                retriever = retrieval_output.retriever
                final_top_k = retrieval_output.final_top_k

                parent_context_output = ParentContextStage(
                    cfg,
                    parent_store=parent_store,
                    stage_logger=logger,
                ).run(StageInput({"retrieval_rows": retrieval_output.retrieval_rows}))

                print(f"[10/11] Generating answer for {row_index}/{len(pending_queries)}")
                generation_output = GenerationStage(
                    cfg,
                    retriever,
                    event_writer=event_writer,
                    logger=logger,
                    generator_factory=build_generator,
                ).run(StageInput({"retrieval_rows": parent_context_output.retrieval_rows, "final_top_k": final_top_k}))
                generation_row = generation_output.generation_rows[0]
                record = generation_row.output_record
            except Exception as ex:
                logger.exception("checkpoint_row_failed question_id=%s row=%s/%s", query.question_id, row_index, len(pending_queries))
                event_writer.write(
                    stage="pipeline",
                    event_type=EventType.PIPELINE_ERROR,
                    message="Question failed; writing fallback error row and continuing.",
                    question_id=query.question_id,
                    diagnostics={"error": str(ex), "row": row_index, "total_rows": len(pending_queries)},
                )
                record = _fallback_error_record(cfg, query, str(ex))
            output_write_start = time.perf_counter()
            event_writer.write(
                stage="output",
                event_type=EventType.OUTPUT_WRITE_START,
                message="Output row write started.",
                question_id=record.question_id,
            )
            run_writer_stage.write(record)
            existing_ids.add(record.question_id)
            event_writer.write(
                stage="output",
                event_type=EventType.OUTPUT_WRITE_END,
                message="Output row write completed.",
                question_id=record.question_id,
                duration_ms=(time.perf_counter() - output_write_start) * 1000,
                metrics={"results_jsonl": str(writer.jsonl_path), "save_csv": cfg.runtime.save_csv},
            )
            written += 1
            retrieval_total_ms += record.retrieval_time_ms
            generation_total_ms += record.generation_time_ms
            logger.info(
                "row_written question_id=%s answer_chars=%s input_tokens=%s output_tokens=%s total_latency_ms=%.2f error=%s",
                record.question_id,
                len(record.generated_answer),
                record.input_tokens,
                record.output_tokens,
                record.total_latency_ms,
                record.error,
            )

        print("[11/11] Writing outputs")
    finally:
        run_writer_stage.close()

    resolved_config = cfg.model_dump()
    resolved_config["generation"]["base_url"] = os.getenv("OLLAMA_BASE_URL", cfg.generation.base_url)
    end_time = time.time()
    output_counts = RunWriterStage.output_row_counts(run_dir)
    performance_metrics = _performance_metrics(
        run_dir=run_dir,
        chunk_count=len(chunks),
        embedding_duration_s=max(embedding_duration_s, 1e-9),
        attempted=attempted,
        start_time=start_time,
        retrieval_total_ms=retrieval_total_ms,
        generation_total_ms=generation_total_ms,
    )
    event_writer.write(
        stage="pipeline",
        event_type=EventType.PIPELINE_END,
        message="Pipeline 1 run completed.",
        duration_ms=(end_time - start_time) * 1000,
        metrics={"attempted": attempted, "written": written, **output_counts, **performance_metrics},
    )
    event_writer.close()
    output_artifacts = _pipeline1_output_artifacts(run_dir)
    write_manifest(
        run_dir,
        {
            "run_id": cfg.experiment.experiment_id,
            "config_path": str(Path(config_path).resolve()),
            "config_hash": file_sha256(config_path),
            "config": cfg.model_dump(),
            "resolved_config": resolved_config,
            "machine": _machine_info(cfg),
            "data_hashes": {
                "documents_path": str(docs_path),
                "documents_sha256": documents_fingerprint if cfg.data.documents_source_type == "jsonl" else None,
                "documents_source_type": cfg.data.documents_source_type,
                "documents_file_glob": cfg.data.documents_file_glob,
                "documents_fingerprint": documents_fingerprint,
                "txt_files_loaded": document_input_info["txt_files_loaded"],
                "questions_path": str(questions_path),
                "questions_sha256": file_sha256(questions_path),
            },
            "document_input": document_input_info,
            "cache_keys": {"chunks": chunks_key, "embeddings": embeddings_key, "index": index_key},
            "resume_compatibility": compatibility_payload,
            "cache_validation": cache_validation,
            "cache_artifact_paths": {
                "chunks": str(chunks_path),
                "embeddings": str(embeddings_path),
                "embeddings_meta": str(embeddings_path.with_suffix(embeddings_path.suffix + ".meta.json")),
                "index": str(index_path),
            },
            "parent_context": {
                "enabled": cfg.parent_context.enabled,
                "config": cfg.parent_context.model_dump(),
                "parent_store_key": parent_store_key,
                "parent_store_path": str(parent_store_path) if parent_store_path else None,
                "parent_store_cache_status": parent_store_cache_status,
            },
            "chunker_versions": chunker_versions,
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "chunk_units": _chunk_unit_counts(chunks),
            "chunk_diagnostics": chunk_diagnostics,
            "output_row_counts": output_counts,
            "models": {
                "embedding_model": cfg.embedding.model_name,
                "embedding_provider": cfg.embedding.provider,
                "embedding_device": cfg.embedding.device,
                "retriever_type": cfg.retrieval.retriever_type,
                "index_type": cfg.index.type,
                "reranker_enabled": cfg.reranker.enabled,
                "reranker_model": cfg.reranker.model_name if cfg.reranker.enabled else None,
                "generator_provider": cfg.generation.provider,
                "generator_model": cfg.generation.model_name,
                "orchestration_prompt_path": cfg.orchestration.prompt_path or DEFAULT_ORCHESTRATION_PROMPT_PATH,
                "orchestration_prompt_version": cfg.orchestration.prompt_version or ORCHESTRATION_PROMPT_VERSION,
                "orchestration_prompt_sha256": file_sha256(
                    _resolve_orchestration_prompt_path(cfg.orchestration.prompt_path, project_root)
                ),
            },
            "run_stats": {
                "n_documents": len(docs),
                "n_chunks": len(chunks),
                "n_queries": len(queries),
                "attempted": attempted,
                "written": written,
                "failed_questions": max(0, attempted - written),
                **performance_metrics,
            },
            "artifacts": output_artifacts,
            "pipeline_version": "0.1.0",
            "git_commit": _git_commit(project_root),
            "start_timestamp_utc": _iso_utc(start_time),
            "end_timestamp_utc": _iso_utc(end_time),
        },
    )
    return run_dir


def _load_documents(cfg: PipelineConfig, docs_path: Path) -> tuple[list, dict]:
    if cfg.data.documents_source_type == "txt_folder":
        docs = JsonlReader.read_txt_folder(str(docs_path), cfg.data.documents_file_glob, cfg.data.documents_recursive)
        return docs, {
            "source_type": "txt_folder",
            "folder_path": str(docs_path),
            "file_glob": cfg.data.documents_file_glob,
            "recursive": cfg.data.documents_recursive,
            "txt_files_loaded": len(docs),
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
        }
    docs = JsonlReader.read_documents(
        str(docs_path),
        require_context_id=True,
        text_field=cfg.data.document_text_field,
        allow_text_fallback=cfg.data.allow_document_text_fallback,
    )
    return docs, {
        "source_type": "jsonl",
        "path": str(docs_path),
        "file_glob": None,
        "txt_files_loaded": None,
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
    }


def _build_citations(items: list) -> list[dict]:
    citations = []
    for rank, item in enumerate(items, start=1):
        metadata = item.metadata or {}
        citations.append(
            {
                "source_file": metadata.get("source_file") or metadata.get("file_name"),
                "source_id": metadata.get("source_id"),
                "chunk_id": item.chunk_id,
                "rank": rank,
                "score": item.score,
                "year": metadata.get("year") or metadata.get("report_year"),
                "month": metadata.get("month"),
            }
        )
    return citations


def dedupe_retrieval_by_chunk_id(items: list, top_k: int) -> list:
    seen: set[str] = set()
    unique = []
    for item in items:
        key = str(item.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def dedupe_retrieval_by_original_context_id(items: list, top_k: int) -> list:
    return dedupe_retrieval_by_chunk_id(items, top_k)


def _duplicate_rate(ids: list[str]) -> float:
    if not ids:
        return 0.0
    return (len(ids) - len(set(ids))) / len(ids)


def retrieve_top_k_unique_contexts(
    question: str,
    retriever,
    reranker,
    top_k: int,
    fetch_k: int,
    max_candidates: int,
) -> tuple[list, list, list[str], bool]:
    candidate_k = fetch_k
    reranker_used = reranker is not None
    raw_retrieved = retriever.retrieve(question, candidate_k)
    ranked = reranker.rerank(question, raw_retrieved, top_k) if reranker is not None else raw_retrieved
    retrieved = dedupe_retrieval_by_chunk_id(ranked, top_k)
    warnings = []
    if len(retrieved) < top_k:
        warnings.append(
            f"Only {len(retrieved)} unique chunks were available after deduplication within fetch_k={fetch_k}; requested top_k={top_k}."
        )
    return raw_retrieved, retrieved, warnings, reranker_used


def _last_candidates(retriever, attribute: str) -> list:
    value = getattr(retriever, attribute, None)
    return list(value) if isinstance(value, list) else []


def _retrieval_diagnostics(retriever) -> dict:
    value = getattr(retriever, "last_retrieval_diagnostics", None)
    if not isinstance(value, dict):
        return {}
    return _json_safe(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    return value


def _log_run_info(
    logger,
    cfg: PipelineConfig,
    docs_count: int,
    chunk_count: int | None,
    question_count: int,
    questions_path: Path,
) -> None:
    reranker_state = "enabled" if cfg.reranker.enabled else "disabled"
    logger.info("experiment_id=%s", cfg.experiment.experiment_id)
    logger.info("document_count=%s", docs_count)
    logger.info("chunk_count=%s", chunk_count if chunk_count is not None else "pending")
    logger.info("question_count=%s", question_count)
    logger.info("embedding_model=%s", cfg.embedding.model_name)
    logger.info("embedding_device=%s", cfg.embedding.device)
    logger.info("generator_model=%s", cfg.generation.model_name)
    logger.info("top_k=%s", cfg.retrieval.top_k)
    logger.info("fetch_k=%s", cfg.retrieval.fetch_k)
    logger.info("reranker=%s", reranker_state)
    if cfg.reranker.enabled and cfg.reranker.model_name:
        logger.info("reranker_model=%s", cfg.reranker.model_name)
    logger.info("documents_source_type=%s", cfg.data.documents_source_type)
    logger.info("documents_file_glob=%s", cfg.data.documents_file_glob)
    logger.info("documents_recursive=%s", cfg.data.documents_recursive)
    logger.info("question_input_path=%s", questions_path)


def _print_cuda_startup_state(cfg: PipelineConfig) -> None:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available and cuda_count > 0 else "<none>"
        cuda_version = torch.version.cuda
        pytorch_version = torch.__version__
        vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if cuda_available and cuda_count > 0
            else 0
        )
        current_device = f"cuda:{torch.cuda.current_device()}" if cuda_available and cuda_count > 0 else "cpu"
    except Exception as ex:
        cuda_available = False
        cuda_count = 0
        gpu_name = f"<unavailable: {ex}>"
        cuda_version = None
        pytorch_version = None
        vram_gb = 0
        current_device = "cpu"

    print(
        "[startup] "
        f"torch_cuda_available={cuda_available} "
        f"cuda_device_count={cuda_count} "
        f"gpu_name={gpu_name} "
        f"cuda_version={cuda_version} "
        f"vram_gb={vram_gb} "
        f"pytorch_version={pytorch_version} "
        f"current_torch_device={current_device} "
        f"embedding_requested_device={cfg.embedding.device} "
        f"embedding_require_cuda={cfg.embedding.require_cuda} "
        f"reranker_requested_device={cfg.reranker.device}"
    )


def _print_embedding_runtime_state(cfg: PipelineConfig, embedder) -> None:
    runtime_device = getattr(embedder, "runtime_device", "<unknown>")
    tensor_device = getattr(embedder, "embedding_tensor_device", "<unknown>")
    requested_device = getattr(embedder, "requested_device", cfg.embedding.device)
    print(
        "[startup] "
        f"embedding_device={requested_device} "
        f"embedding_runtime_device={runtime_device} "
        f"embedding_tensor_device={tensor_device}"
    )


def _print_reranker_runtime_state(cfg: PipelineConfig, reranker) -> None:
    if reranker is None:
        print("[startup] reranker=disabled")
        return
    runtime_device = getattr(reranker, "runtime_device", "<unknown>")
    requested_device = getattr(reranker, "requested_device", cfg.reranker.device)
    print(
        "[startup] "
        f"reranker_device={requested_device} "
        f"reranker_runtime_device={runtime_device}"
    )


def _documents_fingerprint(cfg: PipelineConfig, docs_path: Path) -> str:
    if cfg.data.documents_source_type == "jsonl":
        return file_sha256(docs_path)
    files = _txt_folder_files(docs_path, cfg.data.documents_file_glob, cfg.data.documents_recursive)
    return stable_hash_dict(
        {
            "source_type": "txt_folder",
            "folder_path": str(docs_path),
            "file_glob": cfg.data.documents_file_glob,
            "recursive": cfg.data.documents_recursive,
            "files": [
                {
                    "path": path.relative_to(docs_path).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in files
            ],
        }
    )


def _txt_folder_files(docs_path: Path, file_glob: str, recursive: bool = True) -> list[Path]:
    return list_txt_files(docs_path, file_glob, recursive)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_chunker(cfg: PipelineConfig):
    if cfg.chunking.strategy == "fixed_token":
        return FixedTokenChunker(
            cfg.chunking.chunk_size,
            cfg.chunking.chunk_overlap,
            cfg.chunking.tokenizer_name,
            cfg.chunking.allow_word_fallback,
        )
    if cfg.chunking.strategy == "fixed_word":
        return FixedWordChunker(cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    if cfg.chunking.strategy == "sentence":
        print("Using sentence-aware chunking with regex sentence boundaries and full-sentence overlap.")
        return SentenceChunker(
            cfg.chunking.chunk_size,
            cfg.chunking.chunk_overlap,
            cfg.chunking.chunk_size_unit or "words",
            cfg.chunking.chunk_overlap_unit or "sentences",
            cfg.chunking.tokenizer_name,
        )
    print("Using table-aware chunking that keeps markdown tables intact when possible.")
    return TableAwareChunker(
        cfg.chunking.chunk_size,
        cfg.chunking.chunk_overlap,
        cfg.chunking.max_chunk_chars,
        cfg.chunking.max_chunk_tokens,
        cfg.chunking.oversized_chunk_policy,
        cfg.chunking.oversized_chunk_warning,
    )


def _chunker_versions(cfg: PipelineConfig) -> dict[str, str]:
    versions = {"chunker_implementation": ""}
    if cfg.chunking.strategy == "fixed_token":
        versions["chunker_implementation"] = FIXED_TOKEN_CHUNKER_VERSION
    elif cfg.chunking.strategy == "fixed_word":
        versions["chunker_implementation"] = FIXED_WORD_CHUNKER_VERSION
    elif cfg.chunking.strategy == "sentence":
        versions["chunker_implementation"] = SENTENCE_CHUNKER_VERSION
        versions["sentence_splitter"] = SENTENCE_SPLITTER_VERSION
    else:
        versions["chunker_implementation"] = TABLE_AWARE_CHUNKER_VERSION
    return versions


def _chunk_unit_counts(chunks: list[ChunkRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        unit = str(chunk.metadata.get("chunk_unit") or "unknown")
        counts[unit] = counts.get(unit, 0) + 1
    return counts


def _chunk_diagnostics(chunks: list[ChunkRecord], cfg: PipelineConfig) -> dict[str, int]:
    return {
        "total_chunks": len(chunks),
        "empty_chunks": sum(1 for chunk in chunks if not chunk.text.strip()),
        "over_max_chunk_chars": sum(1 for chunk in chunks if len(chunk.text) > cfg.chunking.max_chunk_chars),
        "over_max_chunk_tokens": sum(1 for chunk in chunks if len(chunk.text.split()) > cfg.chunking.max_chunk_tokens),
        "max_chunk_chars_observed": max((len(chunk.text) for chunk in chunks), default=0),
        "max_chunk_tokens_observed": max((len(chunk.text.split()) for chunk in chunks), default=0),
    }


def _resolve_orchestration_prompt_path(prompt_path: str | None, project_root: Path) -> Path:
    if prompt_path is None:
        return (project_root / DEFAULT_ORCHESTRATION_PROMPT_PATH).resolve()
    path = Path(prompt_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def _run_compatibility_payload(
    config_path: str,
    cfg: PipelineConfig,
    documents_fingerprint: str,
    chunks_key: str,
    embeddings_key: str,
    index_key: str,
) -> dict:
    return {
        "experiment_id": cfg.experiment.experiment_id,
        "config_hash": file_sha256(config_path),
        "documents_fingerprint": documents_fingerprint,
        "cache_keys": {"chunks": chunks_key, "embeddings": embeddings_key, "index": index_key},
        "generation": cfg.generation.model_dump(),
        "orchestration": cfg.orchestration.model_dump(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
    }


def _validate_configured_dense_dim(cfg: PipelineConfig, embeddings) -> None:
    observed_dim = int(embeddings.shape[1]) if len(embeddings.shape) > 1 else None
    if observed_dim != cfg.index.dense_dim:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"index.dense_dim={cfg.index.dense_dim} but generated embeddings have dimension={observed_dim}. "
            "Set index.dense_dim to the embedding model output dimension."
        )


def _prepare_run_dir(run_dir: Path, resume: bool, overwrite: bool) -> None:
    if run_dir.exists() and overwrite:
        for name in ("results.jsonl", "results.csv", "run_manifest.json", "logs.txt", "pipeline1.log", "events.jsonl"):
            path = run_dir / name
            if path.exists():
                path.unlink()
        return
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run already exists and resume=false: {run_dir}")


def _validate_resume_compatible(run_dir: Path, current: dict) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Cannot resume existing run without run_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = manifest.get("resume_compatibility") or {
        "experiment_id": manifest.get("resolved_config", {}).get("experiment", {}).get("experiment_id"),
        "config_hash": manifest.get("config_hash"),
        "documents_fingerprint": manifest.get("data_hashes", {}).get("documents_fingerprint"),
        "cache_keys": manifest.get("cache_keys"),
        "generation": manifest.get("resolved_config", {}).get("generation"),
        "prompt_template_version": manifest.get("config", {}).get("prompt_template_version"),
    }
    mismatches = []
    for key in ("experiment_id", "config_hash", "documents_fingerprint", "cache_keys", "generation", "orchestration", "prompt_template_version"):
        if previous.get(key) != current.get(key):
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(
            "Cannot resume incompatible Pipeline 1 run. Mismatched fields: "
            + ", ".join(mismatches)
            + f". Use runtime.overwrite=true or a new experiment_id. run_dir={run_dir}"
        )


def _validate_embedding_cache(embeddings, chunk_count: int, path: Path, chunks_key: str, embedding_config: dict) -> None:
    if len(embeddings) != chunk_count:
        raise RuntimeError(f"Cached embeddings row count mismatch for {path}: embeddings={len(embeddings)} chunks={chunk_count}")
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        raise RuntimeError(f"Cached embeddings metadata missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {"chunks_key": chunks_key, "embedding": embedding_config}
    if meta != expected:
        raise RuntimeError(f"Cached embeddings metadata mismatch for {path}")


def _validate_index_cache(index, chunk_count: int, embeddings, path: Path) -> None:
    if getattr(index, "ntotal", None) != chunk_count:
        raise RuntimeError(f"FAISS index row count mismatch for {path}: index={getattr(index, 'ntotal', None)} chunks={chunk_count}")
    embedding_dim = int(embeddings.shape[1]) if len(embeddings.shape) > 1 else None
    if getattr(index, "dim", None) != embedding_dim:
        raise RuntimeError(f"FAISS index dimension mismatch for {path}: index={getattr(index, 'dim', None)} embeddings={embedding_dim}")


def _load_chunks(path: Path) -> list[ChunkRecord] | None:
    if not path.exists():
        return None
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(ChunkRecord.model_validate_json(line))
    return chunks


def _save_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def _output_row_counts(run_dir: Path) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for name in ("results.jsonl", "results.csv"):
        path = run_dir / name
        if not path.exists():
            counts[name] = None
            continue
        with path.open("r", encoding="utf-8") as f:
            row_count = sum(1 for line in f if line.strip())
        counts[name] = max(0, row_count - 1) if name.endswith(".csv") and row_count else row_count
    return counts


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _machine_info(cfg: PipelineConfig) -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "embedding_requested_device": cfg.embedding.device,
        "reranker_requested_device": cfg.reranker.device,
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        info.update(
            {
                "torch_cuda_available": cuda_available,
                "cuda_device_count": torch.cuda.device_count(),
                "current_torch_device": f"cuda:{torch.cuda.current_device()}" if cuda_available else "cpu",
                "gpu_name": torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() > 0 else None,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "gpu_vram_gb": (
                    round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                    if cuda_available and torch.cuda.device_count() > 0
                    else 0
                ),
                "gpu_peak_vram_gb": (
                    round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)
                    if cuda_available and torch.cuda.device_count() > 0
                    else 0
                ),
            }
        )
    except Exception as ex:
        info.update(
            {
                "torch_cuda_available": False,
                "cuda_device_count": 0,
                "current_torch_device": "unknown",
                "gpu_name": None,
                "torch_error": str(ex),
            }
        )
    return info


def _pipeline1_output_artifacts(run_dir: Path) -> dict[str, dict[str, str | int | None]]:
    artifacts: dict[str, dict[str, str | int | None]] = {}
    for name in ("results.jsonl", "results.csv", "logs.txt", "logs/pipeline1.log", "events.jsonl"):
        path = run_dir / name
        artifacts[name] = _artifact_record(path)
    return artifacts


def _fallback_error_record(cfg: PipelineConfig, query, error: str) -> OutputRecord:
    final_top_k = (
        cfg.reranker.final_top_k
        if cfg.reranker.enabled and cfg.reranker.final_top_k
        else cfg.retrieval.top_k
    )
    return OutputRecord(
        experiment_id=cfg.experiment.experiment_id,
        question_id=query.question_id,
        uid=query.question_id,
        question=query.question,
        cleaned_question=query.cleaned_question,
        detected_category=query.detected_category,
        category_validated=query.category_validated,
        category_validation_reason=query.category_validation_reason,
        orchestration_error=query.orchestration_error,
        generated_answer="",
        retrieved_chunks=[],
        retrieved_chunk_ids=[],
        retrieved_original_context_ids=[],
        retrieved_context_ids=[],
        retrieved_document_ids=[],
        retrieved_documents=[],
        retrieved_categories=[],
        retrieved_files=[],
        retrieved_file_names=[],
        retrieved_chunk_units=[],
        retrieved_chunk_texts=[],
        retrieved_chunk_metadata=[],
        retrieved_context_texts=[],
        retrieval_scores=[],
        dense_scores=[],
        bm25_scores=[],
        rrf_scores=[],
        rerank_scores=[],
        retrieval_warnings=["Question failed before a complete retrieval/generation record could be produced."],
        category_fallback_used=True,
        retrieval_mode="global_fallback",
        retrieval_diagnostics={
            "error": error,
            "detected_category": query.detected_category,
            "category_validated": query.category_validated,
            "category_validation_reason": query.category_validation_reason,
            "retrieval_mode": "global_fallback",
            "category_filter_applied": False,
            "category_fallback_used": True,
            "number_of_category_results": 0,
            "number_of_global_fallback_results": 0,
            "top_k": final_top_k,
            "fetch_k": cfg.retrieval.fetch_k,
        },
        top_k=final_top_k,
        chunking_strategy=cfg.chunking.strategy,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        embedding_model=cfg.embedding.model_name,
        retriever_type=cfg.retrieval.retriever_type,
        reranker_used=False,
        llm_model=cfg.generation.model_name,
        retrieval_time_ms=0.0,
        generation_time_ms=0.0,
        total_latency_ms=0.0,
        latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        estimated_cost=0.0,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        error=error,
    )


def _performance_metrics(
    run_dir: Path,
    chunk_count: int,
    embedding_duration_s: float,
    attempted: int,
    start_time: float,
    retrieval_total_ms: float,
    generation_total_ms: float,
) -> dict[str, float | int | None]:
    elapsed_s = max(time.time() - start_time, 1e-9)
    peak_vram_gb = None
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            peak_vram_gb = round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)
    except Exception:
        peak_vram_gb = None
    return {
        "embedding_throughput_chunks_per_sec": round(chunk_count / embedding_duration_s, 3) if chunk_count else 0.0,
        "questions_per_sec": round(attempted / elapsed_s, 3) if attempted else 0.0,
        "avg_retrieval_ms": round(retrieval_total_ms / attempted, 3) if attempted else None,
        "avg_generation_ms": round(generation_total_ms / attempted, 3) if attempted else None,
        "gpu_peak_vram_gb": peak_vram_gb,
        "resume_completed_rows": RunWriterStage.output_row_counts(run_dir).get("results.jsonl") or 0,
    }


def _artifact_record(path: Path) -> dict[str, str | int | None]:
    if not path.exists():
        return {"path": str(path), "sha256": None, "size_bytes": None}
    return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _iso_utc(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
