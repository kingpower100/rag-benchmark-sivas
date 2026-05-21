from __future__ import annotations

import json
import os
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
from src.pipeline1.generation.cost_estimator import estimate_cost
from src.pipeline1.generation.factory import build_generator
from src.pipeline1.generation.prompt_builder import PROMPT_TEMPLATE_VERSION, PromptBudget, build_prompt_with_stats, dedupe_prompt_contexts
from src.pipeline1.indexing.factory import build_index
from src.pipeline1.io.jsonl_reader import JsonlReader, list_txt_files
from src.pipeline1.io.manifest_writer import write_manifest
from src.pipeline1.io.result_writer import ResultWriter
from src.pipeline1.metadata import TREASURY_METADATA_SCHEMA_VERSION
from src.pipeline1.preflight import run_preflight_checks
from src.pipeline1.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.pipeline1.retrieval.factory import build_retriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.output_record import OutputRecord
from src.pipeline1.telemetry.logger import build_logger
from src.pipeline1.utils.hashing import file_sha256, stable_hash_dict
from src.pipeline1.utils.seed import set_seed
from tqdm.auto import tqdm


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
    logger = build_logger(run_dir / "logs.txt", cfg.runtime.log_level)

    print("[2/10] Running preflight checks")
    preflight_errors = run_preflight_checks(cfg, project_root)
    if preflight_errors:
        raise RuntimeError("; ".join(preflight_errors))
    docs_path = project_root / cfg.data.documents_path
    questions_path = project_root / cfg.data.questions_path
    print("[3/10] Loading documents")
    docs, document_input_info = _load_documents(cfg, docs_path)
    logger.info("document_input=%s", document_input_info)

    cache_dir = project_root / "data" / "processed"
    chunker_versions = _chunker_versions(cfg)
    documents_fingerprint = _documents_fingerprint(cfg, docs_path)
    chunks_key = stable_hash_dict(
        {
            "documents_fingerprint": documents_fingerprint,
            "documents_source_type": cfg.data.documents_source_type,
            "documents_file_glob": cfg.data.documents_file_glob,
            "documents_recursive": cfg.data.documents_recursive,
            "document_text_field": cfg.data.document_text_field,
            "allow_document_text_fallback": cfg.data.allow_document_text_fallback,
            "metadata_schema_version": TREASURY_METADATA_SCHEMA_VERSION,
            "chunking": cfg.chunking.model_dump(),
            "chunker_versions": chunker_versions,
        }
    )
    chunks_path = cache_dir / "chunks" / f"{chunks_key}.jsonl"
    print("[4/10] Chunking documents")
    chunks = _load_chunks(chunks_path)
    if chunks is None:
        chunks = _build_chunker(cfg).chunk_documents(docs, show_progress=True)
        _save_chunks(chunks_path, chunks)
    else:
        logger.info("Loaded cached chunks: %s", chunks_path)
    chunk_diagnostics = _chunk_diagnostics(chunks, cfg)
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

    embedder = build_embedder(cfg.embedding)
    _print_embedding_runtime_state(cfg, embedder)
    embeddings_key = stable_hash_dict(
        {
            "chunks_key": chunks_key,
            "embedding": cfg.embedding.model_dump(),
        }
    )
    embeddings_path = cache_dir / "embeddings" / f"{embeddings_key}.npy"
    print("[5/10] Generating embeddings")
    cache_validation = {"embeddings": "not_loaded", "index": "not_loaded"}
    embeddings = EmbeddingCache.load(embeddings_path)
    if embeddings is None:
        embeddings = embedder.encode_texts([chunk.text for chunk in chunks], show_progress=True)
        EmbeddingCache.save(embeddings_path, embeddings, {"chunks_key": chunks_key, "embedding": cfg.embedding.model_dump()})
        cache_validation["embeddings"] = "built"
    else:
        try:
            _validate_embedding_cache(embeddings, len(chunks), embeddings_path, chunks_key, cfg.embedding.model_dump())
            cache_validation["embeddings"] = "validated"
        except RuntimeError:
            if cfg.runtime.cache_mismatch_policy != "rebuild":
                raise
            embeddings_path.unlink(missing_ok=True)
            embeddings_path.with_suffix(embeddings_path.suffix + ".meta.json").unlink(missing_ok=True)
            embeddings = embedder.encode_texts([chunk.text for chunk in chunks], show_progress=True)
            EmbeddingCache.save(embeddings_path, embeddings, {"chunks_key": chunks_key, "embedding": cfg.embedding.model_dump()})
            cache_validation["embeddings"] = "rebuilt_after_mismatch"
        logger.info("Loaded cached embeddings: %s", embeddings_path)

    index_key = stable_hash_dict(
        {
            "embeddings_key": embeddings_key,
            "index": cfg.index.model_dump(),
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
    index_path = cache_dir / "indexes" / f"{index_key}.faiss"
    index = build_index(cfg.index)
    print("[6/10] Building/loading FAISS index")
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
        logger.info("Loaded FAISS index: %s", index_path)
    else:
        index.build(embeddings)
        index.save(str(index_path))
        cache_validation["index"] = "built"
        logger.info("Built FAISS index: %s", index_path)

    print("[7/10] Loading questions")
    queries = list(
        JsonlReader.iter_queries(
            str(questions_path),
            cfg.data.question_id_field,
            cfg.data.question_field,
            logger,
            cfg.data.allow_unsafe_query_fields,
        )
    )
    _log_run_info(logger, cfg, docs_count=len(docs), chunk_count=len(chunks), question_count=len(queries), questions_path=questions_path)

    retriever = build_retriever(cfg.retrieval, embedder, index, chunks)
    reranker = (
        CrossEncoderReranker(cfg.reranker.model_name, cfg.reranker.device)
        if cfg.reranker.enabled and cfg.reranker.model_name
        else None
    )
    _print_reranker_runtime_state(cfg, reranker)
    final_top_k = cfg.reranker.final_top_k or cfg.retrieval.top_k
    generator = build_generator(cfg.generation)
    writer = ResultWriter(run_dir, save_csv=cfg.runtime.save_csv, logger=logger)
    existing_ids = writer.load_existing_question_ids() if cfg.runtime.resume else set()
    pending_queries = [query for query in queries if query.question_id not in existing_ids]

    attempted = 0
    written = 0
    retrieval_rows = []
    try:
        print("[8/10] Retrieving contexts")
        for index, query in enumerate(tqdm(pending_queries, desc="Retrieving contexts", unit="question"), start=1):
            attempted += 1
            logger.info(
                "row_start phase=retrieval question_id=%s row=%s/%s",
                query.question_id,
                index,
                len(pending_queries),
            )
            retrieval_start = time.perf_counter()
            raw_retrieved, retrieved, retrieval_warnings, reranker_used = retrieve_top_k_unique_contexts(
                query.question,
                retriever,
                reranker,
                final_top_k,
                cfg.retrieval.fetch_k,
                max_candidates=len(chunks),
            )
            raw_dense_retrieved = _last_candidates(retriever, "last_dense_candidates")
            raw_bm25_retrieved = _last_candidates(retriever, "last_bm25_candidates")
            fused_retrieved = _last_candidates(retriever, "last_fused_candidates")
            retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
            for warning in retrieval_warnings:
                logger.warning("row_retrieval_warning question_id=%s warning=%s", query.question_id, warning)
            logger.info(
                "row_retrieved question_id=%s raw_candidates=%s unique_final_contexts=%s scores=%s retrieval_time_ms=%.2f",
                query.question_id,
                len(raw_retrieved),
                len(retrieved),
                len([item.score for item in retrieved]),
                retrieval_time_ms,
            )
            retrieval_rows.append(
                (
                    query,
                    raw_retrieved,
                    raw_dense_retrieved,
                    raw_bm25_retrieved,
                    fused_retrieved,
                    retrieved,
                    retrieval_time_ms,
                    reranker_used,
                    retrieval_warnings,
                )
            )

        print("[9/10] Generating answers")
        for index, (
            query,
            raw_retrieved,
            raw_dense_retrieved,
            raw_bm25_retrieved,
            fused_retrieved,
            retrieved,
            retrieval_time_ms,
            reranker_used,
            retrieval_warnings,
        ) in enumerate(
            tqdm(retrieval_rows, desc="Generating answers", unit="question"), start=1
        ):
            prompt_contexts = dedupe_prompt_contexts(retrieved)
            logger.info(
                "row_start phase=generation question_id=%s row=%s/%s saved_contexts=%s prompt_contexts=%s",
                query.question_id,
                index,
                len(retrieval_rows),
                len(retrieved),
                len(prompt_contexts),
            )
            prompt, prompt_stats = build_prompt_with_stats(
                cfg.generation.system_prompt,
                query.question,
                prompt_contexts,
                include_metadata_headers=cfg.generation.include_metadata_headers,
                budget=PromptBudget(
                    max_prompt_tokens=cfg.generation.max_prompt_tokens,
                    max_context_tokens=cfg.generation.max_context_tokens,
                    max_chunk_tokens=cfg.generation.max_chunk_tokens,
                    max_context_chars=cfg.generation.max_context_chars,
                    max_chunk_chars=cfg.generation.max_chunk_chars,
                    tokenizer_name=cfg.chunking.tokenizer_name,
                    context_truncation_strategy=cfg.generation.context_truncation_strategy,
                ),
            )
            if cfg.generation.log_prompt_stats:
                logger.info("prompt_stats question_id=%s stats=%s", query.question_id, prompt_stats)
            query_metadata = retriever.extract_query_metadata(query.question) if hasattr(retriever, "extract_query_metadata") else None
            generation_start = time.perf_counter()
            error = None
            try:
                generation = generator.generate(prompt)
                answer = generation.answer
                input_tokens = generation.input_tokens
                output_tokens = generation.output_tokens
            except Exception as ex:
                answer = ""
                input_tokens = 0
                output_tokens = 0
                error = str(ex)
                logger.exception("row_generation_error question_id=%s error=%s", query.question_id, error)
            generation_time_ms = (time.perf_counter() - generation_start) * 1000
            total_tokens = input_tokens + output_tokens
            cost = (
                estimate_cost(
                    input_tokens,
                    output_tokens,
                    cfg.telemetry.pricing.input_per_1k_tokens_usd,
                    cfg.telemetry.pricing.output_per_1k_tokens_usd,
                )
                if cfg.telemetry.estimate_cost
                else 0.0
            )

            record = OutputRecord(
                experiment_id=cfg.experiment.experiment_id,
                question_id=query.question_id,
                uid=query.question_id,
                question=query.question,
                generated_answer=answer,
                retrieved_chunk_ids=[item.chunk_id for item in retrieved],
                retrieved_original_context_ids=[item.original_context_id for item in retrieved],
                raw_retrieved_context_ids=[item.chunk_id for item in raw_retrieved],
                raw_retrieved_original_context_ids=[item.original_context_id for item in raw_retrieved],
                raw_dense_retrieved_context_ids=[item.chunk_id for item in raw_dense_retrieved],
                raw_bm25_retrieved_context_ids=[item.chunk_id for item in raw_bm25_retrieved],
                retrieved_context_ids=[item.chunk_id for item in retrieved],
                retrieved_document_ids=[item.metadata.get("doc_id") or item.metadata.get("document_id") or item.original_context_id for item in retrieved],
                raw_retrieved_document_ids=[
                    item.metadata.get("doc_id") or item.metadata.get("document_id") or item.original_context_id
                    for item in raw_retrieved
                ],
                retrieved_file_names=[item.metadata.get("file_name") or item.metadata.get("source_file") for item in retrieved],
                retrieved_files=[item.metadata.get("source_file") or item.metadata.get("file_name") for item in retrieved],
                raw_retrieved_file_names=[
                    item.metadata.get("file_name") or item.metadata.get("source_file") for item in raw_retrieved
                ],
                citations=_build_citations(retrieved),
                retrieved_chunk_units=[item.chunk_unit for item in retrieved],
                retrieved_chunk_texts=[item.text for item in retrieved],
                retrieved_chunk_metadata=[dict(item.metadata) for item in retrieved],
                retrieved_context_texts=[item.text for item in retrieved],
                retrieval_scores=[item.score for item in retrieved],
                dense_scores=[item.dense_score for item in retrieved],
                bm25_scores=[item.bm25_score for item in retrieved],
                rrf_scores=[item.rrf_score for item in retrieved],
                rerank_scores=[item.rerank_score for item in retrieved],
                ranking_score_type="rerank_score" if reranker_used else (
                    retrieved[0].ranking_score_type if retrieved else cfg.retrieval.retriever_type
                ),
                retrieval_mode=cfg.retrieval.retriever_type,
                retrieved_unique_count=len({item.chunk_id for item in retrieved}),
                raw_retrieved_unique_count=len({item.chunk_id for item in raw_retrieved}),
                raw_duplicate_rate=_duplicate_rate([item.chunk_id for item in raw_retrieved]),
                retrieval_warnings=retrieval_warnings,
                query_metadata={} if query_metadata is None else {
                    "company_names": sorted(query_metadata.company_names),
                    "company_symbols": sorted(query_metadata.company_symbols),
                    "years": sorted(query_metadata.years),
                    "report_periods": sorted(query_metadata.report_periods),
                    "file_names": sorted(query_metadata.file_names),
                    "source_datasets": sorted(query_metadata.source_datasets),
                },
                top_k=final_top_k,
                chunking_strategy=cfg.chunking.strategy,
                chunk_size=cfg.chunking.chunk_size,
                chunk_overlap=cfg.chunking.chunk_overlap,
                embedding_model=cfg.embedding.model_name,
                retriever_type=cfg.retrieval.retriever_type,
                reranker_used=reranker_used,
                llm_model=cfg.generation.model_name,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_latency_ms=retrieval_time_ms + generation_time_ms,
                latency_ms=retrieval_time_ms + generation_time_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                token_usage={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
                estimated_cost=cost,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                prompt_stats=prompt_stats,
                prompt_chars=prompt_stats.get("prompt_chars"),
                prompt_tokens=prompt_stats.get("prompt_tokens"),
                context_chars_before=prompt_stats.get("context_chars_before"),
                context_chars_after=prompt_stats.get("context_chars_after"),
                context_tokens_before=prompt_stats.get("context_tokens_before"),
                context_tokens_after=prompt_stats.get("context_tokens_after"),
                chunks_before=prompt_stats.get("chunks_before"),
                chunks_after=prompt_stats.get("chunks_after"),
                chunks_truncated=prompt_stats.get("chunks_truncated"),
                chunks_dropped=prompt_stats.get("chunks_dropped"),
                error=error,
            )
            writer.write(record)
            written += 1
            logger.info(
                "row_written question_id=%s answer_chars=%s input_tokens=%s output_tokens=%s total_latency_ms=%.2f error=%s",
                query.question_id,
                len(answer),
                input_tokens,
                output_tokens,
                retrieval_time_ms + generation_time_ms,
                error,
            )

        print("[10/10] Writing outputs")
    finally:
        writer.close()

    resolved_config = cfg.model_dump()
    resolved_config["generation"]["base_url"] = os.getenv("OLLAMA_BASE_URL", cfg.generation.base_url)
    end_time = time.time()
    output_counts = _output_row_counts(run_dir)
    write_manifest(
        run_dir,
        {
            "config_path": str(Path(config_path).resolve()),
            "config_hash": file_sha256(config_path),
            "config": cfg.model_dump(),
            "resolved_config": resolved_config,
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
            "chunker_versions": chunker_versions,
            "metadata_schema_version": TREASURY_METADATA_SCHEMA_VERSION,
            "chunk_units": _chunk_unit_counts(chunks),
            "chunk_diagnostics": chunk_diagnostics,
            "output_row_counts": output_counts,
            "run_stats": {"n_documents": len(docs), "n_queries": len(queries), "attempted": attempted, "written": written},
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
            "metadata_schema_version": TREASURY_METADATA_SCHEMA_VERSION,
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
        "metadata_schema_version": TREASURY_METADATA_SCHEMA_VERSION,
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
    candidate_k = max(fetch_k if reranker is not None else top_k, top_k)
    raw_retrieved = []
    retrieved = []
    reranker_used = reranker is not None
    while True:
        raw_retrieved = retriever.retrieve(question, candidate_k)
        ranked = reranker.rerank(question, raw_retrieved, top_k) if reranker is not None else raw_retrieved
        retrieved = dedupe_retrieval_by_chunk_id(ranked, top_k)
        if len(retrieved) >= top_k or candidate_k >= max_candidates:
            break
        candidate_k = min(max_candidates, max(candidate_k + 1, candidate_k * 2))
    warnings = []
    if len(retrieved) < top_k:
        warnings.append(
            f"Only {len(retrieved)} unique chunks were available after deduplication; requested top_k={top_k}."
        )
    return raw_retrieved, retrieved, warnings, reranker_used


def _last_candidates(retriever, attribute: str) -> list:
    value = getattr(retriever, attribute, None)
    return list(value) if isinstance(value, list) else []


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
        current_device = f"cuda:{torch.cuda.current_device()}" if cuda_available and cuda_count > 0 else "cpu"
    except Exception as ex:
        cuda_available = False
        cuda_count = 0
        gpu_name = f"<unavailable: {ex}>"
        current_device = "cpu"

    print(
        "[startup] "
        f"torch_cuda_available={cuda_available} "
        f"cuda_device_count={cuda_count} "
        f"gpu_name={gpu_name} "
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
        return SentenceChunker(cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
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
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
    }


def _prepare_run_dir(run_dir: Path, resume: bool, overwrite: bool) -> None:
    if run_dir.exists() and overwrite:
        for name in ("results.jsonl", "results.csv", "run_manifest.json", "logs.txt", "pipeline1.log"):
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
    for key in ("experiment_id", "config_hash", "documents_fingerprint", "cache_keys", "generation", "prompt_template_version"):
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


def _iso_utc(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
