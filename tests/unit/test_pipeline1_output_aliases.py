from src.pipeline1.schemas.output_record import OutputRecord


def test_output_record_adds_uid_retrieved_files_citations_latency_and_token_usage():
    record = OutputRecord.model_validate(
        {
            "experiment_id": "exp",
            "question_id": "UID0001",
            "question": "Q?",
            "generated_answer": "2602",
            "retrieved_chunk_ids": ["chunk-1"],
            "retrieved_original_context_ids": ["doc-key-1"],
            "retrieved_context_texts": ["context"],
            "retrieval_scores": [0.75],
            "retrieved_chunk_metadata": [
                {
                    "source_file": "docs/sivas_manual_01.md",
                    "file_name": "sivas_manual_01.md",
                    "source_id": "doc-key-1",
                    "doc_key": "doc-key-1",
                }
            ],
            "top_k": 1,
            "chunking_strategy": "table_aware",
            "chunk_size": 900,
            "chunk_overlap": 150,
            "embedding_model": "embed",
            "retriever_type": "hybrid_rrf",
            "reranker_used": True,
            "llm_model": "llm",
            "retrieval_time_ms": 10.0,
            "generation_time_ms": 20.0,
            "total_latency_ms": 30.0,
            "input_tokens": 11,
            "output_tokens": 3,
            "total_tokens": 14,
        }
    )

    assert record.uid == "UID0001"
    assert record.retrieved_chunks == ["chunk-1"]
    assert record.retrieved_chunks == record.retrieved_chunk_ids
    assert record.retrieved_files == ["sivas_manual_01.md"]
    assert record.latency_ms == 30.0
    assert record.token_usage == {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}
    assert record.citations == [
        {
            "source_file": "docs/sivas_manual_01.md",
            "source_id": "doc-key-1",
            "chunk_id": "chunk-1",
            "rank": 1,
            "score": 0.75,
            "year": None,
            "month": None,
        }
    ]


def test_output_record_accepts_explicit_retrieved_chunks_alias():
    record = OutputRecord.model_validate(
        {
            "experiment_id": "exp",
            "question_id": "Q001",
            "question": "Q?",
            "generated_answer": "A",
            "retrieved_chunks": ["doc-a:chunk:0001"],
            "retrieved_chunk_ids": ["doc-a:chunk:0001"],
            "retrieved_original_context_ids": ["doc-a"],
            "retrieved_context_texts": ["context"],
            "retrieval_scores": [0.9],
            "top_k": 1,
            "chunking_strategy": "fixed_word",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "embedding_model": "embed",
            "retriever_type": "category_aware_dense",
            "reranker_used": False,
            "llm_model": "llm",
            "retrieval_time_ms": 1.0,
            "generation_time_ms": 1.0,
            "total_latency_ms": 2.0,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }
    )

    assert record.retrieved_chunks == ["doc-a:chunk:0001"]
    assert record.retrieved_chunks == record.retrieved_chunk_ids
