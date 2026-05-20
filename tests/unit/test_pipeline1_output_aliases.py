from src.pipeline1.schemas.output_record import OutputRecord


def test_output_record_adds_uid_retrieved_files_citations_latency_and_token_usage():
    record = OutputRecord.model_validate(
        {
            "experiment_id": "exp",
            "question_id": "UID0001",
            "question": "Q?",
            "generated_answer": "2602",
            "retrieved_chunk_ids": ["chunk-1"],
            "retrieved_original_context_ids": ["treasury_bulletin_1941_01.txt"],
            "retrieved_context_texts": ["context"],
            "retrieval_scores": [0.75],
            "retrieved_chunk_metadata": [
                {
                    "source_file": "treasury_bulletin_1941_01.txt",
                    "file_name": "treasury_bulletin_1941_01.txt",
                    "source_id": "treasury_bulletin_1941_01",
                    "year": 1941,
                    "month": "01",
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
    assert record.retrieved_files == ["treasury_bulletin_1941_01.txt"]
    assert record.latency_ms == 30.0
    assert record.token_usage == {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}
    assert record.citations == [
        {
            "source_file": "treasury_bulletin_1941_01.txt",
            "source_id": "treasury_bulletin_1941_01",
            "chunk_id": "chunk-1",
            "rank": 1,
            "score": 0.75,
            "year": 1941,
            "month": "01",
        }
    ]
