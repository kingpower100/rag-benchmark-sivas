from __future__ import annotations

import json
import logging

from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.schemas.query import QueryRecord


class JsonlReader:
    @staticmethod
    def read_documents(
        path: str,
        require_context_id: bool = False,
        text_field: str = "cleaned_context",
        allow_text_fallback: bool = False,
    ) -> list[DocumentRecord]:
        docs: list[DocumentRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                context_id = row.get("context_id") or row.get("original_context_id")
                doc_id = row.get("document_id") or row.get("id") or context_id
                if require_context_id and not context_id:
                    raise ValueError(f"Document row {line_number} is missing required context_id/original_context_id.")
                text = row.get(text_field)
                if text is None or str(text).strip() == "":
                    if not allow_text_fallback:
                        raise ValueError(
                            f"Document row {line_number} is missing non-empty configured text field "
                            f"{text_field!r}. Set data.allow_document_text_fallback=true only for unsafe fallback."
                        )
                    text = row.get("context") or row.get("text")
                if text is None or str(text).strip() == "":
                    raise ValueError(f"Document row {line_number} has no usable document text.")
                docs.append(DocumentRecord(
                    document_id=str(doc_id),
                    original_context_id=str(context_id) if context_id is not None else None,
                    text=str(text),
                    metadata={
                        k: v
                        for k, v in row.items()
                        if k not in {"document_id", "id", "text", "context", text_field, "context_id", "original_context_id"}
                    },
                ))
        return docs

    @staticmethod
    def iter_queries(
        path: str,
        question_id_field: str,
        question_field: str,
        logger: logging.Logger | None = None,
        allow_unsafe_fields: bool = False,
    ):
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
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    unsafe = forbidden_fields & set(row)
                    if unsafe and not allow_unsafe_fields:
                        fields = ", ".join(sorted(unsafe))
                        raise ValueError(
                            f"Pipeline 1 query file contains answer/gold-bearing fields on line {line_number}: {fields}. "
                            "Use questions_only.jsonl, or set data.allow_unsafe_query_fields=true only for an explicit unsafe override."
                        )
                    qid = row.get(question_id_field)
                    question = row.get(question_field)
                    if qid is None or question is None:
                        continue
                    yield QueryRecord(question_id=str(qid), question=str(question))
                except ValueError:
                    raise
                except Exception as ex:
                    if logger:
                        logger.warning("Skipping malformed query row: %s", ex)
