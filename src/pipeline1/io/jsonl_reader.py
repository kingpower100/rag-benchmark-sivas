from __future__ import annotations

import json
import logging
from pathlib import Path

from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.metadata import normalize_metadata


class JsonlReader:
    @staticmethod
    def read_documents(
        path: str,
        require_context_id: bool = False,
        text_field: str = "cleaned_context",
        allow_text_fallback: bool = False,
        dataset_schema: str = "sivas",
    ) -> list[DocumentRecord]:
        docs: list[DocumentRecord] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as ex:
                    raise ValueError(f"Document file has invalid JSON on line {line_number}: {ex}") from ex
                if dataset_schema == "sivas":
                    docs.append(_sivas_document_from_row(row, line_number))
                    continue
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
                metadata = {
                    k: v
                    for k, v in row.items()
                    if k not in {"document_id", "id", "text", "context", text_field, "context_id", "original_context_id"}
                }
                docs.append(DocumentRecord(
                    document_id=str(doc_id),
                    original_context_id=str(context_id) if context_id is not None else None,
                    text=str(text),
                    metadata=normalize_metadata(metadata, str(context_id) if context_id is not None else None),
                ))
        if not docs:
            raise ValueError(f"No documents loaded from {path}.")
        return docs

    @staticmethod
    def read_txt_folder(
        path: str,
        file_glob: str = "*.txt",
        recursive: bool = True,
    ) -> list[DocumentRecord]:
        docs: list[DocumentRecord] = []
        root = Path(path)
        for file_path in list_txt_files(root, file_glob, recursive):
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as ex:
                raise ValueError(f"Unable to read text file as UTF-8: {file_path}") from ex
            if not text.strip():
                continue
            file_name = file_path.name
            relative_path = file_path.relative_to(root).as_posix()
            filename_metadata = {
                "source_file": relative_path,
                "file_name": file_name,
                "source_id": Path(relative_path).with_suffix("").as_posix(),
            }
            metadata = normalize_metadata(
                {
                    **filename_metadata,
                    "source_file": relative_path,
                    "source_path": relative_path,
                    "original_context_id": relative_path,
                },
                relative_path,
            )
            docs.append(
                DocumentRecord(
                    document_id=relative_path,
                    original_context_id=relative_path,
                    text=text,
                    metadata=metadata,
                )
            )
        if not docs:
            raise ValueError(f"No documents loaded from text folder {path}.")
        return docs

    @staticmethod
    def iter_queries(
        path: str,
        question_id_field: str,
        question_field: str,
        logger: logging.Logger | None = None,
        allow_unsafe_fields: bool = False,
        dataset_schema: str = "sivas",
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
        resolved_question_field = "frage" if dataset_schema == "sivas" else question_field
        seen_question_ids: set[str] = set()
        loaded = 0
        with open(path, "r", encoding="utf-8-sig") as f:
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
                            "Use the configured Pipeline 1 questions file without answers/gold contexts, or set "
                            "data.allow_unsafe_query_fields=true only for an explicit unsafe override."
                        )
                    qid = row.get(question_id_field)
                    question = row.get(resolved_question_field)
                    if qid is None or str(qid).strip() == "":
                        raise ValueError(
                            f"Question row {line_number} is missing non-empty question ID field {question_id_field!r}."
                        )
                    if question is None or str(question).strip() == "":
                        raise ValueError(
                            f"Question row {line_number} is missing non-empty question field {resolved_question_field!r}."
                        )
                    qid = str(qid)
                    if qid in seen_question_ids:
                        raise ValueError(f"Question file contains duplicate question_id: {qid}")
                    seen_question_ids.add(qid)
                    loaded += 1
                    yield QueryRecord(question_id=qid, question=str(question))
                except ValueError:
                    raise
                except Exception as ex:
                    if logger:
                        logger.warning("Skipping malformed query row: %s", ex)
        if loaded == 0:
            raise ValueError(f"No questions loaded from {path}.")


def list_txt_files(root: Path, file_glob: str = "*.txt", recursive: bool = True) -> list[Path]:
    if not root.exists():
        return []
    if recursive and "**" not in file_glob and "/" not in file_glob and "\\" not in file_glob:
        iterator = root.rglob(file_glob)
    else:
        iterator = root.glob(file_glob)
    return sorted(path for path in iterator if path.is_file())


def _sivas_document_from_row(row: dict, line_number: int) -> DocumentRecord:
    stable_id = row.get("doc_key")
    if stable_id is None or str(stable_id).strip() == "":
        stable_id = row.get("doc_id")
    if stable_id is None or str(stable_id).strip() == "":
        raise ValueError(f"SIVAS document row {line_number} is missing stable ID doc_key/doc_id.")
    text = row.get("text")
    if text is None or str(text).strip() == "":
        raise ValueError(f"SIVAS document row {line_number} is missing non-empty text field 'text'.")
    metadata = {
        "doc_id": row.get("doc_id"),
        "doc_key": str(stable_id),
        "doc_name": row.get("doc_name"),
        "kategorie": row.get("kategorie"),
        "wissensart": row.get("wissensart"),
        "titel": row.get("titel"),
        "quellpfad": row.get("quellpfad"),
        "sprache": row.get("sprache"),
        "content_hash": row.get("content_hash"),
        "ingestion_version": row.get("ingestion_version"),
        "source_dataset": "sivas",
        "file_name": row.get("doc_name"),
        "source_file": row.get("quellpfad") or row.get("doc_name"),
        "document_id": str(stable_id),
        "original_context_id": str(stable_id),
    }
    return DocumentRecord(
        document_id=str(stable_id),
        original_context_id=str(stable_id),
        text=str(text),
        metadata=metadata,
    )
