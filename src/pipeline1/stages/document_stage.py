from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline1.io.jsonl_reader import JsonlReader
from src.pipeline1.metadata import METADATA_SCHEMA_VERSION
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.stages.base import BaseStage, StageInput, StageOutput


@dataclass(frozen=True)
class DocumentStageOutput(StageOutput):
    documents: list[DocumentRecord] = None
    document_input_info: dict = None


class DocumentStage(BaseStage):
    stage_name = "documents"

    def __init__(self, cfg: PipelineConfig, docs_path: Path) -> None:
        self.cfg = cfg
        self.docs_path = docs_path

    def run(self, stage_input: StageInput | None = None) -> DocumentStageOutput:
        docs, document_input_info = self._load_documents()
        return DocumentStageOutput(
            stage_name=self.stage_name,
            artifacts={"documents": docs},
            diagnostics=document_input_info,
            metadata={"documents_path": str(self.docs_path)},
            documents=docs,
            document_input_info=document_input_info,
        )

    def _load_documents(self) -> tuple[list[DocumentRecord], dict]:
        if self.cfg.data.documents_source_type == "txt_folder":
            docs = JsonlReader.read_txt_folder(
                str(self.docs_path),
                self.cfg.data.documents_file_glob,
                self.cfg.data.documents_recursive,
            )
            return docs, {
                "source_type": "txt_folder",
                "folder_path": str(self.docs_path),
                "file_glob": self.cfg.data.documents_file_glob,
                "recursive": self.cfg.data.documents_recursive,
                "txt_files_loaded": len(docs),
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
            }
        docs = JsonlReader.read_documents(
            str(self.docs_path),
            require_context_id=self.cfg.data.dataset_schema != "sivas",
            text_field=self.cfg.data.document_text_field,
            allow_text_fallback=self.cfg.data.allow_document_text_fallback,
            dataset_schema=self.cfg.data.dataset_schema,
        )
        return docs, {
            "source_type": "jsonl",
            "path": str(self.docs_path),
            "file_glob": None,
            "txt_files_loaded": None,
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
        }
