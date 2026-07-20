"""ParentContextStage: resolve retrieved child chunks to their Markdown parent sections."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.pipeline1.parent_context.parent_store import ParentStore
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.stages.base import BaseStage, StageInput, StageOutput
from src.pipeline1.stages.retrieval_stage import RetrievalRow

logger = logging.getLogger("pipeline1.parent_context")

# ---------------------------------------------------------------------------
# Token counting — uses tiktoken when available, falls back to char estimate.
# ---------------------------------------------------------------------------
def _load_token_counter(tokenizer_name: str):
    try:
        import tiktoken

        enc = tiktoken.get_encoding(tokenizer_name)
        return lambda text: len(enc.encode(text, disallowed_special=()))
    except Exception:
        return lambda text: max(1, len(text) // 4)


@dataclass
class GenerationContext:
    """Context item passed to prompt building — either a parent section or child fallback."""

    text: str
    parent_id: str | None = None
    trigger_child_id: str | None = None
    trigger_child_rank: int | None = None
    trigger_child_score: float | None = None
    contributing_child_ids: list[str] = field(default_factory=list)
    source_document_id: str | None = None
    parent_title: str | None = None
    parent_context_expanded: bool = False
    oversized_parent: bool = False
    parent_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_child(cls, child, rank: int, expanded: bool = False) -> "GenerationContext":
        return cls(
            text=child.text,
            parent_id=None,
            trigger_child_id=child.chunk_id,
            trigger_child_rank=rank,
            trigger_child_score=child.score,
            contributing_child_ids=[child.chunk_id],
            source_document_id=child.original_context_id,
            parent_title=None,
            parent_context_expanded=expanded,
            metadata=dict(child.metadata or {}),
        )


@dataclass(frozen=True)
class ParentContextStageOutput(StageOutput):
    retrieval_rows: list[RetrievalRow] = field(default_factory=list)
    diagnostics_per_query: list[dict] = field(default_factory=list)


class ParentContextStage(BaseStage):
    stage_name = "parent_context"

    def __init__(
        self,
        cfg: PipelineConfig,
        parent_store: ParentStore | None = None,
        stage_logger=None,
    ) -> None:
        self.cfg = cfg
        self.parent_store = parent_store
        self._logger = stage_logger
        self._count_tokens = _load_token_counter(cfg.chunking.tokenizer_name)

    def run(self, stage_input: StageInput) -> ParentContextStageOutput:
        retrieval_rows: list[RetrievalRow] = list(stage_input.payload["retrieval_rows"])
        pc_cfg = self.cfg.parent_context

        if not pc_cfg.enabled:
            return ParentContextStageOutput(
                stage_name=self.stage_name,
                artifacts={},
                diagnostics={"parent_context_enabled": False},
                metadata={},
                retrieval_rows=retrieval_rows,
            )

        new_rows: list[RetrievalRow] = []
        all_diagnostics: list[dict] = []
        for row in retrieval_rows:
            new_row, diag = self._expand_row(row)
            new_rows.append(new_row)
            all_diagnostics.append(diag)

        return ParentContextStageOutput(
            stage_name=self.stage_name,
            artifacts={},
            diagnostics={"parent_context_enabled": True, "rows_processed": len(new_rows)},
            metadata={"parent_unit": pc_cfg.parent_unit},
            retrieval_rows=new_rows,
            diagnostics_per_query=all_diagnostics,
        )

    def _expand_row(self, row: RetrievalRow) -> tuple[RetrievalRow, dict]:
        pc_cfg = self.cfg.parent_context
        parent_store = self.parent_store
        max_tokens = pc_cfg.max_parent_tokens

        candidates = list(row.raw_retrieved)

        generation_contexts: list[GenerationContext] = []
        seen_parent_ids: set[str] = set()
        duplicate_parent_count = 0
        missing_parent_count = 0
        fallback_to_child_count = 0
        oversized_parent_count = 0
        child_to_parent: dict[str, str | None] = {}

        for rank, child in enumerate(candidates, start=1):
            if len(generation_contexts) >= pc_cfg.unique_parent_top_k:
                break

            parent_id: str | None = None
            if parent_store is not None:
                parent_id = parent_store.resolve_parent_id(child.chunk_id)
            child_to_parent[child.chunk_id] = parent_id

            parent_missing = parent_id is None or (
                parent_store is not None and parent_id not in parent_store
            )

            if parent_missing:
                missing_parent_count += 1
                if pc_cfg.missing_parent_policy == "error":
                    raise RuntimeError(
                        f"No parent section found for chunk {child.chunk_id!r}. "
                        "Set missing_parent_policy: use_child to use the child as fallback."
                    )
                generation_contexts.append(GenerationContext.from_child(child, rank, expanded=False))
                fallback_to_child_count += 1
                continue

            parent = parent_store.get(parent_id)  # type: ignore[union-attr]
            if parent is None:
                missing_parent_count += 1
                if pc_cfg.missing_parent_policy == "error":
                    raise RuntimeError(f"Parent {parent_id!r} not found in store.")
                generation_contexts.append(GenerationContext.from_child(child, rank, expanded=False))
                fallback_to_child_count += 1
                continue

            if pc_cfg.deduplicate and parent_id in seen_parent_ids:
                duplicate_parent_count += 1
                continue

            # Apply oversized-parent policy: prefer a more specific candidate that fits.
            selected_parent = parent
            selected_parent_id = parent_id
            parent_tok = self._count_tokens(parent.parent_text)
            is_oversized = parent_tok > max_tokens

            if is_oversized:
                entry = parent_store.get_mapping_entry(child.chunk_id)  # type: ignore[union-attr]
                for cand_id in (entry.candidate_parent_ids if entry else []):
                    cand = parent_store.get(cand_id)  # type: ignore[union-attr]
                    if cand is None:
                        continue
                    cand_tok = self._count_tokens(cand.parent_text)
                    if cand_tok <= max_tokens:
                        selected_parent = cand
                        selected_parent_id = cand_id
                        parent_tok = cand_tok
                        is_oversized = False
                        break
                if is_oversized:
                    oversized_parent_count += 1
                    logger.warning(
                        "Parent %r for chunk %r has %d tokens (limit %d); "
                        "no smaller candidate found — prompt budget will truncate.",
                        parent_id, child.chunk_id, parent_tok, max_tokens,
                    )

            generation_contexts.append(
                GenerationContext(
                    text=selected_parent.parent_text,
                    parent_id=selected_parent_id,
                    trigger_child_id=child.chunk_id,
                    trigger_child_rank=rank,
                    trigger_child_score=child.score,
                    contributing_child_ids=[child.chunk_id],
                    source_document_id=selected_parent.document_id,
                    parent_title=selected_parent.parent_title,
                    parent_context_expanded=True,
                    oversized_parent=is_oversized,
                    parent_tokens=parent_tok,
                    metadata=dict(selected_parent.metadata),
                )
            )
            seen_parent_ids.add(selected_parent_id)

        diag: dict = {
            "parent_context_enabled": True,
            "retrieved_child_candidate_count": len(candidates),
            "retrieved_child_top_k_count": len(row.retrieved),
            "selected_unique_parent_count": len(generation_contexts),
            "expanded_parent_ids": [
                g.parent_id for g in generation_contexts if g.parent_context_expanded
            ],
            "child_to_parent": dict(child_to_parent),
            "duplicate_parent_count": duplicate_parent_count,
            "missing_parent_count": missing_parent_count,
            "parent_fallback_to_child_count": fallback_to_child_count,
            "oversized_parent_count": oversized_parent_count,
            "parent_tokenizer_name": self.cfg.chunking.tokenizer_name,
            "max_parent_tokens": max_tokens,
            "oversized_parent_policy": "prefer_deeper_section",
            "child_provenance_preserved": True,
        }

        new_row = RetrievalRow(
            query=row.query,
            raw_retrieved=row.raw_retrieved,
            raw_dense_retrieved=row.raw_dense_retrieved,
            raw_bm25_retrieved=row.raw_bm25_retrieved,
            fused_retrieved=row.fused_retrieved,
            retrieved=row.retrieved,
            retrieval_time_ms=row.retrieval_time_ms,
            reranker_used=row.reranker_used,
            retrieval_warnings=row.retrieval_warnings,
            retrieval_diagnostics=row.retrieval_diagnostics,
            generation_contexts=generation_contexts,
            parent_context_diagnostics=diag,
        )
        return new_row, diag
