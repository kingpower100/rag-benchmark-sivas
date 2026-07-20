"""Parent section store: load, save, fingerprint, and map child chunks to parents."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline1.parent_context.markdown_parser import (
    MAPPING_POLICY_VERSION,
    MARKDOWN_PARENT_PARSER_VERSION,
    PARENT_BOUNDARY_POLICY_VERSION,
    PARENT_METADATA_SCHEMA_VERSION,
    MarkdownSection,
    parse_markdown_sections,
)
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.utils.hashing import stable_hash_dict

PARENT_STORE_FORMAT_VERSION = "2.0"


@dataclass
class ChildMappingEntry:
    chunk_id: str
    parent_id: str | None
    policy: str
    boundary_spanning: bool
    mapping_type: str = "unknown"
    candidate_parent_ids: list = field(default_factory=list)


class ParentStore:
    """Holds parent sections and child->parent mapping for one dataset."""

    def __init__(
        self,
        sections: list[MarkdownSection],
        child_mapping: dict[str, ChildMappingEntry],
    ) -> None:
        self._sections: dict[str, MarkdownSection] = {s.parent_id: s for s in sections}
        self._sections_by_doc: dict[str, list[MarkdownSection]] = {}
        for s in sections:
            self._sections_by_doc.setdefault(s.document_id, []).append(s)
        self._child_mapping = child_mapping

    def get(self, parent_id: str) -> MarkdownSection | None:
        return self._sections.get(parent_id)

    def resolve_parent_id(self, chunk_id: str) -> str | None:
        entry = self._child_mapping.get(chunk_id)
        return entry.parent_id if entry else None

    def get_mapping_entry(self, chunk_id: str) -> ChildMappingEntry | None:
        return self._child_mapping.get(chunk_id)

    def get_sections_for_document(self, doc_id: str) -> list[MarkdownSection]:
        return list(self._sections_by_doc.get(doc_id, []))

    def __contains__(self, parent_id: str) -> bool:
        return parent_id in self._sections

    def all_sections(self) -> list[MarkdownSection]:
        return list(self._sections.values())

    def validate(self) -> list[str]:
        """Return validation error strings (empty = OK)."""
        errors: list[str] = []
        for chunk_id, entry in self._child_mapping.items():
            if entry.parent_id and entry.parent_id not in self._sections:
                errors.append(
                    f"Chunk {chunk_id!r} references missing parent_id {entry.parent_id!r}"
                )
        return errors

    def save(self, store_dir: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)

        with (store_dir / "sections.jsonl").open("w", encoding="utf-8") as f:
            for s in self._sections.values():
                record = {
                    "parent_id": s.parent_id,
                    "document_id": s.document_id,
                    "original_context_id": s.original_context_id,
                    "parent_title": s.parent_title,
                    "heading_level": s.heading_level,
                    "section_index": s.section_index,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                    "parent_text": s.parent_text,
                    "metadata": s.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with (store_dir / "child_mapping.jsonl").open("w", encoding="utf-8") as f:
            for entry in self._child_mapping.values():
                f.write(
                    json.dumps(
                        {
                            "chunk_id": entry.chunk_id,
                            "parent_id": entry.parent_id,
                            "policy": entry.policy,
                            "boundary_spanning": entry.boundary_spanning,
                            "mapping_type": entry.mapping_type,
                            "candidate_parent_ids": entry.candidate_parent_ids,
                        }
                    )
                    + "\n"
                )

    @classmethod
    def load(cls, store_dir: Path) -> "ParentStore":
        sections_path = store_dir / "sections.jsonl"
        mapping_path = store_dir / "child_mapping.jsonl"

        if not sections_path.exists():
            raise FileNotFoundError(f"Parent store sections not found: {sections_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Parent store mapping not found: {mapping_path}")

        sections: list[MarkdownSection] = []
        with sections_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sections.append(
                    MarkdownSection(
                        parent_id=r["parent_id"],
                        document_id=r["document_id"],
                        original_context_id=r.get("original_context_id"),
                        parent_title=r["parent_title"],
                        heading_level=r["heading_level"],
                        section_index=r["section_index"],
                        start_char=r["start_char"],
                        end_char=r["end_char"],
                        parent_text=r["parent_text"],
                        metadata=r.get("metadata", {}),
                    )
                )

        child_mapping: dict[str, ChildMappingEntry] = {}
        with mapping_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                entry = ChildMappingEntry(
                    chunk_id=r["chunk_id"],
                    parent_id=r.get("parent_id"),
                    policy=r.get("policy", "unknown"),
                    boundary_spanning=r.get("boundary_spanning", False),
                    mapping_type=r.get("mapping_type", "unknown"),
                    candidate_parent_ids=r.get("candidate_parent_ids", []),
                )
                child_mapping[entry.chunk_id] = entry

        return cls(sections, child_mapping)

    @classmethod
    def compute_fingerprint(
        cls,
        documents_fingerprint: str,
        chunks_key: str,
        chunking_config: dict,
    ) -> str:
        return stable_hash_dict(
            {
                "format_version": PARENT_STORE_FORMAT_VERSION,
                "parser_version": MARKDOWN_PARENT_PARSER_VERSION,
                "boundary_policy_version": PARENT_BOUNDARY_POLICY_VERSION,
                "mapping_policy_version": MAPPING_POLICY_VERSION,
                "metadata_schema_version": PARENT_METADATA_SCHEMA_VERSION,
                "parent_unit": "markdown_section",
                "documents_fingerprint": documents_fingerprint,
                "chunks_key": chunks_key,
                "chunking": chunking_config,
            }
        )

    @classmethod
    def build(
        cls,
        docs: list[DocumentRecord],
        chunks: list[ChunkRecord],
    ) -> "ParentStore":
        """Parse documents into parent sections and map chunks to parents."""
        sections_by_doc: dict[str, list[MarkdownSection]] = {}
        all_sections: list[MarkdownSection] = []
        for doc in docs:
            doc_sections = parse_markdown_sections(
                document_id=doc.document_id,
                original_context_id=doc.original_context_id,
                text=doc.text,
                inherited_metadata=doc.metadata,
            )
            sections_by_doc[doc.document_id] = doc_sections
            all_sections.extend(doc_sections)

        docs_by_id: dict[str, DocumentRecord] = {d.document_id: d for d in docs}
        child_mapping = _map_chunks_to_parents(chunks, sections_by_doc, docs_by_id)
        return cls(all_sections, child_mapping)


# ---------------------------------------------------------------------------
# Internal child-to-parent mapping: deepest_containing_then_overlap_v2
# ---------------------------------------------------------------------------

def _map_chunks_to_parents(
    chunks: list[ChunkRecord],
    sections_by_doc: dict[str, list[MarkdownSection]],
    docs_by_id: dict[str, DocumentRecord],
) -> dict[str, ChildMappingEntry]:
    result: dict[str, ChildMappingEntry] = {}

    # Group chunks by document
    chunks_by_doc: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    for doc_id, doc_chunks in chunks_by_doc.items():
        sections = sections_by_doc.get(doc_id, [])
        doc = docs_by_id.get(doc_id)

        if not sections or doc is None:
            for chunk in doc_chunks:
                result[chunk.chunk_id] = ChildMappingEntry(
                    chunk_id=chunk.chunk_id,
                    parent_id=None,
                    policy="no_sections",
                    boundary_spanning=False,
                    mapping_type="no_sections",
                    candidate_parent_ids=[],
                )
            continue

        if len(sections) == 1:
            mapping_type = (
                "document_level_fallback" if sections[0].heading_level == 0
                else "sole_section"
            )
            for chunk in doc_chunks:
                result[chunk.chunk_id] = ChildMappingEntry(
                    chunk_id=chunk.chunk_id,
                    parent_id=sections[0].parent_id,
                    policy="sole_section",
                    boundary_spanning=False,
                    mapping_type=mapping_type,
                    candidate_parent_ids=[],
                )
            continue

        # Multiple sections: find chunk and section positions in normalized doc text.
        # Hierarchical sections OVERLAP (H1 contains H2 contains H3). We pick the
        # deepest (most specific) section that fully contains the chunk.
        norm_doc = " ".join(doc.text.split())

        # Compute normalized position ranges for each section in norm_doc.
        # Do NOT advance a search cursor between sections — sections overlap hierarchically.
        section_positions: list[tuple[MarkdownSection, int, int]] = []
        for s in sorted(sections, key=lambda x: x.start_char):
            norm_s = " ".join(s.parent_text.split())
            if not norm_s:
                continue
            pos = norm_doc.find(norm_s)
            if pos >= 0:
                section_positions.append((s, pos, pos + len(norm_s)))

        sorted_chunks = sorted(doc_chunks, key=lambda c: c.chunk_start)
        search_from = 0

        for chunk in sorted_chunks:
            # Find chunk text in normalized doc. Try raw text first (SentenceChunker
            # typically produces single-space-joined text), fall back to normalized.
            pos = norm_doc.find(chunk.text, max(0, search_from))
            if pos < 0:
                pos = norm_doc.find(chunk.text, 0)
            search_text = chunk.text

            if pos < 0:
                norm_chunk = " ".join(chunk.text.split())
                pos = norm_doc.find(norm_chunk, max(0, search_from))
                if pos < 0:
                    pos = norm_doc.find(norm_chunk, 0)
                search_text = norm_chunk

            if pos < 0:
                result[chunk.chunk_id] = ChildMappingEntry(
                    chunk_id=chunk.chunk_id,
                    parent_id=None,
                    policy="text_not_found",
                    boundary_spanning=False,
                    mapping_type="text_not_found",
                    candidate_parent_ids=[],
                )
                continue

            chunk_end = pos + len(search_text)

            # Classify each section relative to this chunk's range.
            fully_containing: list[tuple[MarkdownSection, int, int]] = []
            overlapping: list[tuple[MarkdownSection, int, int, int]] = []

            for section, sec_start, sec_end in section_positions:
                if sec_start <= pos and sec_end >= chunk_end:
                    # Section fully contains the chunk.
                    fully_containing.append((section, sec_start, sec_end))
                else:
                    ov = max(0, min(chunk_end, sec_end) - max(pos, sec_start))
                    if ov > 0:
                        overlapping.append((section, sec_start, sec_end, ov))

            if fully_containing:
                # deepest_containing: highest level number = most specific heading.
                # Tie-break: smallest span (most focused), then earliest section_index.
                fully_containing.sort(
                    key=lambda x: (-x[0].heading_level, x[2] - x[1], x[0].section_index)
                )
                primary = fully_containing[0][0]
                candidates = [fc[0].parent_id for fc in fully_containing[1:]]
                mapping_type = (
                    "document_level_fallback" if primary.heading_level == 0
                    else "fully_contained"
                )
                policy = "deepest_containing"
                boundary_spanning = False

            elif overlapping:
                # No section fully contains the chunk: pick by overlap, then depth, then index.
                overlapping.sort(
                    key=lambda x: (-x[3], -x[0].heading_level, x[0].section_index)
                )
                primary = overlapping[0][0]
                candidates = [
                    o[0].parent_id for o in overlapping[1:]
                    if o[0].parent_id != primary.parent_id
                ]
                mapping_type = "boundary_spanning" if len(overlapping) > 1 else "overlap_based"
                policy = "largest_overlap_deepest"
                boundary_spanning = len(overlapping) > 1

            else:
                # Chunk found in norm_doc but overlaps no known section. Fallback.
                if section_positions:
                    primary = min(section_positions, key=lambda x: x[1])[0]
                    candidates = []
                else:
                    result[chunk.chunk_id] = ChildMappingEntry(
                        chunk_id=chunk.chunk_id,
                        parent_id=None,
                        policy="no_overlap",
                        boundary_spanning=False,
                        mapping_type="no_overlap",
                        candidate_parent_ids=[],
                    )
                    continue
                mapping_type = "no_overlap_fallback"
                policy = "no_overlap_fallback"
                boundary_spanning = False

            result[chunk.chunk_id] = ChildMappingEntry(
                chunk_id=chunk.chunk_id,
                parent_id=primary.parent_id,
                policy=policy,
                boundary_spanning=boundary_spanning,
                mapping_type=mapping_type,
                candidate_parent_ids=candidates,
            )
            search_from = max(0, pos)

    return result
