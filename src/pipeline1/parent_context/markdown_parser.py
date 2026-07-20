"""Parse document text into logical Markdown parent sections."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

MARKDOWN_PARENT_PARSER_VERSION = "markdown_heading_v1"
PARENT_BOUNDARY_POLICY_VERSION = "largest_overlap_v1"
PARENT_METADATA_SCHEMA_VERSION = "1.0"
MAPPING_POLICY_VERSION = "deepest_containing_then_overlap_v2"

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+)$", re.MULTILINE)


@dataclass
class MarkdownSection:
    parent_id: str
    document_id: str
    original_context_id: str | None
    parent_title: str
    heading_level: int
    section_index: int
    start_char: int
    end_char: int
    parent_text: str
    metadata: dict = field(default_factory=dict)


def compute_parent_id(document_id: str, heading_level: int, section_index: int, title: str) -> str:
    payload = f"doc={document_id}|level={heading_level}|idx={section_index}|title={title}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_markdown_sections(
    document_id: str,
    original_context_id: str | None,
    text: str,
    inherited_metadata: dict | None = None,
) -> list[MarkdownSection]:
    """Parse document text into parent sections bounded by Markdown headings.

    Boundary policy: a section begins at a heading and ends before the next heading
    of the same or higher level (fewer or equal #). Lower-level sub-headings and
    their content remain inside the parent section.
    """
    if not text:
        return []

    meta = dict(inherited_metadata or {})

    # Find all headings with their positions
    headings: list[tuple[int, int, str]] = []  # (start_pos, level, title)
    try:
        for m in _HEADING_RE.finditer(text):
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((m.start(), level, title))
    except Exception:
        # Malformed input: treat as no headings
        headings = []

    raw_sections: list[tuple[int, int, int, str]] = []  # (start, end, level, title)

    if not headings:
        # No headings: one document-level section
        raw_sections.append((0, len(text), 0, "[document]"))
    else:
        # Content before the first heading becomes an introductory section
        if headings[0][0] > 0:
            intro = text[: headings[0][0]].strip()
            if intro:
                raw_sections.append((0, headings[0][0], 0, "[intro]"))

        for i, (pos, level, title) in enumerate(headings):
            # End is at the next heading of same or higher level
            end = len(text)
            for j in range(i + 1, len(headings)):
                next_pos, next_level, _ = headings[j]
                if next_level <= level:
                    end = next_pos
                    break
            raw_sections.append((pos, end, level, title))

    result: list[MarkdownSection] = []
    section_index = 0
    for start, end, level, title in raw_sections:
        section_text = text[start:end].strip()
        if not section_text:
            continue
        if level > 0:
            # Skip heading-only sections with no body content
            first_nl = section_text.find('\n')
            body = section_text[first_nl + 1:].strip() if first_nl != -1 else ""
            if not body:
                continue
        parent_id = compute_parent_id(document_id, level, section_index, title)
        result.append(
            MarkdownSection(
                parent_id=parent_id,
                document_id=document_id,
                original_context_id=original_context_id,
                parent_title=title,
                heading_level=level,
                section_index=section_index,
                start_char=start,
                end_char=end,
                parent_text=section_text,
                metadata={
                    **meta,
                    "document_id": document_id,
                    "original_context_id": original_context_id,
                },
            )
        )
        section_index += 1

    return result
