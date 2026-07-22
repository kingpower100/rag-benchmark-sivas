from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ChunkGroundTruthError(ValueError):
    """Raised when chunk-level ground truth cannot be used safely."""


@dataclass(frozen=True)
class ChunkGroundTruth:
    by_question: dict[str, set[str]]
    path: Path
    chunk_config_ids: set[str]
    package_metadata: dict[str, Any]

    @property
    def question_count(self) -> int:
        return len(self.by_question)

    @property
    def gold_chunk_count(self) -> int:
        return sum(len(chunks) for chunks in self.by_question.values())

    @property
    def unique_gold_chunk_count(self) -> int:
        return len({chunk for chunks in self.by_question.values() for chunk in chunks})


class ChunkGroundTruthLoader:
    """Load gold chunk annotations into question_id -> relevant chunk IDs."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> ChunkGroundTruth:
        if not self.path.exists():
            raise FileNotFoundError(f"Chunk-level ground-truth file does not exist: {self.path}")
        if not self.path.is_file():
            raise FileNotFoundError(f"Chunk-level ground-truth path is not a file: {self.path}")

        by_question: dict[str, set[str]] = {}
        chunk_config_ids: set[str] = set()
        records = 0
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    records += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ChunkGroundTruthError(
                            f"Malformed chunk-level JSONL at {self.path}:{line_number}: {exc.msg}"
                        ) from exc
                    raw_qid = record.get("question_id")
                    if not isinstance(raw_qid, str):
                        raise ChunkGroundTruthError(
                            f"Chunk-level ground truth row {line_number} is missing required field 'question_id' "
                            "as a string."
                        )
                    qid = raw_qid.strip()
                    if not qid:
                        raise ChunkGroundTruthError(
                            f"Chunk-level ground truth row {line_number} is missing required field 'question_id'."
                        )
                    raw_chunks = record.get("gold_relevant_chunk_ids")
                    if not isinstance(raw_chunks, list):
                        raise ChunkGroundTruthError(
                            f"Chunk-level ground truth row {line_number} for question_id={qid!r} "
                            "must contain list field 'gold_relevant_chunk_ids'."
                        )
                    invalid_chunks = [item for item in raw_chunks if not isinstance(item, str)]
                    if invalid_chunks:
                        raise ChunkGroundTruthError(
                            f"Chunk-level ground truth row {line_number} for question_id={qid!r} "
                            "contains non-string gold chunk identifiers."
                        )
                    chunks = {item.strip() for item in raw_chunks if item.strip()}
                    if not chunks:
                        raise ChunkGroundTruthError(
                            f"Chunk-level ground truth row {line_number} for question_id={qid!r} "
                            "contains no non-empty gold chunk IDs."
                        )
                    by_question.setdefault(qid, set()).update(chunks)
                    chunk_config_id = str(record.get("chunk_config_id") or "").strip()
                    if chunk_config_id:
                        chunk_config_ids.add(chunk_config_id)
        except OSError as exc:
            raise ChunkGroundTruthError(f"Chunk-level ground-truth file is not readable: {self.path}") from exc

        if records == 0:
            raise ChunkGroundTruthError(f"Chunk-level ground-truth file is empty: {self.path}")
        if not by_question or not any(by_question.values()):
            raise ChunkGroundTruthError(
                f"Chunk-level evaluation is enabled but no gold chunks could be loaded from {self.path}."
            )

        return ChunkGroundTruth(
            by_question=by_question,
            path=self.path,
            chunk_config_ids=chunk_config_ids,
            package_metadata=self._load_package_metadata(),
        )

    def _load_package_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for filename in (
            "integration_package.json",
            "final_annotation_manifest.json",
            "final_annotation_validation.json",
        ):
            path = self.path.parent / filename
            if not path.exists():
                continue
            try:
                metadata[filename] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ChunkGroundTruthError(f"Could not read chunk annotation metadata file {path}: {exc}") from exc
        for path in self.path.parent.glob("chunk_mapping_summary_*.json"):
            try:
                metadata[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ChunkGroundTruthError(f"Could not read chunk annotation metadata file {path}: {exc}") from exc
        return metadata


def compute_chunk_retrieval_metrics_for_ks(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: set[str],
    ks: list[int],
) -> dict[str, float]:
    ranked = [str(item).strip() for item in retrieved_chunk_ids if str(item or "").strip()]
    gold = {str(item).strip() for item in gold_chunk_ids if str(item or "").strip()}
    output: dict[str, float] = {}
    for k in ks:
        at_k = ranked[:k]
        relevant_seen = set(at_k) & gold
        hit = 1.0 if relevant_seen else 0.0
        recall = len(relevant_seen) / len(gold) if gold else 0.0
        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(at_k, start=1):
            if chunk_id in gold:
                reciprocal_rank = 1.0 / rank
                break
        output[f"chunk_hit_at_{k}"] = hit
        output[f"chunk_recall_at_{k}"] = recall
        output[f"chunk_mrr_at_{k}"] = reciprocal_rank
        output[f"chunk_ndcg_at_{k}"] = _chunk_ndcg_at_k(at_k, gold, k)
    return output


def _chunk_ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked[:k], start=1):
        gain = 0.0
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            if chunk_id in gold:
                gain = 1.0
        dcg += gain / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
