import csv
import json
import os
from pathlib import Path
from typing import Any

from src.pipeline1.schemas.output_record import OutputRecord


class ResultWriter:
    def __init__(self, run_dir: Path, save_csv: bool = True, logger=None) -> None:
        self.run_dir = run_dir
        self.save_csv = save_csv
        self.logger = logger
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "results.jsonl"
        self.csv_path = self.run_dir / "results.csv"
        self._csv_file = None
        self._csv_writer = None

    def load_existing_question_ids(self) -> set[str]:
        """Return IDs of valid rows that should not be regenerated on resume."""
        ids: set[str] = set()
        for row in self._load_jsonl_rows():
            if _is_valid_result_row(row):
                ids.add(str(row.get("question_id")))
        return ids

    def write(self, record) -> None:
        validated = OutputRecord.model_validate(record)
        export_row = validated.to_export_record()
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(export_row, ensure_ascii=False) + "\n")
        if self.save_csv:
            flat_row = validated.model_dump()
            if self._csv_writer is None:
                exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
                self._csv_file = self.csv_path.open("a", encoding="utf-8", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(flat_row.keys()))
                if not exists:
                    self._csv_writer.writeheader()
            self._csv_writer.writerow(flat_row)
            self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()

    def reconcile_outputs(self) -> list[dict[str, Any]]:
        """Rewrite official outputs from canonical valid rows.

        Canonical rows are keyed by question_id. Invalid or malformed rows are
        excluded, the newest valid row for each question_id wins, and final rows
        are sorted deterministically by question_id before both artifacts are
        atomically rewritten from the same row collection.
        """
        by_id: dict[str, dict[str, Any]] = {}
        for row in self._load_jsonl_rows():
            if not _is_valid_result_row(row):
                continue
            qid = str(row.get("question_id") or "")
            by_id[qid] = row

        sorted_rows = [by_id[qid] for qid in sorted(by_id.keys(), key=_qid_sort_key)]
        _atomic_write_jsonl(self.jsonl_path, sorted_rows)
        if self.save_csv or self.csv_path.exists():
            _atomic_write_csv(self.csv_path, sorted_rows)
        return sorted_rows

    def compact_deduplicate(self) -> int:
        """Backward-compatible wrapper for existing compaction callers."""
        if not self.jsonl_path.exists():
            return 0
        return len(self.reconcile_outputs())

    def _load_jsonl_rows(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows


def _is_valid_result_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if not str(row.get("question_id") or "").strip():
        return False
    if row.get("error") is not None:
        return False
    answer = row.get("answer") if row.get("answer") is not None else row.get("generated_answer")
    return bool(str(answer or "").strip())


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = _csv_fieldnames(rows)
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for preferred in ("question_id", "question", "answer", "generated_answer", "error"):
        if any(preferred in row for row in rows):
            fieldnames.append(preferred)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _qid_sort_key(qid: str) -> tuple:
    """Sort Q001..Q096 numerically; everything else lexicographically after."""
    if qid and qid[0].upper() == "Q" and qid[1:].isdigit():
        return (0, int(qid[1:]))
    return (1, qid)
