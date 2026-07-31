"""Tests for ResultWriter.compact_deduplicate() — resume correctness.

Covers:
- Valid row is preserved unchanged.
- Empty-answer row is replaced by regenerated valid row.
- Error row is replaced by regenerated valid row.
- Final results.jsonl has exactly one row per question_id.
- Final results.csv has exactly one row per question_id.
- No duplicates after resume (4 invalid rows repaired → exactly 96 unique rows).
- Exactly 96 rows after repairing Q022, Q023, Q057, Q084.
- Ordering is Q001 through Q096.
- Idempotent: running compact twice returns the same row count.
- Atomic write: temp file is used so an interrupted rewrite cannot destroy the original.
- Missing rows are not invented (compact only deduplicates).
- Empty file is handled gracefully.
- No CSV file present is handled gracefully.
- Row ordering in CSV matches JSONL after compaction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import unittest.mock as mock
from pathlib import Path

import pytest

from src.pipeline1.io.result_writer import ResultWriter, _qid_sort_key
from src.pipeline1.orchestrator import _reconciled_result_stats


# ── helpers ───────────────────────────────────────────────────────────────────

def _valid_row(question_id: str, answer: str = "Valid answer") -> dict:
    return {"question_id": question_id, "answer": answer, "error": None}


def _invalid_row(question_id: str, answer: str = "", error: str | None = None) -> dict:
    return {"question_id": question_id, "answer": answer, "error": error}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── sort-key unit tests ────────────────────────────────────────────────────────

def test_qid_sort_key_numeric():
    assert _qid_sort_key("Q001") < _qid_sort_key("Q010")
    assert _qid_sort_key("Q010") < _qid_sort_key("Q096")


def test_qid_sort_key_non_q_ids_after_q_ids():
    assert _qid_sort_key("Q096") < _qid_sort_key("other")
    assert _qid_sort_key("Q001") < _qid_sort_key("abc")


def test_qid_sort_key_q_prefix_case_insensitive():
    assert _qid_sort_key("q001") == _qid_sort_key("Q001")


# ── compact_deduplicate: basic preservation ───────────────────────────────────

def test_compact_preserves_valid_rows(tmp_path):
    """A file with only valid, unique rows is unchanged in content."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "results.jsonl", [
        _valid_row("Q001"),
        _valid_row("Q002"),
    ])
    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    rows = _read_jsonl(run_dir / "results.jsonl")
    assert count == 2
    assert [r["question_id"] for r in rows] == ["Q001", "Q002"]
    assert rows[0]["answer"] == "Valid answer"


# ── empty-answer row replaced ──────────────────────────────────────────────────

def test_compact_replaces_empty_answer_row(tmp_path):
    """Original empty-answer row is superseded by the regenerated valid row."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "results.jsonl", [
        _valid_row("Q001"),
        _invalid_row("Q022", answer=""),           # original G03 failure pattern
        _valid_row("Q022", answer="Fixed Q022"),   # regenerated
    ])
    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    rows = _read_jsonl(run_dir / "results.jsonl")

    assert count == 2
    assert len(rows) == 2
    q022 = next(r for r in rows if r["question_id"] == "Q022")
    assert q022["answer"] == "Fixed Q022"
    assert not q022.get("error")


# ── error row replaced ─────────────────────────────────────────────────────────

def test_compact_replaces_error_row(tmp_path):
    """Original error row is superseded by the regenerated valid row."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "results.jsonl", [
        _valid_row("Q001"),
        _invalid_row("Q057", error="OpenAI timeout"),
        _valid_row("Q057", answer="Retry success"),
    ])
    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    rows = _read_jsonl(run_dir / "results.jsonl")

    assert count == 2
    q057 = next(r for r in rows if r["question_id"] == "Q057")
    assert q057["answer"] == "Retry success"
    assert not q057.get("error")


def test_compact_removes_unrepaired_invalid_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "results.jsonl", [
        _valid_row("Q001"),
        _invalid_row("Q002", answer=""),
        _invalid_row("Q003", answer="Recovered text", error="provider error"),
    ])

    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    rows = _read_jsonl(run_dir / "results.jsonl")

    assert count == 1
    assert [row["question_id"] for row in rows] == ["Q001"]


def test_compact_drops_malformed_jsonl_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text(
        json.dumps(_valid_row("Q001")) + "\n{bad json\n" + json.dumps(_valid_row("Q002")) + "\n",
        encoding="utf-8",
    )

    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    rows = _read_jsonl(run_dir / "results.jsonl")

    assert count == 2
    assert [row["question_id"] for row in rows] == ["Q001", "Q002"]


# ── G03 scenario: 4 invalid rows repaired ─────────────────────────────────────

def test_compact_96_rows_after_4_invalid_repaired(tmp_path):
    """Exact G03 scenario: 96-question run with 4 empty-answer rows regenerated."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    invalid_ids = {"Q022", "Q023", "Q057", "Q084"}

    # Original run: 92 valid + 4 invalid = 96 rows
    rows: list[dict] = []
    for i in range(1, 97):
        qid = f"Q{i:03d}"
        if qid in invalid_ids:
            rows.append(_invalid_row(qid))
        else:
            rows.append(_valid_row(qid))

    # Resume appended 4 new valid rows (now 100 rows, 4 duplicate IDs)
    for qid in sorted(invalid_ids):
        rows.append(_valid_row(qid, f"Fixed {qid}"))

    assert len(rows) == 100
    _write_jsonl(run_dir / "results.jsonl", rows)

    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()

    result_rows = _read_jsonl(run_dir / "results.jsonl")
    ids = [r["question_id"] for r in result_rows]

    assert count == 96, f"Expected 96 rows, got {count}"
    assert len(result_rows) == 96
    assert len(set(ids)) == 96, "Duplicate question_ids remain"

    # Every fixed row has the new valid answer
    for qid in invalid_ids:
        row = next(r for r in result_rows if r["question_id"] == qid)
        assert row["answer"] == f"Fixed {qid}"
        assert not row.get("error")


# ── duplicate detection ────────────────────────────────────────────────────────

def test_compact_removes_all_duplicates(tmp_path):
    """After compaction no question_id appears more than once."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_valid_row("Q001")] * 5  # 5 copies of Q001
    _write_jsonl(run_dir / "results.jsonl", rows)
    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    result_rows = _read_jsonl(run_dir / "results.jsonl")
    assert count == 1
    assert len(result_rows) == 1


# ── ordering ───────────────────────────────────────────────────────────────────

def test_compact_sorts_q001_through_q096(tmp_path):
    """Rows are written in Q001 → Q096 deterministic order after compaction."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_valid_row(f"Q{i:03d}") for i in range(96, 0, -1)]  # Q096 first
    _write_jsonl(run_dir / "results.jsonl", rows)
    writer = ResultWriter(run_dir, save_csv=False)
    writer.compact_deduplicate()
    result_rows = _read_jsonl(run_dir / "results.jsonl")
    ids = [r["question_id"] for r in result_rows]
    assert ids == [f"Q{i:03d}" for i in range(1, 97)]


def test_compact_ordering_with_duplicates(tmp_path):
    """Ordering is correct even when duplicate rows were present before compaction."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _invalid_row("Q003"),
        _valid_row("Q001"),
        _valid_row("Q002"),
        _valid_row("Q003", "Fixed"),  # replacement for Q003
    ]
    _write_jsonl(run_dir / "results.jsonl", rows)
    writer = ResultWriter(run_dir, save_csv=False)
    writer.compact_deduplicate()
    result_rows = _read_jsonl(run_dir / "results.jsonl")
    assert [r["question_id"] for r in result_rows] == ["Q001", "Q002", "Q003"]


# ── idempotency ────────────────────────────────────────────────────────────────

def test_compact_is_idempotent(tmp_path):
    """Running compact twice produces the same result as running it once."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_valid_row(f"Q{i:03d}") for i in range(1, 97)]
    _write_jsonl(run_dir / "results.jsonl", rows)
    writer = ResultWriter(run_dir, save_csv=False)
    count1 = writer.compact_deduplicate()
    count2 = writer.compact_deduplicate()
    assert count1 == count2 == 96
    result_rows = _read_jsonl(run_dir / "results.jsonl")
    assert len(result_rows) == 96


def test_compact_idempotent_after_resume_scenario(tmp_path):
    """Compact on an already-compacted G03-like file adds no rows."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_valid_row(f"Q{i:03d}") for i in range(1, 97)]
    _write_jsonl(run_dir / "results.jsonl", rows)
    writer = ResultWriter(run_dir, save_csv=False)
    writer.compact_deduplicate()  # first compact
    count = writer.compact_deduplicate()  # second compact
    assert count == 96
    result_rows = _read_jsonl(run_dir / "results.jsonl")
    assert len(result_rows) == 96


def test_second_resume_is_idempotent_with_96_hashes_unchanged(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    invalid_ids = {"Q022", "Q023", "Q057", "Q084"}
    rows = [
        _invalid_row(qid) if qid in invalid_ids else _valid_row(qid)
        for qid in (f"Q{i:03d}" for i in range(1, 97))
    ]
    rows.extend(_valid_row(qid, f"Fixed {qid}") for qid in sorted(invalid_ids))
    _write_jsonl(run_dir / "results.jsonl", rows)

    writer = ResultWriter(run_dir, save_csv=True)
    writer.compact_deduplicate()
    first_json_hash = hashlib.sha256((run_dir / "results.jsonl").read_bytes()).hexdigest()
    first_csv_hash = hashlib.sha256((run_dir / "results.csv").read_bytes()).hexdigest()

    assert writer.load_existing_question_ids() == {f"Q{i:03d}" for i in range(1, 97)}
    writer.compact_deduplicate()

    assert hashlib.sha256((run_dir / "results.jsonl").read_bytes()).hexdigest() == first_json_hash
    assert hashlib.sha256((run_dir / "results.csv").read_bytes()).hexdigest() == first_csv_hash


# ── CSV deduplication ──────────────────────────────────────────────────────────

def test_compact_deduplicates_csv(tmp_path):
    """results.csv is also deduplicated to match results.jsonl."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fieldnames = ["question_id", "answer", "error"]
    csv_rows_in = [
        {"question_id": "Q001", "answer": "Good", "error": ""},
        {"question_id": "Q002", "answer": "", "error": ""},   # invalid
        {"question_id": "Q002", "answer": "Fixed", "error": ""},  # new valid
    ]
    _write_csv(run_dir / "results.csv", csv_rows_in, fieldnames)
    _write_jsonl(run_dir / "results.jsonl", [
        {"question_id": "Q001", "answer": "Good", "error": None},
        {"question_id": "Q002", "answer": "", "error": None},
        {"question_id": "Q002", "answer": "Fixed", "error": None},
    ])

    writer = ResultWriter(run_dir, save_csv=True)
    writer.compact_deduplicate()

    csv_result = _read_csv(run_dir / "results.csv")
    assert len(csv_result) == 2
    q002 = next(r for r in csv_result if r["question_id"] == "Q002")
    assert q002["answer"] == "Fixed"


def test_compact_csv_ordering_matches_jsonl(tmp_path):
    """CSV row order after compaction matches JSONL order."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fieldnames = ["question_id", "answer", "error"]
    csv_rows_in = [
        {"question_id": "Q003", "answer": "c", "error": ""},
        {"question_id": "Q001", "answer": "a", "error": ""},
        {"question_id": "Q002", "answer": "b", "error": ""},
    ]
    _write_csv(run_dir / "results.csv", csv_rows_in, fieldnames)
    _write_jsonl(run_dir / "results.jsonl", [
        {"question_id": "Q003", "answer": "c", "error": None},
        {"question_id": "Q001", "answer": "a", "error": None},
        {"question_id": "Q002", "answer": "b", "error": None},
    ])

    writer = ResultWriter(run_dir, save_csv=True)
    writer.compact_deduplicate()

    json_ids = [r["question_id"] for r in _read_jsonl(run_dir / "results.jsonl")]
    csv_ids = [r["question_id"] for r in _read_csv(run_dir / "results.csv")]
    assert json_ids == csv_ids == ["Q001", "Q002", "Q003"]


def test_compact_no_csv_file_is_graceful(tmp_path):
    """If results.csv does not exist, compact completes without error."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "results.jsonl", [_valid_row("Q001"), _valid_row("Q002")])
    writer = ResultWriter(run_dir, save_csv=True)  # save_csv=True but no csv on disk
    count = writer.compact_deduplicate()
    assert count == 2
    assert [row["question_id"] for row in _read_csv(run_dir / "results.csv")] == ["Q001", "Q002"]


def test_compact_jsonl_and_csv_have_identical_question_id_sets(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "results.jsonl", [
        _valid_row("Q003"),
        _invalid_row("Q002"),
        _valid_row("Q001"),
        _valid_row("Q002", "Fixed"),
    ])

    writer = ResultWriter(run_dir, save_csv=True)
    writer.compact_deduplicate()

    json_ids = [row["question_id"] for row in _read_jsonl(run_dir / "results.jsonl")]
    csv_ids = [row["question_id"] for row in _read_csv(run_dir / "results.csv")]
    assert json_ids == csv_ids == ["Q001", "Q002", "Q003"]


# ── atomic write safety ────────────────────────────────────────────────────────

def test_compact_uses_temp_file_for_atomic_replace(tmp_path):
    """compact_deduplicate calls os.replace with a .tmp source path."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "results.jsonl", [
        _invalid_row("Q001"),
        _valid_row("Q001", "replaced"),
    ])

    seen_srcs: list[str] = []
    real_replace = os.replace

    def spy_replace(src, dst):
        seen_srcs.append(str(src))   # Path objects on Windows — stringify
        real_replace(src, dst)

    with mock.patch("src.pipeline1.io.result_writer.os.replace", side_effect=spy_replace):
        writer = ResultWriter(run_dir, save_csv=False)
        writer.compact_deduplicate()

    assert any(s.endswith(".tmp") for s in seen_srcs), (
        "Expected os.replace to be called with a .tmp source; calls: " + str(seen_srcs)
    )


def test_compact_original_not_destroyed_if_atomic_swap_fails(tmp_path):
    """If os.replace fails the original file is untouched and the temp remains.

    The implementation writes a temp file first; if the final rename fails the
    original is still readable.  This test simulates a failure at the swap step.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original_content = (
        json.dumps(_valid_row("Q001")) + "\n"
        + json.dumps(_invalid_row("Q002")) + "\n"
    )
    (run_dir / "results.jsonl").write_text(original_content, encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError("Simulated rename failure")

    with mock.patch("src.pipeline1.io.result_writer.os.replace", side_effect=failing_replace):
        writer = ResultWriter(run_dir, save_csv=False)
        with pytest.raises(OSError, match="Simulated rename failure"):
            writer.compact_deduplicate()

    # Original file must be intact (the swap never happened).
    assert (run_dir / "results.jsonl").read_text(encoding="utf-8") == original_content
    # Temp file was written before the failed swap → two-phase atomic approach confirmed.
    assert (run_dir / "results.jsonl.tmp").exists()


# ── edge cases ─────────────────────────────────────────────────────────────────

def test_compact_empty_file_returns_zero(tmp_path):
    """Empty results.jsonl returns 0 and no error."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text("", encoding="utf-8")
    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    assert count == 0


def test_compact_missing_file_returns_zero(tmp_path):
    """Missing results.jsonl returns 0 and no error."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    assert count == 0


def test_compact_does_not_invent_missing_rows(tmp_path):
    """compact_deduplicate never adds rows that were not already present."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_valid_row(f"Q{i:03d}") for i in range(1, 96)]  # Q001-Q095, Q096 missing
    _write_jsonl(run_dir / "results.jsonl", rows)
    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    assert count == 95
    result_ids = {r["question_id"] for r in _read_jsonl(run_dir / "results.jsonl")}
    assert "Q096" not in result_ids


def test_manifest_stats_use_reconciled_rows_and_detect_missing_ids():
    class Query:
        def __init__(self, question_id: str) -> None:
            self.question_id = question_id

    queries = [Query(f"Q{i:03d}") for i in range(1, 97)]
    rows = [_valid_row(f"Q{i:03d}") for i in range(1, 96)]

    stats = _reconciled_result_stats(queries, rows)

    assert stats["expected_questions"] == 96
    assert stats["processed_questions"] == 95
    assert stats["successful_questions"] == 95
    assert stats["failed_questions"] == 1
    assert stats["missing_question_ids"] == ["Q096"]
    assert stats["run_status"] == "FAIL"


def test_manifest_stats_pass_for_canonical_q001_through_q096():
    class Query:
        def __init__(self, question_id: str) -> None:
            self.question_id = question_id

    queries = [Query(f"Q{i:03d}") for i in range(1, 97)]
    rows = [_valid_row(f"Q{i:03d}") for i in range(1, 97)]

    stats = _reconciled_result_stats(queries, rows)

    assert stats["expected_questions"] == 96
    assert stats["processed_questions"] == 96
    assert stats["successful_questions"] == 96
    assert stats["failed_questions"] == 0
    assert stats["run_status"] == "PASS"


# ── generated_answer alias support ────────────────────────────────────────────

def test_compact_keeps_rows_with_generated_answer_alias(tmp_path):
    """Rows using the generated_answer alias (pre-canonicalisation) are preserved."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "results.jsonl", [
        {"question_id": "Q001", "generated_answer": "OK", "error": None},
        {"question_id": "Q001", "generated_answer": "Better", "error": None},
    ])
    writer = ResultWriter(run_dir, save_csv=False)
    count = writer.compact_deduplicate()
    rows = _read_jsonl(run_dir / "results.jsonl")
    assert count == 1
    assert rows[0]["generated_answer"] == "Better"
