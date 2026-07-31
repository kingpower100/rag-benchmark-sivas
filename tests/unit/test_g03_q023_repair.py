from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import pytest

from scripts.repair.g03_q023_repair import (
    EXPECTED_G03_IDS,
    REPAIR_QUESTION_ID,
    create_q023_dataset,
    merge_g03_q023_repair,
    repair_config_diff_report,
    validate_g03_repair_outputs,
)
from src.evaluation.source_validation import validate_pipeline1_source
from src.pipeline3.stages.validation_stage import _failed_pipeline1_manifests


def test_q023_extraction_produces_exactly_one_row(tmp_path):
    source = tmp_path / "questions_fixed.jsonl"
    output = tmp_path / "data" / "repair" / "questions_Q023_only.jsonl"
    rows = [
        {"question_id": "Q001", "frage": "one", "extra": 1},
        {"question_id": "Q023", "frage": "twenty three", "extra": {"kept": True}},
        {"question_id": "Q096", "frage": "last", "extra": 96},
    ]
    _write_jsonl(source, rows)

    report = create_q023_dataset(source, output)
    extracted = _read_jsonl(output)

    assert report["row_count"] == 1
    assert extracted == [rows[1]]
    assert _read_jsonl(source) == rows


def test_repair_config_diff_only_contains_allowed_fields():
    report = repair_config_diff_report()
    paths = {item["path"] for item in report["allowed_differences"]}

    assert report["valid"] is True
    assert report["disallowed_differences"] == []
    assert paths == {
        "experiment.experiment_id",
        "data.questions_path",
        "generation.reasoning_effort",
        "runtime.resume",
    }


def test_merge_validates_repair_row_and_rejects_invalid(tmp_path):
    official_dir, repair_dir, official_cfg, repair_cfg = _build_runs(tmp_path, repair_answer="")

    with pytest.raises(RuntimeError, match="Repair row is invalid"):
        merge_g03_q023_repair(official_dir, repair_dir, official_cfg, repair_cfg, tmp_path / "backup")


def test_merge_rejects_repair_row_without_finish_reason(tmp_path):
    official_dir, repair_dir, official_cfg, repair_cfg = _build_runs(tmp_path, completion_diagnostics={})

    with pytest.raises(RuntimeError, match="completion_diagnostics.finish_reason"):
        merge_g03_q023_repair(official_dir, repair_dir, official_cfg, repair_cfg, tmp_path / "backup")


def test_merge_rejects_non_pass_repair_manifest(tmp_path):
    official_dir, repair_dir, official_cfg, repair_cfg = _build_runs(tmp_path)
    manifest = _read_json(repair_dir / "run_manifest.json")
    manifest["run_status"] = "FAIL"
    manifest["failed_questions"] = 1
    _write_json(repair_dir / "run_manifest.json", manifest)

    with pytest.raises(RuntimeError, match="PASS"):
        merge_g03_q023_repair(official_dir, repair_dir, official_cfg, repair_cfg, tmp_path / "backup")


def test_merge_inserts_q023_and_records_provenance(tmp_path):
    official_dir, repair_dir, official_cfg, repair_cfg = _build_runs(tmp_path)
    backup_dir = tmp_path / "G03_before_Q023_repair"

    result = merge_g03_q023_repair(official_dir, repair_dir, official_cfg, repair_cfg, backup_dir)

    rows = _read_jsonl(official_dir / "results.jsonl")
    csv_rows = _read_csv(official_dir / "results.csv")
    manifest = _read_json(official_dir / "run_manifest.json")
    provenance = manifest["repair_provenance"]

    assert result["row_count"] == 96
    assert [row["question_id"] for row in rows] == EXPECTED_G03_IDS
    assert [row["question_id"] for row in csv_rows] == EXPECTED_G03_IDS
    assert len({row["question_id"] for row in rows}) == 96
    repaired_row = next(row for row in rows if row["question_id"] == REPAIR_QUESTION_ID)
    assert repaired_row["answer"] == "Fixed Q023"
    assert repaired_row["experiment_id"] == "G03"
    assert repaired_row["config_id"] == "G03"
    assert repaired_row["repair_provenance"]["repair_experiment_id"] == "G03_Q023_repair"
    assert repaired_row["repair_provenance"]["repair_generation_overrides"] == {
        "reasoning_effort": "none",
        "max_tokens": 512,
    }
    assert manifest["processed_questions"] == 96
    assert manifest["successful_questions"] == 96
    assert manifest["failed_questions"] == 0
    assert manifest["run_status"] == "PASS"
    assert manifest["run_stats"]["n_queries"] == 96
    assert manifest["run_stats"]["run_status"] == "PASS"
    assert provenance["repaired_question_ids"] == ["Q023"]
    assert provenance["repair_experiment_id"] == "G03_Q023_repair"
    assert provenance["original_experiment_id"] == "G03"
    assert provenance["original_config_hash"]
    assert provenance["repair_config_hash"]
    assert provenance["repair_generation_overrides"] == {"reasoning_effort": "none", "max_tokens": 512}
    assert backup_dir.exists()
    assert validate_g03_repair_outputs(official_dir)["valid"] is True


def test_merge_is_idempotent_except_repair_timestamp(tmp_path):
    official_dir, repair_dir, official_cfg, repair_cfg = _build_runs(tmp_path)
    backup_dir = tmp_path / "G03_before_Q023_repair"

    merge_g03_q023_repair(official_dir, repair_dir, official_cfg, repair_cfg, backup_dir)
    first_rows = (official_dir / "results.jsonl").read_text(encoding="utf-8")
    first_csv = (official_dir / "results.csv").read_text(encoding="utf-8")
    first_manifest = _read_json(official_dir / "run_manifest.json")

    merge_g03_q023_repair(official_dir, repair_dir, official_cfg, repair_cfg, backup_dir)
    second_manifest = _read_json(official_dir / "run_manifest.json")

    assert (official_dir / "results.jsonl").read_text(encoding="utf-8") == first_rows
    assert (official_dir / "results.csv").read_text(encoding="utf-8") == first_csv
    first_manifest["repair_provenance"]["repair_timestamp"] = "<ignored>"
    second_manifest["repair_provenance"]["repair_timestamp"] = "<ignored>"
    assert second_manifest == first_manifest


def test_merge_uses_atomic_replaces(tmp_path, monkeypatch):
    official_dir, repair_dir, official_cfg, repair_cfg = _build_runs(tmp_path)
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(src, dst):
        calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("scripts.repair.g03_q023_repair.os.replace", spy_replace)

    merge_g03_q023_repair(official_dir, repair_dir, official_cfg, repair_cfg, tmp_path / "backup")

    assert any(src.endswith(".jsonl.tmp") for src, _ in calls)
    assert any(src.endswith(".csv.tmp") for src, _ in calls)
    assert any(src.endswith(".json.tmp") for src, _ in calls)


def test_pipeline2_and_pipeline3_tolerate_repair_provenance(tmp_path):
    official_dir, repair_dir, official_cfg, repair_cfg = _build_runs(tmp_path)
    merge_g03_q023_repair(official_dir, repair_dir, official_cfg, repair_cfg, tmp_path / "backup")
    rows = _read_jsonl(official_dir / "results.jsonl")
    manifest = _read_json(official_dir / "run_manifest.json")

    class SourceValidation:
        expected_experiment_id = "G03"
        expected_retriever_type = "elasticsearch_hybrid_rrf"
        expected_orchestration_enabled = False
        require_hybrid_diagnostics = True
        require_reranker_diagnostics = True
        require_routing_diagnostics = False
        require_routing_reconciliation = False

    validate_pipeline1_source(
        results_path=official_dir / "results.jsonl",
        manifest=manifest,
        rows=rows,
        source_validation=SourceValidation(),
        pipeline_name="Pipeline 2",
    )
    assert _failed_pipeline1_manifests([manifest]) == []


def _build_runs(
    tmp_path: Path,
    *,
    repair_answer: str = "Fixed Q023",
    completion_diagnostics: dict | None = None,
) -> tuple[Path, Path, Path, Path]:
    official_dir = tmp_path / "data" / "runs" / "pipeline1" / "G03"
    repair_dir = tmp_path / "data" / "runs" / "pipeline1" / "G03_Q023_repair"
    official_dir.mkdir(parents=True)
    repair_dir.mkdir(parents=True)
    official_cfg = tmp_path / "configs" / "pipeline1" / "final_experiments" / "G03_gpt55.yaml"
    repair_cfg = tmp_path / "configs" / "pipeline1" / "final_experiments" / "G03_Q023_repair.yaml"
    official_cfg.parent.mkdir(parents=True)
    official_cfg.write_text("experiment:\n  experiment_id: G03\n", encoding="utf-8")
    repair_cfg.write_text("experiment:\n  experiment_id: G03_Q023_repair\n", encoding="utf-8")

    official_rows = [_row(qid, f"Answer {qid}") for qid in EXPECTED_G03_IDS if qid != "Q023"]
    repair_row = _row("Q023", repair_answer, experiment_id="G03_Q023_repair")
    repair_row["completion_diagnostics"] = (
        {"finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 20}
        if completion_diagnostics is None
        else completion_diagnostics
    )
    _write_jsonl(official_dir / "results.jsonl", official_rows)
    _write_csv(official_dir / "results.csv", official_rows)
    _write_jsonl(repair_dir / "results.jsonl", [repair_row])
    _write_csv(repair_dir / "results.csv", [repair_row])
    _write_json(official_dir / "run_manifest.json", _manifest("G03", 96, 95, failed=1, status="FAIL"))
    _write_json(official_dir / "manifest.json", _manifest("G03", 96, 95, failed=1, status="FAIL"))
    _write_json(repair_dir / "run_manifest.json", _manifest("G03_Q023_repair", 1, 1))
    _write_json(repair_dir / "manifest.json", _manifest("G03_Q023_repair", 1, 1))
    return official_dir, repair_dir, official_cfg, repair_cfg


def _row(question_id: str, answer: str, *, experiment_id: str = "G03") -> dict:
    return {
        "question_id": question_id,
        "experiment_id": experiment_id,
        "config_id": experiment_id,
        "question": f"Question {question_id}",
        "answer": answer,
        "generated_answer": answer,
        "error": None,
        "retriever_type": "elasticsearch_hybrid_rrf",
        "reranker_applied": True,
        "retrieval_diagnostics": {
            "retriever_type": "elasticsearch_hybrid_rrf",
            "retrieval_scope": "global",
            "final_context_count": 5,
            "top_k": 5,
            "dense_candidate_count": 20,
            "bm25_candidate_count": 20,
            "fused_candidate_count": 20,
            "reranked_candidate_count": 5,
            "reranker_applied": True,
        },
    }


def _manifest(run_id: str, expected: int, successful: int, *, failed: int = 0, status: str = "PASS") -> dict:
    return {
        "run_id": run_id,
        "config_hash": f"{run_id}-hash",
        "expected_questions": expected,
        "processed_questions": successful,
        "successful_questions": successful,
        "failed_questions": failed,
        "run_status": status,
        "run_stats": {
            "n_queries": expected,
            "expected_questions": expected,
            "processed_questions": successful,
            "successful_questions": successful,
            "failed_questions": failed,
            "run_status": status,
        },
        "models": {"retriever_type": "elasticsearch_hybrid_rrf"},
        "orchestration_enabled": False,
        "artifacts": {},
        "start_timestamp_utc": "2026-01-01T00:00:00+00:00",
        "end_timestamp_utc": "2026-01-01T00:00:01+00:00",
        "resolved_config": {"experiment": {"experiment_id": run_id}},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
