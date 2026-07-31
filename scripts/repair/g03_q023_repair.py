from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline1.io.result_writer import _is_valid_result_row, _qid_sort_key
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.utils.hashing import file_sha256

EXPECTED_G03_IDS = [f"Q{i:03d}" for i in range(1, 97)]
REPAIR_QUESTION_ID = "Q023"
REPAIR_EXPERIMENT_ID = "G03_Q023_repair"
REPAIR_REASON = "Original generation returned empty visible answer with finish_reason=length"
REPAIR_GENERATION_OVERRIDES = {"reasoning_effort": "none", "max_tokens": 512}
ALLOWED_REPAIR_CONFIG_DIFFS = {
    ("experiment", "experiment_id"),
    ("data", "questions_path"),
    ("generation", "reasoning_effort"),
    ("runtime", "resume"),
}


def create_q023_dataset(
    source_path: Path = Path("data/raw/questions_fixed.jsonl"),
    output_path: Path = Path("data/repair/questions_Q023_only.jsonl"),
) -> dict[str, Any]:
    rows = _read_jsonl(source_path)
    matches = [row for row in rows if str(row.get("question_id")) == REPAIR_QUESTION_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {REPAIR_QUESTION_ID} row in {source_path}, found {len(matches)}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_path, matches)
    written = _read_jsonl(output_path)
    if len(written) != 1 or str(written[0].get("question_id")) != REPAIR_QUESTION_ID:
        raise RuntimeError(f"Repair dataset validation failed: {output_path}")
    return {"path": str(output_path), "row_count": 1, "question_id": REPAIR_QUESTION_ID}


def merge_g03_q023_repair(
    official_run_dir: Path = Path("data/runs/pipeline1/G03"),
    repair_run_dir: Path = Path("data/runs/pipeline1/G03_Q023_repair"),
    official_config_path: Path = Path("configs/pipeline1/final_experiments/G03_gpt55.yaml"),
    repair_config_path: Path = Path("configs/pipeline1/final_experiments/G03_Q023_repair.yaml"),
    backup_dir: Path = Path("data/runs/pipeline1/G03_before_Q023_repair"),
) -> dict[str, Any]:
    official_run_dir = official_run_dir.resolve()
    repair_run_dir = repair_run_dir.resolve()
    backup_dir = backup_dir.resolve()

    _ensure_run_dir(official_run_dir, "official G03")
    _ensure_run_dir(repair_run_dir, "G03 Q023 repair")
    if not backup_dir.exists():
        shutil.copytree(official_run_dir, backup_dir)

    official_manifest = _load_manifest(official_run_dir)
    repair_manifest = _load_manifest(repair_run_dir)
    official_rows = _canonical_rows(_read_jsonl(official_run_dir / "results.jsonl"))
    repair_row = _load_valid_repair_row(repair_run_dir)
    official_repair_row = _repair_row_for_official_g03(repair_row)

    by_id = {str(row["question_id"]): row for row in official_rows}
    by_id[REPAIR_QUESTION_ID] = official_repair_row
    merged_rows = [by_id[qid] for qid in EXPECTED_G03_IDS if qid in by_id]
    _validate_merged_rows(merged_rows)

    _atomic_write_jsonl(official_run_dir / "results.jsonl", merged_rows)
    _atomic_write_csv(official_run_dir / "results.csv", merged_rows)

    updated_manifest = _updated_official_manifest(
        manifest=official_manifest,
        repair_manifest=repair_manifest,
        official_run_dir=official_run_dir,
        official_config_path=official_config_path,
        repair_config_path=repair_config_path,
        backup_dir=backup_dir,
        merged_rows=merged_rows,
    )
    _atomic_write_json(official_run_dir / "run_manifest.json", updated_manifest)
    _atomic_write_json(official_run_dir / "manifest.json", updated_manifest)

    return {
        "official_run_dir": str(official_run_dir),
        "repair_run_dir": str(repair_run_dir),
        "backup_dir": str(backup_dir),
        "row_count": len(merged_rows),
        "repaired_question_ids": [REPAIR_QUESTION_ID],
        "run_status": updated_manifest["run_status"],
    }


def repair_config_diff_report(
    official_config_path: Path = Path("configs/pipeline1/final_experiments/G03_gpt55.yaml"),
    repair_config_path: Path = Path("configs/pipeline1/final_experiments/G03_Q023_repair.yaml"),
) -> dict[str, Any]:
    official = PipelineConfig.from_yaml(str(official_config_path)).model_dump()
    repair = PipelineConfig.from_yaml(str(repair_config_path)).model_dump()
    diffs = _diff_dicts(official, repair)
    disallowed = [
        {"path": ".".join(path), "official": left, "repair": right}
        for path, left, right in diffs
        if tuple(path) not in ALLOWED_REPAIR_CONFIG_DIFFS
    ]
    return {
        "allowed_differences": [
            {"path": ".".join(path), "official": left, "repair": right}
            for path, left, right in diffs
            if tuple(path) in ALLOWED_REPAIR_CONFIG_DIFFS
        ],
        "disallowed_differences": disallowed,
        "valid": not disallowed,
    }


def validate_g03_repair_outputs(run_dir: Path = Path("data/runs/pipeline1/G03")) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    json_rows = _read_jsonl(run_dir / "results.jsonl")
    csv_rows = _read_csv(run_dir / "results.csv")
    manifest = _load_manifest(run_dir)
    json_ids = [str(row.get("question_id")) for row in json_rows]
    csv_ids = [str(row.get("question_id")) for row in csv_rows]
    empty = [row.get("question_id") for row in json_rows if not _answer_text(row)]
    errors = [row.get("question_id") for row in json_rows if row.get("error") is not None]
    duplicates = sorted({qid for qid in json_ids if json_ids.count(qid) > 1})
    missing = [qid for qid in EXPECTED_G03_IDS if qid not in set(json_ids)]
    provenance = manifest.get("repair_provenance") if isinstance(manifest, dict) else {}
    ok = (
        len(json_rows) == 96
        and len(csv_rows) == 96
        and len(set(json_ids)) == 96
        and json_ids == csv_ids == EXPECTED_G03_IDS
        and not empty
        and not errors
        and not duplicates
        and not missing
        and manifest.get("run_status") == "PASS"
        and manifest.get("failed_questions") == 0
        and provenance.get("repaired_question_ids") == [REPAIR_QUESTION_ID]
        and provenance.get("repair_experiment_id") == REPAIR_EXPERIMENT_ID
    )
    report = {
        "jsonl_rows": len(json_rows),
        "csv_rows": len(csv_rows),
        "jsonl_unique_ids": len(set(json_ids)),
        "csv_unique_ids": len(set(csv_ids)),
        "empty_answers": empty,
        "errors": errors,
        "duplicates": duplicates,
        "missing": missing,
        "jsonl_csv_id_parity": json_ids == csv_ids,
        "canonical_ordering": json_ids == EXPECTED_G03_IDS,
        "manifest_pass": manifest.get("run_status") == "PASS" and manifest.get("failed_questions") == 0,
        "repair_provenance_present": bool(provenance),
        "valid": ok,
    }
    if not ok:
        raise RuntimeError(f"G03 repair validation failed: {report}")
    return report


def _load_valid_repair_row(repair_run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(repair_run_dir / "results.jsonl")
    if len(rows) != 1:
        raise RuntimeError(f"Repair results must contain exactly one row, found {len(rows)}.")
    row = rows[0]
    if str(row.get("question_id")) != REPAIR_QUESTION_ID:
        raise RuntimeError(f"Repair row question_id must be {REPAIR_QUESTION_ID}, got {row.get('question_id')!r}.")
    if not _is_valid_result_row(row):
        raise RuntimeError("Repair row is invalid: answer must be non-empty and error must be null.")
    diagnostics = row.get("completion_diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics.get("finish_reason"):
        raise RuntimeError("Repair row must include completion_diagnostics.finish_reason.")
    manifest = _load_manifest(repair_run_dir)
    run_stats = manifest.get("run_stats") if isinstance(manifest.get("run_stats"), dict) else {}
    if (manifest.get("run_status") or run_stats.get("run_status")) != "PASS":
        raise RuntimeError("Repair manifest must report PASS before merge.")
    if int(manifest.get("failed_questions", run_stats.get("failed_questions", -1))) != 0:
        raise RuntimeError("Repair manifest must report zero failed questions before merge.")
    return row


def _repair_row_for_official_g03(row: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(row))
    merged["experiment_id"] = "G03"
    if "config_id" in merged:
        merged["config_id"] = "G03"
    merged["repair_provenance"] = {
        "repair_experiment_id": REPAIR_EXPERIMENT_ID,
        "repair_reason": REPAIR_REASON,
        "repair_generation_overrides": dict(REPAIR_GENERATION_OVERRIDES),
    }
    return merged


def _updated_official_manifest(
    *,
    manifest: dict[str, Any],
    repair_manifest: dict[str, Any],
    official_run_dir: Path,
    official_config_path: Path,
    repair_config_path: Path,
    backup_dir: Path,
    merged_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(manifest))
    original_config_hash = manifest.get("config_hash") or (
        file_sha256(official_config_path) if official_config_path.exists() else None
    )
    repair_config_hash = repair_manifest.get("config_hash") or (
        file_sha256(repair_config_path) if repair_config_path.exists() else None
    )
    counts = {"results.jsonl": len(merged_rows), "results.csv": len(merged_rows)}

    updated.update(
        {
            "expected_questions": 96,
            "processed_questions": 96,
            "successful_questions": 96,
            "failed_questions": 0,
            "run_status": "PASS",
            "output_row_counts": counts,
        }
    )
    run_stats = updated.setdefault("run_stats", {})
    run_stats.update(
        {
            "n_queries": 96,
            "attempted": run_stats.get("attempted"),
            "written": run_stats.get("written"),
            "expected_questions": 96,
            "processed_questions": 96,
            "successful_questions": 96,
            "failed_questions": 0,
            "run_status": "PASS",
        }
    )
    updated["failed_question_ids"] = []
    updated["repair_provenance"] = {
        "repaired_question_ids": [REPAIR_QUESTION_ID],
        "repair_experiment_id": REPAIR_EXPERIMENT_ID,
        "repair_reason": REPAIR_REASON,
        "repair_generation_overrides": dict(REPAIR_GENERATION_OVERRIDES),
        "original_experiment_id": "G03",
        "original_config_hash": original_config_hash,
        "repair_config_hash": repair_config_hash,
        "repair_timestamp": _utc_now(),
        "repair_run_dir": str(official_run_dir.parent / REPAIR_EXPERIMENT_ID),
        "backup_dir": str(backup_dir),
        "scientific_policy": (
            "G03 repaired mixed-runtime run: 95 original rows plus one targeted "
            "Q023 repair row generated with reasoning_effort='none'."
        ),
    }
    artifacts = updated.setdefault("artifacts", {})
    for name in ("results.jsonl", "results.csv"):
        path = official_run_dir / name
        artifacts[name] = {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return updated


def _validate_merged_rows(rows: list[dict[str, Any]]) -> None:
    ids = [str(row.get("question_id")) for row in rows]
    if ids != EXPECTED_G03_IDS:
        missing = [qid for qid in EXPECTED_G03_IDS if qid not in set(ids)]
        duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
        raise RuntimeError(f"Merged rows are not canonical. missing={missing}, duplicates={duplicates}, ids={ids}")
    invalid = [qid for qid, row in zip(ids, rows) if not _is_valid_result_row(row)]
    if invalid:
        raise RuntimeError(f"Merged rows contain invalid canonical rows: {invalid}")


def _canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if _is_valid_result_row(row):
            by_id[str(row["question_id"])] = row
    return [by_id[qid] for qid in sorted(by_id.keys(), key=_qid_sort_key)]


def _diff_dicts(left: Any, right: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        diffs: list[tuple[tuple[str, ...], Any, Any]] = []
        for key in sorted(set(left) | set(right)):
            diffs.extend(_diff_dicts(left.get(key), right.get(key), (*prefix, str(key))))
        return diffs
    if left != right:
        return [(prefix, left, right)]
    return []


def _ensure_run_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} run directory: {path}")
    for name in ("results.jsonl", "run_manifest.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing {label} artifact: {path / name}")


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        path = run_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _answer_text(row: dict[str, Any]) -> str:
    return str(row.get("answer") if row.get("answer") is not None else row.get("generated_answer") or "").strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = _csv_fieldnames(rows)
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G03 Q023 targeted repair utilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create-dataset", help="Extract data/repair/questions_Q023_only.jsonl.")
    sub.add_parser("merge", help="Merge validated G03_Q023_repair row into official G03.")
    sub.add_parser("validate", help="Validate repaired official G03 artifacts.")
    sub.add_parser("config-diff", help="Report resolved config differences between G03 and repair config.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "create-dataset":
        print(json.dumps(create_q023_dataset(), indent=2))
    elif args.command == "merge":
        print(json.dumps(merge_g03_q023_repair(), indent=2))
    elif args.command == "validate":
        print(json.dumps(validate_g03_repair_outputs(), indent=2))
    elif args.command == "config-diff":
        print(json.dumps(repair_config_diff_report(), indent=2))


if __name__ == "__main__":
    main()
