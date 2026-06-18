import csv
import json
from pathlib import Path

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
        if not self.jsonl_path.exists():
            return set()
        ids = set()
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(str(json.loads(line).get("question_id")))
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
