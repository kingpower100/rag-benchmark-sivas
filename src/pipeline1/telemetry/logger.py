import logging
from pathlib import Path


def build_logger(log_path: Path, level: str = "INFO", extra_log_paths: list[Path] | None = None) -> logging.Logger:
    logger = logging.getLogger("pipeline1")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for path in [log_path, *(extra_log_paths or [])]:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
