import json
from functools import lru_cache
from pathlib import Path

ORCHESTRATION_PROMPT_VERSION = "sivas_orchestration_v1"
DEFAULT_ORCHESTRATION_PROMPT_PATH = "src/pipeline1/prompts/orchestration_prompt.txt"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_orchestration_prompt(
    question: str,
    categories: list[str],
    prompt_path: str | None = None,
) -> str:
    resolved = _resolve_prompt_path(prompt_path)
    categories_json = json.dumps(categories, ensure_ascii=False)
    question_json = json.dumps(question, ensure_ascii=False)
    null_json = json.dumps(None)
    return (
        _load_prompt_template(resolved)
        .replace("{{categories_json}}", categories_json)
        .replace("{{module_json}}", null_json)
        .replace("{{program_json}}", null_json)
        .replace("{{role_json}}", null_json)
        .replace("{{role_description_json}}", null_json)
        .replace("{{question_json}}", question_json)
    )


def _resolve_prompt_path(prompt_path: str | None) -> Path:
    if prompt_path is None:
        return (_PROJECT_ROOT / DEFAULT_ORCHESTRATION_PROMPT_PATH).resolve()
    path = Path(prompt_path)
    resolved = path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Orchestration prompt file not found: {resolved} "
            f"(prompt_path={prompt_path!r})"
        )
    if resolved.stat().st_size == 0:
        raise ValueError(f"Orchestration prompt file is empty: {resolved}")
    return resolved


@lru_cache(maxsize=8)
def _load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
