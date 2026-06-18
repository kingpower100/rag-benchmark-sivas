import json
from functools import lru_cache
from pathlib import Path


ORCHESTRATION_PROMPT_VERSION = "sivas_orchestration_v1"
ORCHESTRATION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "orchestration_prompt.txt"


def build_orchestration_prompt(question: str, categories: list[str]) -> str:
    categories_json = json.dumps(categories, ensure_ascii=False)
    question_json = json.dumps(question, ensure_ascii=False)
    return (
        _load_orchestration_prompt_template()
        .replace("{{categories_json}}", categories_json)
        .replace("{{question_json}}", question_json)
    )


@lru_cache(maxsize=1)
def _load_orchestration_prompt_template() -> str:
    return ORCHESTRATION_PROMPT_PATH.read_text(encoding="utf-8").strip()
