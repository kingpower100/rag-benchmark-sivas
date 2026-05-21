from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROMPT_TEMPLATE_VERSION = "v5_ranked_context_budget"


@dataclass(frozen=True)
class PromptBudget:
    max_prompt_tokens: int = 8192
    max_context_tokens: int = 6000
    max_chunk_tokens: int = 1800
    max_context_chars: int = 24000
    max_chunk_chars: int = 8000
    tokenizer_name: str = "cl100k_base"
    context_truncation_strategy: str = "ranked_budget"


def build_prompt(system_prompt: str, question: str, contexts: list, include_metadata_headers: bool = False) -> str:
    prompt, _ = build_prompt_with_stats(system_prompt, question, contexts, include_metadata_headers)
    return prompt


def build_prompt_with_stats(
    system_prompt: str,
    question: str,
    contexts: list,
    include_metadata_headers: bool = False,
    budget: PromptBudget | None = None,
) -> tuple[str, dict[str, Any]]:
    budget = budget or PromptBudget()
    encoder = _load_token_encoder(budget.tokenizer_name)
    raw_context_texts = [_format_context(item, include_metadata_headers) for item in contexts]
    budgeted_contexts, context_stats = _budget_contexts(raw_context_texts, budget, encoder)
    context_text = "\n\n".join(f"[{idx}] {text}" for idx, text in enumerate(budgeted_contexts, start=1))
    if "{context}" in system_prompt or "{question}" in system_prompt:
        prompt = system_prompt.strip().format(context=context_text, question=question)
    else:
        prompt = (
            f"{system_prompt.strip()}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            "Final Answer:"
        )
    prompt = _truncate_to_token_budget(prompt, budget.max_prompt_tokens, encoder)
    stats = {
        **context_stats,
        "prompt_chars": len(prompt),
        "prompt_tokens": _token_count(prompt, encoder),
    }
    return prompt, stats


def dedupe_prompt_contexts(contexts: list) -> list:
    seen = set()
    output = []
    for item in contexts:
        key = " ".join(item.text.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _format_context(item, include_metadata_headers: bool) -> str:
    if not include_metadata_headers:
        return item.text
    metadata = getattr(item, "metadata", {}) or {}
    pieces = []
    for label, value in (
        ("Company", metadata.get("company_name")),
        ("Symbol", metadata.get("company_symbol")),
        ("Year", metadata.get("year") or metadata.get("report_year")),
        ("Month", metadata.get("month")),
        ("Source file", metadata.get("source_file")),
        ("Page", metadata.get("page_number")),
    ):
        if value is not None and str(value).strip():
            pieces.append(f"{label}: {value}")
    if not pieces:
        return item.text
    return f"[{' | '.join(pieces)}]\n{item.text}"


def _budget_contexts(texts: list[str], budget: PromptBudget, encoder) -> tuple[list[str], dict[str, Any]]:
    output: list[str] = []
    before_chars = sum(len(text) for text in texts)
    before_tokens = sum(_token_count(text, encoder) for text in texts)
    after_chars = 0
    after_tokens = 0
    chunks_truncated = 0
    for text in texts:
        trimmed = _truncate_chunk(text, budget, encoder)
        if trimmed != text:
            chunks_truncated += 1
        chunk_chars = len(trimmed)
        chunk_tokens = _token_count(trimmed, encoder)
        if not trimmed.strip():
            continue
        if after_chars + chunk_chars > budget.max_context_chars or after_tokens + chunk_tokens > budget.max_context_tokens:
            break
        output.append(trimmed)
        after_chars += chunk_chars
        after_tokens += chunk_tokens
    return output, {
        "context_chars_before": before_chars,
        "context_chars_after": after_chars,
        "context_tokens_before": before_tokens,
        "context_tokens_after": after_tokens,
        "chunks_before": len(texts),
        "chunks_after": len(output),
        "chunks_truncated": chunks_truncated,
        "chunks_dropped": len(texts) - len(output),
    }


def _truncate_chunk(text: str, budget: PromptBudget, encoder) -> str:
    trimmed = text[: budget.max_chunk_chars]
    return _truncate_to_token_budget(trimmed, budget.max_chunk_tokens, encoder).strip()


def _truncate_to_token_budget(text: str, max_tokens: int, encoder) -> str:
    if _token_count(text, encoder) <= max_tokens:
        return text
    if encoder is not None:
        return encoder.decode(encoder.encode(text)[:max_tokens])
    return " ".join(text.split()[:max_tokens])


def _token_count(text: str, encoder) -> int:
    if encoder is not None:
        return len(encoder.encode(text))
    return len((text or "").split())


def _load_token_encoder(tokenizer_name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(tokenizer_name)
    except Exception:
        return None
