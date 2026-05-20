PROMPT_TEMPLATE_VERSION = "v4_optional_metadata_headers_and_template_prompts"


def build_prompt(system_prompt: str, question: str, contexts: list, include_metadata_headers: bool = False) -> str:
    context_text = "\n\n".join(
        f"[{idx}] {_format_context(item, include_metadata_headers)}" for idx, item in enumerate(contexts, start=1)
    )
    if "{context}" in system_prompt or "{question}" in system_prompt:
        return system_prompt.strip().format(context=context_text, question=question)
    return (
        f"{system_prompt.strip()}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved Context:\n{context_text}\n\n"
        "Final Answer:"
    )


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
