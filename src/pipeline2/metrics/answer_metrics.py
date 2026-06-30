from __future__ import annotations

import re
from functools import lru_cache
from importlib import metadata
from typing import Any, Iterable


DEFAULT_ABSTENTION_PATTERNS = (
    # English
    "unknown",
    "not found",
    "n/a",
    "na",
    "cannot determine",
    "can't determine",
    "insufficient information",
    # German
    "unbekannt",
    "nicht gefunden",
    "nicht verfügbar",
    "nicht bekannt",
    "keine information",
    "kann nicht bestimmt werden",
    "nicht bestimmbar",
    "keine angabe",
    "k.a.",
)
_TOKEN_RE = re.compile(r"[a-zA-ZäöüßÄÖÜ0-9]+", re.UNICODE)
_RELEVANCY_STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "of", "on", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
    # German
    "und", "oder", "der", "die", "das", "dem", "den", "des", "ein", "eine",
    "einen", "einem", "eines", "ist", "sind", "war", "wurden", "wird",
    "werden", "hat", "haben", "hatte", "hatten", "wird", "wurde", "auch",
    "als", "auf", "mit", "für", "von", "bei", "aus", "nach", "zu", "in",
    "im", "an", "am", "es", "er", "sie", "wir", "ihr", "wie", "was",
    "wenn", "ob", "da", "hier", "so", "nicht", "noch", "aber", "nur",
    "kann", "muss", "soll", "über",
}

def resolve_ground_truth_answer(row: dict[str, Any], qa_by_id: dict[str, dict[str, Any]]) -> str:
    qid = str(row.get("question_id", ""))
    qa = qa_by_id.get(qid, {})
    for key in (
        "ground_truth_answer",
        "answer",
        "gold_answer",
        "expected_answer",
        "program_answer",
        "original_answer",
        "referenzantwort",
    ):
        if key in qa and qa[key] is not None:
            return str(qa[key])
    return ""


def compute_answer_metrics(
    generated_answer: str,
    ground_truth_answer: str,
    question: str = "",
    abstention_patterns: Iterable[str] | None = None,
) -> dict[str, Any]:
    generated = generated_answer or ""
    truth = ground_truth_answer or ""
    non_empty = 1.0 if generated.strip() else 0.0
    abstained = 1.0 if is_abstention(generated, abstention_patterns) else 0.0
    gen_norm = _normalized_text(generated)
    truth_norm = _normalized_text(truth)
    if not truth.strip():
        answer_match_status = "no_gold"
    elif gen_norm == truth_norm:
        answer_match_status = "match"
    else:
        answer_match_status = "mismatch"
    return {
        "normalized_generated_answer": gen_norm,
        "normalized_gold_answer": truth_norm,
        "answer_match_status": answer_match_status,
        "non_empty_answer_rate": non_empty,
        "answer_coverage_rate": non_empty,  # backward-compatible alias; canonical name is non_empty_answer_rate
        "abstention_rate": abstained,
        "answer_relevancy_score": answer_relevancy_score(question, generated),
    }


class BertScoreScorer:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.torch = torch
        self.device = resolved_device
        self.model_name = model_name
        self.tokenizer_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def score(self, generated_answer: str, ground_truth_answer: str) -> dict[str, float]:
        if not (generated_answer or "").strip() or not (ground_truth_answer or "").strip():
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        generated_embeddings = self._token_embeddings(generated_answer)
        reference_embeddings = self._token_embeddings(ground_truth_answer)
        if generated_embeddings is None or reference_embeddings is None:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

        similarity = generated_embeddings @ reference_embeddings.T
        precision = float(similarity.max(dim=1).values.mean().item())
        recall = float(similarity.max(dim=0).values.mean().item())
        f1 = 0.0 if precision + recall == 0.0 else (2 * precision * recall) / (precision + recall)
        return {
            "bertscore_precision": precision,
            "bertscore_recall": recall,
            "bertscore_f1": f1,
        }

    def _token_embeddings(self, text: str):
        torch = self.torch
        encoded = self.tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state[0]
        input_ids = encoded["input_ids"][0].tolist()
        attention = encoded["attention_mask"][0].bool()
        special = torch.tensor(
            self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True),
            device=self.device,
            dtype=torch.bool,
        )
        mask = attention & ~special
        if int(mask.sum().item()) == 0:
            mask = attention
        token_embeddings = hidden[mask]
        if token_embeddings.numel() == 0:
            return None
        return torch.nn.functional.normalize(token_embeddings, p=2, dim=1)


@lru_cache(maxsize=4)
def build_bert_score_scorer(model_name: str, device: str, max_length: int) -> BertScoreScorer:
    return BertScoreScorer(model_name=model_name, device=device, max_length=max_length)


def compute_bert_score(
    generated_answer: str,
    ground_truth_answer: str,
    scorer: BertScoreScorer,
) -> dict[str, float]:
    return scorer.score(generated_answer, ground_truth_answer)


def bert_score_model_metadata(scorer: BertScoreScorer | None, configured_model_name: str) -> dict[str, str]:
    return {
        "provider": "transformers",
        "model_name": str(getattr(scorer, "model_name", configured_model_name)),
        "tokenizer_name": str(getattr(scorer, "tokenizer_name", configured_model_name)),
        "model_revision": "unknown",
        "local_cache_path": "unknown",
        "device_used": str(getattr(scorer, "device", "unknown")),
        "transformers_version": _package_version("transformers"),
        "torch_version": _package_version("torch"),
    }


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def is_abstention(text: str, patterns: Iterable[str] | None = None) -> bool:
    normalized = _normalized_text(text)
    if not normalized:
        return True
    canonical = normalized.strip(" .!?:;")
    configured = tuple(patterns or DEFAULT_ABSTENTION_PATTERNS)
    return any(canonical == _normalized_text(pattern).strip(" .!?:;") for pattern in configured)


def answer_relevancy_score(question: str, generated_answer: str) -> float:
    """Deterministic lexical-overlap baseline, not a semantic correctness metric."""
    question_tokens = _content_tokens(question)
    answer_tokens = _content_tokens(generated_answer)
    if not question_tokens or not answer_tokens:
        return 0.0
    return len(question_tokens & answer_tokens) / len(answer_tokens)


def _normalized_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _content_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").lower()) if token not in _RELEVANCY_STOPWORDS}


