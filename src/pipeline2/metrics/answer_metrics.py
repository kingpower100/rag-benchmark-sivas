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
        "question_answer_lexical_f1": question_answer_lexical_f1(question, generated),
    }


class OfficialBertScorer:
    """BERTScore using the official bert-score library (Zhang et al., 2020).

    Wraps ``bert_score.BERTScorer``, which automatically selects the optimal
    transformer layer for the chosen model and supports optional IDF weighting
    and baseline rescaling.  Results are comparable to published BERTScore
    benchmarks when ``rescale_with_baseline=True``.

    Output keys are ``official_bertscore_precision/recall/f1``.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        idf: bool = False,
        rescale_with_baseline: bool = False,
    ) -> None:
        try:
            from bert_score import BERTScorer
        except ImportError:
            raise ImportError(
                "The 'bert-score' package is required for BERTScore computation. "
                "Install with: pip install bert-score>=0.3.0"
            ) from None

        self.model_name = model_name
        self.idf = idf
        self.rescale_with_baseline = rescale_with_baseline
        resolved_device = None if device == "auto" else device
        self._scorer = BERTScorer(
            model_type=model_name,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            device=resolved_device,
        )
        self.device = str(getattr(self._scorer, "device", device))

    def score(self, generated_answer: str, ground_truth_answer: str) -> dict[str, float]:
        _zero = {
            "official_bertscore_precision": 0.0,
            "official_bertscore_recall": 0.0,
            "official_bertscore_f1": 0.0,
        }
        if not (generated_answer or "").strip() or not (ground_truth_answer or "").strip():
            return _zero
        P, R, F1 = self._scorer.score([generated_answer], [ground_truth_answer], verbose=False)
        return {
            "official_bertscore_precision": float(P[0]),
            "official_bertscore_recall": float(R[0]),
            "official_bertscore_f1": float(F1[0]),
        }


@lru_cache(maxsize=4)
def build_bert_score_scorer(
    model_name: str,
    device: str,
    idf: bool,
    rescale_with_baseline: bool,
) -> OfficialBertScorer:
    return OfficialBertScorer(
        model_name=model_name,
        device=device,
        idf=idf,
        rescale_with_baseline=rescale_with_baseline,
    )


def compute_bert_score(
    generated_answer: str,
    ground_truth_answer: str,
    scorer: "OfficialBertScorer",
) -> dict[str, float]:
    return scorer.score(generated_answer, ground_truth_answer)


def bert_score_model_metadata(
    scorer: "OfficialBertScorer | None",
    configured_model_name: str,
    idf: bool = False,
    rescale_with_baseline: bool = False,
) -> dict[str, Any]:
    return {
        "implementation": "official_bert_score_library",
        "library_version": _package_version("bert-score"),
        "model_name": str(getattr(scorer, "model_name", configured_model_name)),
        "idf": idf,
        "rescale_with_baseline": rescale_with_baseline,
        "device_used": str(getattr(scorer, "device", "unknown")) if scorer is not None else "unknown",
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


def question_answer_lexical_f1(question: str, generated_answer: str) -> float:
    """Token-overlap F1 between question and answer content tokens.

    Computes 2·|Q∩A| / (|Q| + |A|) where Q and A are sets of content tokens
    (stopwords removed, regex-extracted).  The symmetric F1 formulation means
    both coverage of question vocabulary (recall) and on-topic phrasing
    (precision) are penalised, unlike the deprecated ``|Q∩A|/|A|`` formula
    which rewarded pure repetition of question terms.

    Limitation: pairs with zero lexical overlap always score 0.0 regardless of
    semantic correctness.  An embedding-based scorer is required to distinguish
    factually correct non-overlapping answers from genuinely irrelevant ones.

    This is a diagnostic-only metric; it is not a semantic correctness or
    factual accuracy measure.
    """
    question_tokens = _content_tokens(question)
    answer_tokens = _content_tokens(generated_answer)
    if not question_tokens or not answer_tokens:
        return 0.0
    intersection = len(question_tokens & answer_tokens)
    return 2 * intersection / (len(question_tokens) + len(answer_tokens))


def _normalized_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _content_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").lower()) if token not in _RELEVANCY_STOPWORDS}


