from __future__ import annotations

import logging

logger = logging.getLogger("pipeline3.judge")

PROMPT_VERSION = "v2"

COMBINED_JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system operating in the German ERP domain.

Evaluate the generated answer across exactly six dimensions. Respond ONLY with a valid JSON object — no markdown, no code blocks, no text outside the JSON.

Scoring scale: 0 = Very Poor, 5 = Excellent.
Exception: hallucination uses 0 = No hallucination (best), 5 = Severe hallucination (worst).
Note: the hallucination score is reversed when computing the overall score — a lower hallucination score contributes more positively to the final result.

Evaluate each metric independently. The score assigned to one metric must not influence any other metric. For example, an answer can be correct but incomplete, faithful but incorrect, relevant but hallucinated, or well-supported by context but still missing required details.

--- SCORING RUBRICS ---

correctness — Is the generated answer factually correct compared to the ground truth answer?
  5 = Fully correct; all details match the ground truth exactly.
  3 = Mostly correct; minor inaccuracies or omissions that do not change the core meaning.
  1 = Mostly incorrect; major factual errors relative to the ground truth.
  0 = Entirely wrong or directly contradicts the ground truth.

faithfulness — Is every statement in the generated answer supported by the retrieved context?
  5 = Every statement is directly supported by the retrieved context; nothing is added beyond it.
  3 = Most statements are supported; one or two minor details come from outside the context.
  1 = Many statements are unsupported by the context; significant external claims are present.
  0 = The answer ignores or contradicts the retrieved context throughout.

relevancy — Does the generated answer directly and specifically address the user's question?
  5 = Directly and completely addresses the question with no off-topic content.
  3 = Addresses the question but includes unnecessary information or minor tangents.
  1 = Only tangentially related to the question; does not directly answer it.
  0 = Does not address the question at all.

completeness — Does the generated answer contain all important information present in the ground truth?
  5 = Contains all key information from the ground truth; nothing important is missing.
  3 = Covers the main points but omits some secondary or supporting details.
  1 = Missing most key information present in the ground truth.
  0 = Entirely missing the required content from the ground truth.

hallucination — Does the generated answer contain statements NOT supported by context or ground truth? (0 = none, best; 5 = severe, worst)
  0 = No hallucination; every claim is grounded in the context or ground truth.
  1 = Minimal hallucination; one minor unimportant claim not in context or ground truth.
  3 = Moderate hallucination; several statements are fabricated or unsupported.
  5 = Severe hallucination; the answer is dominated by fabricated statements not found in context or ground truth.

context_relevance — Are the retrieved contexts relevant and useful for answering the user's question?
  5 = All retrieved context is directly relevant and useful for answering the question.
  3 = Some context is relevant; other portions are off-topic or only loosely related.
  1 = Most retrieved context is irrelevant; very little useful information is present.
  0 = Retrieved context is entirely irrelevant to the question.

--- OUTPUT FORMAT ---

Required JSON structure (no other keys allowed):
{{
  "correctness": <integer 0-5>,
  "faithfulness": <integer 0-5>,
  "relevancy": <integer 0-5>,
  "completeness": <integer 0-5>,
  "hallucination": <integer 0-5>,
  "context_relevance": <integer 0-5>,
  "overall_score": <float>,
  "reasoning": "<one sentence summary of the evaluation>"
}}

---

Question:
{question}

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

Generated Answer:
{generated_answer}
"""


def build_combined_judge_prompt(
    question: str,
    ground_truth: str,
    context: str,
    generated_answer: str,
) -> str:
    return COMBINED_JUDGE_PROMPT_TEMPLATE.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        context=context.strip(),
        generated_answer=generated_answer.strip(),
    )


def format_context(
    context_texts: list[str],
    max_chars: int = 6000,
    question_id: str = "",
) -> tuple[str, bool]:
    """Format retrieved context texts for the judge prompt.

    Returns:
        (formatted_text, was_truncated) — was_truncated is True when the
        joined context exceeded max_chars and was cut off.
    """
    if not context_texts:
        return "[No context retrieved]", False
    joined = "\n\n---\n\n".join(t.strip() for t in context_texts if t.strip())
    original_len = len(joined)
    if original_len > max_chars:
        joined = joined[:max_chars] + "\n...[truncated]"
        logger.warning(
            "Context truncated for question_id=%r: original=%d chars, used=%d chars (limit=%d)",
            question_id or "(unknown)",
            original_len,
            max_chars,
            max_chars,
        )
        return joined, True
    return joined, False
