from __future__ import annotations

from typing import Any


def compute_category_metrics(
    detected_category: str | None,
    gold_kategorie: str | None,
) -> dict[str, Any]:
    """Compare the orchestration LLM's predicted category against the gold label.

    Returns category_correct=None when either value is missing so that the
    metric is excluded from averages rather than penalising missing predictions.
    """
    predicted = detected_category.strip() if detected_category else None
    gold = gold_kategorie.strip() if gold_kategorie else None
    if predicted is None or gold is None:
        return {
            "category_correct": None,
            "category_predicted": predicted,
            "category_gold": gold,
        }
    correct = predicted.lower() == gold.lower()
    return {
        "category_correct": 1.0 if correct else 0.0,
        "category_predicted": predicted,
        "category_gold": gold,
    }
