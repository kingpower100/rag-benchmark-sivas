from __future__ import annotations

from collections import defaultdict
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


def compute_category_routing_report(
    per_question: list[dict[str, Any]],
    sivas_categories: list[str],
) -> dict[str, Any]:
    """Compute aggregate category routing metrics.

    If no predictions are present, routing is flagged as inactive and no
    metrics are computed (avoids null metrics in the report).
    """
    total = len(per_question)
    rows_with_prediction = [
        row for row in per_question
        if row.get("category_predicted") is not None
    ]
    active_count = len(rows_with_prediction)

    if active_count == 0:
        return {
            "category_routing_active": False,
            "category_routing_coverage": 0.0,
            "total_questions": total,
            "questions_with_prediction": 0,
            "message": (
                "Category routing produced no predictions. "
                "All category metrics are omitted to avoid null metrics."
            ),
        }

    # Confusion matrix: gold_category → predicted_category → count
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows_with_prediction:
        gold = row.get("category_gold") or "unknown"
        pred = row.get("category_predicted") or "unknown"
        confusion[gold][pred] += 1

    all_classes: set[str] = set(sivas_categories)
    for row in rows_with_prediction:
        if row.get("category_gold"):
            all_classes.add(row["category_gold"])
        if row.get("category_predicted"):
            all_classes.add(row["category_predicted"])

    per_class: dict[str, dict[str, Any]] = {}
    for cls in sorted(all_classes):
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in all_classes if other != cls)
        fn = sum(confusion[cls][other] for other in all_classes if other != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_precision = sum(v["precision"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(per_class) if per_class else 0.0

    accuracy_rows = [row for row in rows_with_prediction if row.get("category_correct") is not None]
    category_accuracy = (
        sum(row["category_correct"] for row in accuracy_rows) / len(accuracy_rows)
        if accuracy_rows else 0.0
    )

    return {
        "category_routing_active": True,
        "category_routing_coverage": round(active_count / total, 6) if total > 0 else 0.0,
        "total_questions": total,
        "questions_with_prediction": active_count,
        "category_accuracy": round(category_accuracy, 6),
        "category_precision_macro": round(macro_precision, 6),
        "category_recall_macro": round(macro_recall, 6),
        "per_class_metrics": per_class,
        "confusion_matrix": {gold: dict(preds) for gold, preds in confusion.items()},
    }
