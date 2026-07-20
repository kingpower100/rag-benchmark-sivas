from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RankedEntry:
    experiment_id: str
    score: float
    rank: int
    group_id: str
    extra: dict = field(default_factory=dict)


def competition_rank(entries: list[tuple[str, float]]) -> list[tuple[str, float, int]]:
    """Assign competition ranks (1, 2, 2, 4) to score pairs."""
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda x: (-x[1], x[0]))
    result: list[tuple[str, float, int]] = []
    rank = 1
    i = 0
    while i < len(sorted_entries):
        current_score = sorted_entries[i][1]
        group: list[tuple[str, float]] = []
        j = i
        while j < len(sorted_entries) and sorted_entries[j][1] == current_score:
            group.append(sorted_entries[j])
            j += 1
        for exp_id, score in group:
            result.append((exp_id, score, rank))
        rank += len(group)
        i = j
    return result


def rank_retrieval(
    experiment_scores: dict[str, float],
    comparison_groups: list,
) -> dict[str, int]:
    """Return experiment_id to retrieval rank, computed per comparison group."""
    ranks: dict[str, int] = {}
    for group in comparison_groups:
        group_entries = [
            (eid, experiment_scores[eid])
            for eid in group.experiment_ids
            if eid in experiment_scores
        ]
        for exp_id, _score, rank in competition_rank(group_entries):
            ranks[exp_id] = rank
    return ranks


def rank_rqi(
    experiment_rqi: dict[str, Optional[float]],
    comparison_groups: list,
    ranking_mode: str,
) -> dict[str, Optional[int]]:
    """Return experiment_id to RQI rank, leaving ineligible entries unranked."""
    ranks: dict[str, Optional[int]] = {}
    for group in comparison_groups:
        group_entries = [
            (eid, experiment_rqi[eid])
            for eid in group.experiment_ids
            if eid in experiment_rqi and experiment_rqi[eid] is not None
        ]
        for exp_id, _score, rank in competition_rank(group_entries):
            ranks[exp_id] = rank
        for eid in group.experiment_ids:
            if eid not in ranks:
                ranks[eid] = None
    return ranks
