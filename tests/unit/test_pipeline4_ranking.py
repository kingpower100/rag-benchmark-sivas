from __future__ import annotations

import pytest

from src.pipeline4.ranking import competition_rank, rank_retrieval
from src.pipeline4.validation import ComparisonGroup


class TestCompetitionRank:
    def test_empty_returns_empty(self):
        assert competition_rank([]) == []

    def test_single_entry_rank_one(self):
        result = competition_rank([("a", 0.5)])
        assert result == [("a", 0.5, 1)]

    def test_distinct_scores_sequential_ranks(self):
        entries = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        result = competition_rank(entries)
        by_exp = {e[0]: e[2] for e in result}
        assert by_exp["a"] == 1
        assert by_exp["b"] == 2
        assert by_exp["c"] == 3

    def test_tied_scores_same_rank(self):
        entries = [("a", 0.8), ("b", 0.8), ("c", 0.5)]
        result = competition_rank(entries)
        by_exp = {e[0]: e[2] for e in result}
        assert by_exp["a"] == 1
        assert by_exp["b"] == 1
        assert by_exp["c"] == 3

    def test_competition_skip_after_tie(self):
        entries = [("a", 0.8), ("b", 0.8), ("c", 0.8), ("d", 0.5)]
        result = competition_rank(entries)
        by_exp = {e[0]: e[2] for e in result}
        assert by_exp["a"] == 1
        assert by_exp["b"] == 1
        assert by_exp["c"] == 1
        assert by_exp["d"] == 4

    def test_tie_breaking_by_experiment_id(self):
        entries = [("b", 0.8), ("a", 0.8)]
        result = competition_rank(entries)
        ranked = [(e[0], e[2]) for e in sorted(result, key=lambda x: x[0])]
        # Both have rank 1 due to tie; order in result is alphabetical
        assert all(r == 1 for _, r in ranked)
        # Verify "a" comes before "b" in the sorted output
        order = [e[0] for e in result]
        assert order.index("a") < order.index("b")

    def test_unordered_input_correctly_ranked(self):
        entries = [("c", 0.3), ("a", 0.9), ("b", 0.6)]
        result = competition_rank(entries)
        by_exp = {e[0]: e[2] for e in result}
        assert by_exp["a"] == 1
        assert by_exp["b"] == 2
        assert by_exp["c"] == 3

    def test_all_tied_all_rank_one(self):
        entries = [("x", 0.5), ("y", 0.5), ("z", 0.5)]
        result = competition_rank(entries)
        ranks = {e[0]: e[2] for e in result}
        assert all(r == 1 for r in ranks.values())

    def test_rank_retrieval_per_group(self):
        group_a = ComparisonGroup(key=("h1", 96), experiment_ids=["a1", "a2"])
        group_b = ComparisonGroup(key=("h2", 96), experiment_ids=["b1"])
        scores = {"a1": 0.8, "a2": 0.6, "b1": 0.95}
        ranks = rank_retrieval(scores, [group_a, group_b])
        # Within group_a: a1 ranks 1, a2 ranks 2
        assert ranks["a1"] == 1
        assert ranks["a2"] == 2
        # b1 is the only one in its group, ranks 1
        assert ranks["b1"] == 1

    def test_rank_retrieval_tied_within_group(self):
        group = ComparisonGroup(key=("h1", 96), experiment_ids=["a", "b", "c"])
        scores = {"a": 0.7, "b": 0.7, "c": 0.5}
        ranks = rank_retrieval(scores, [group])
        assert ranks["a"] == 1
        assert ranks["b"] == 1
        assert ranks["c"] == 3
