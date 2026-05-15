from src.pipeline2.aggregation.summarizer import build_leaderboard, summarize_by_experiment


def test_summary_aggregates_final_metrics_and_success_rates():
    rows = [
        {
            "experiment_id": "exp",
            "hit_at_3": 1.0,
            "recall_at_3": 0.5,
            "context_precision_at_3": 0.5,
            "mrr_at_3": 1.0,
            "ndcg_at_3": 0.75,
            "duplicate_context_rate": 0.0,
            "raw_duplicate_rate": 0.25,
            "numeric_accuracy": 1.0,
            "exact_match": 1.0,
            "relative_error": 0.1,
            "numeric_parse_success": 1.0,
            "non_empty_answer_rate": 1.0,
            "answer_coverage_rate": 1.0,
            "abstention_rate": 0.0,
            "answer_relevancy_score": 0.5,
            "retrieval_time_ms": 40.0,
            "generation_time_ms": 60.0,
            "total_latency_ms": 100.0,
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
            "estimated_cost": 0.01,
            "pipeline1_error": None,
            "evaluation_errors": [],
        },
        {
            "experiment_id": "exp",
            "hit_at_3": 0.0,
            "recall_at_3": 0.0,
            "context_precision_at_3": 0.0,
            "mrr_at_3": 0.0,
            "ndcg_at_3": 0.0,
            "duplicate_context_rate": 0.0,
            "raw_duplicate_rate": 0.0,
            "numeric_accuracy": 0.0,
            "exact_match": 0.0,
            "relative_error": 1.0,
            "numeric_parse_success": 0.0,
            "non_empty_answer_rate": 0.0,
            "answer_coverage_rate": 0.0,
            "abstention_rate": 1.0,
            "answer_relevancy_score": 0.0,
            "total_latency_ms": 0.0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "pipeline1_error": "generation failed",
            "evaluation_errors": ["bad retrieved ids"],
        },
    ]

    summary = summarize_by_experiment(rows)[0]

    assert summary["n_questions"] == 2
    assert summary["pipeline_success_rate"] == 0.5
    assert summary["eval_success_rate"] == 0.5
    assert summary["mean_hit_at_3"] == 1.0
    assert summary["mean_recall_at_3"] == 0.5
    assert summary["mean_context_precision_at_3"] == 0.5
    assert summary["mean_mrr_at_3"] == 1.0
    assert summary["mean_ndcg_at_3"] == 0.75
    assert summary["mean_duplicate_context_rate"] == 0.0
    assert summary["mean_raw_duplicate_rate"] == 0.25
    assert summary["mean_numeric_accuracy"] == 1.0
    assert summary["mean_exact_match"] == 1.0
    assert summary["mean_relative_error"] == 0.1
    assert summary["median_relative_error"] == 0.1
    assert summary["numeric_parse_success_rate"] == 1.0
    assert summary["mean_non_empty_answer_rate"] == 1.0
    assert summary["mean_answer_coverage_rate"] == 1.0
    assert summary["mean_abstention_rate"] == 0.0
    assert summary["mean_answer_relevancy"] == 0.5
    assert summary["mean_retrieval_time_ms"] == 40.0
    assert summary["mean_generation_time_ms"] == 60.0
    assert summary["mean_total_latency_ms"] == 100.0
    assert summary["mean_input_tokens"] == 7.0
    assert summary["mean_output_tokens"] == 3.0
    assert summary["mean_total_tokens"] == 10.0
    assert summary["mean_estimated_cost"] == 0.01


def test_leaderboard_ranks_experiments_by_configured_metric_descending():
    summary = [
        {"experiment_id": "exp_a", "mean_recall_at_5": 0.4, "mean_numeric_accuracy": 0.9},
        {"experiment_id": "exp_b", "mean_recall_at_5": 0.7, "mean_numeric_accuracy": 0.5},
        {"experiment_id": "exp_c", "mean_recall_at_5": 0.6, "mean_numeric_accuracy": 0.8},
    ]

    leaderboard = build_leaderboard(summary, "mean_recall_at_5", sort_ascending=False)

    assert [row["rank"] for row in leaderboard] == [1, 2, 3]
    assert [row["experiment_id"] for row in leaderboard] == ["exp_b", "exp_c", "exp_a"]
    assert all(row["sort_metric"] == "mean_recall_at_5" for row in leaderboard)


def test_leaderboard_sorting_supports_ascending_latency():
    summary = [
        {"experiment_id": "slow", "mean_total_latency_ms": 120.0},
        {"experiment_id": "fast", "mean_total_latency_ms": 20.0},
        {"experiment_id": "missing", "mean_recall_at_5": 0.9},
    ]

    leaderboard = build_leaderboard(summary, "mean_total_latency_ms", sort_ascending=True)

    assert [row["experiment_id"] for row in leaderboard] == ["fast", "slow", "missing"]


def test_leaderboard_supports_new_metric_columns_without_special_cases():
    summary = [
        {"experiment_id": "exp_a", "mean_ndcg_at_5": 0.4},
        {"experiment_id": "exp_b", "mean_ndcg_at_5": 0.8},
    ]

    leaderboard = build_leaderboard(summary, "mean_ndcg_at_5", sort_ascending=False)

    assert [row["experiment_id"] for row in leaderboard] == ["exp_b", "exp_a"]
