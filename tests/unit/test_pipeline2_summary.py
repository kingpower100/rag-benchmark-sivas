from src.pipeline2.aggregation.summarizer import summarize_by_experiment


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
            "exact_match": 1.0,
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
            "exact_match": 0.0,
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
    assert summary["mean_hit_at_3"] == 0.5
    assert summary["mean_recall_at_3"] == 0.25
    assert summary["mean_context_precision_at_3"] == 0.25
    assert summary["mean_mrr_at_3"] == 0.5
    assert summary["mean_ndcg_at_3"] == 0.375
    assert summary["mean_duplicate_context_rate"] == 0.0
    assert summary["mean_raw_duplicate_rate"] == 0.125
    assert summary["mean_exact_match"] == 0.5
    assert summary["mean_non_empty_answer_rate"] == 0.5
    assert summary["mean_answer_coverage_rate"] == 0.5
    assert summary["mean_abstention_rate"] == 0.5
    assert summary["diagnostic_mean_answer_relevancy"] == 0.25
    assert summary["mean_retrieval_time_ms"] == 40.0
    assert summary["mean_generation_time_ms"] == 60.0
    assert summary["mean_total_latency_ms"] == 50.0
    assert summary["mean_input_tokens"] == 7.0
    assert summary["mean_output_tokens"] == 3.0
    assert summary["mean_total_tokens"] == 5.0
    assert summary["mean_estimated_cost"] == 0.005


def test_summary_keeps_pipeline1_failures_in_metric_denominators():
    rows = [
        {
            "experiment_id": "exp",
            "recall_at_5": 1.0,
            "mrr_at_5": 1.0,
            "exact_match": 1.0,
            "pipeline1_error": None,
            "evaluation_errors": [],
        },
        {
            "experiment_id": "exp",
            "recall_at_5": 1.0,
            "mrr_at_5": 1.0,
            "exact_match": 1.0,
            "pipeline1_error": None,
            "evaluation_errors": [],
        },
        {
            "experiment_id": "exp",
            "recall_at_5": 0.0,
            "mrr_at_5": 0.0,
            "exact_match": 0.0,
            "pipeline1_error": "generation failed",
            "evaluation_errors": [],
        },
        {
            "experiment_id": "exp",
            "recall_at_5": 0.0,
            "mrr_at_5": 0.0,
            "exact_match": 0.0,
            "pipeline1_error": "timeout",
            "evaluation_errors": [],
        },
    ]

    summary = summarize_by_experiment(rows)[0]

    assert summary["n_questions"] == 4
    assert summary["mean_exact_match"] == 0.5
    assert summary["mean_recall_at_5"] == 0.5
    assert summary["mean_mrr_at_5"] == 0.5
    assert summary["pipeline_success_rate"] == 0.5


def test_summary_tracks_unknown_count_and_rate():
    rows = [
        {"experiment_id": "exp", "is_unknown": 1.0, "evaluation_errors": [], "pipeline1_error": None},
        {"experiment_id": "exp", "is_unknown": 1.0, "evaluation_errors": [], "pipeline1_error": None},
        {"experiment_id": "exp", "is_unknown": 0.0, "evaluation_errors": [], "pipeline1_error": None},
        {"experiment_id": "exp", "is_unknown": 0.0, "evaluation_errors": [], "pipeline1_error": None},
    ]

    summary = summarize_by_experiment(rows)[0]

    assert summary["unknown_count"] == 2
    assert summary["unknown_rate"] == 0.5
