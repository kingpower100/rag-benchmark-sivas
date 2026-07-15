from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.pipeline4.schemas import Pipeline4Config, RetrievalScoreWeights, RQIWeights


class TestRetrievalScoreWeights:
    def test_default_weights_sum_to_one(self):
        w = RetrievalScoreWeights()
        total = w.recall_at_5 + w.mrr_at_5 + w.ndcg_at_5 + w.context_precision_at_5
        assert abs(total - 1.0) < 1e-9

    def test_invalid_weights_raise(self):
        with pytest.raises(Exception):
            RetrievalScoreWeights(recall_at_5=0.4, mrr_at_5=0.4, ndcg_at_5=0.4, context_precision_at_5=0.4)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            RetrievalScoreWeights(recall_at_5=0.35, mrr_at_5=0.25, ndcg_at_5=0.20, context_precision_at_5=0.20, extra=0.0)


class TestRQIWeights:
    def test_default_weights_sum_to_one(self):
        w = RQIWeights()
        total = w.correctness + w.faithfulness + w.context_relevance + w.recall_at_5 + w.no_unknown
        assert abs(total - 1.0) < 1e-9

    def test_invalid_weights_raise(self):
        with pytest.raises(Exception):
            RQIWeights(correctness=0.3, faithfulness=0.3, context_relevance=0.3, recall_at_5=0.3, no_unknown=0.3)


class TestPipeline4Config:
    def test_default_config_valid(self):
        cfg = Pipeline4Config()
        assert cfg.ranking_mode == "retrieval_only"

    def test_invalid_ranking_mode(self):
        with pytest.raises(Exception):
            Pipeline4Config(ranking_mode="invalid_mode")

    def test_from_yaml(self):
        data = {
            "pipeline2_runs_dir": "data/eval/runs/pipeline2",
            "pipeline3_runs_dir": "data/eval/runs/pipeline3",
            "output_dir": "data/eval/runs/pipeline4",
            "run_id": "test_run",
            "ranking_mode": "retrieval_only",
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "test.yaml"
            cfg_path.write_text(yaml.dump(data), encoding="utf-8")
            cfg = Pipeline4Config.from_yaml(str(cfg_path))
        assert cfg.run_id == "test_run"
        assert cfg.ranking_mode == "retrieval_only"

    def test_from_yaml_with_extends(self):
        base_data = {
            "ranking_mode": "retrieval_only",
            "run_id": "base_run",
        }
        override_data = {
            "extends": "base.yaml",
            "run_id": "override_run",
        }
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "base.yaml"
            base_path.write_text(yaml.dump(base_data), encoding="utf-8")
            override_path = Path(tmp) / "experiment.yaml"
            override_path.write_text(yaml.dump(override_data), encoding="utf-8")
            cfg = Pipeline4Config.from_yaml(str(override_path))
        assert cfg.run_id == "override_run"
        assert cfg.ranking_mode == "retrieval_only"

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            Pipeline4Config(unknown_field="value")
