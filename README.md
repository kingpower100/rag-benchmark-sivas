# RAG Benchmark

Config-driven RAG benchmark with two stages:

- Pipeline 1 runs retrieval and generation, then writes per-question outputs and a run manifest.
- Pipeline 2 evaluates Pipeline 1 outputs with deterministic automatic metrics.

## Prerequisites

- Python 3.11+
- Ollama running with the configured generation model, currently `qwen2.5:7b`

```bash
ollama serve
ollama pull qwen2.5:7b
pip install -e .
```

`OLLAMA_BASE_URL` can override `generation.base_url` for local, remote, or container runs.

## Current Configs

Pipeline 1 experiments:

- `configs/pipeline1/experiments/01_fixed512_faiss_dense_norerank_qwen25.yaml`
- `configs/pipeline1/experiments/02_fixed512_faiss_dense_rerank5_qwen25.yaml`
- `configs/pipeline1/experiments/03_fixed512_faiss_dense_rerank10_qwen25.yaml`
- `configs/pipeline1/experiments/04_fixed512_faiss_dense_rerank10_metaboost_qwen25.yaml`
- `configs/pipeline1/experiments/05_tableaware512_faiss_dense_rerank3_qwen25.yaml`
- `configs/pipeline1/experiments/06_fixed512_es_dense_knn_qwen25.yaml`
- `configs/pipeline1/experiments/07_fixed512_es_dense_script_score_qwen25.yaml`
- `configs/pipeline1/experiments/08_fixed512_es_hybrid_rrf_qwen25.yaml`
- `configs/pipeline1/experiments/09_fixed512_es_hybrid_rrf_dateaware_qwen25.yaml`
- `configs/pipeline1/experiments/10_fixed512_es_hybrid_rrf_dateaware_llama3.yaml`

Primary Pipeline 2 evaluation:

- `configs/pipeline2/experiments/01_eval_fixed512_faiss_dense_norerank_qwen25.yaml`

## Run Pipeline 1

```bash
python -m src.pipeline1.main --config configs/pipeline1/experiments/01_fixed512_faiss_dense_norerank_qwen25.yaml
```

Outputs are written under:

- `data/runs/pipeline1/<experiment_id>/results.jsonl`
- `data/runs/pipeline1/<experiment_id>/results.csv`
- `data/runs/pipeline1/<experiment_id>/run_manifest.json`
- `data/runs/pipeline1/<experiment_id>/logs.txt`

Pipeline 1 preflight checks:

- raw dataset paths exist and are non-empty
- `retrieval.fetch_k >= retrieval.top_k`
- `chunking.chunk_overlap < chunking.chunk_size`
- Ollama is reachable at `${OLLAMA_BASE_URL:-generation.base_url}/api/tags`

For tests only, Ollama preflight can be skipped with:

```bash
export PIPELINE1_SKIP_OLLAMA_PREFLIGHT=1
```

## Run Pipeline 2

Run evaluation after Pipeline 1 has produced `results.jsonl`:

```bash
python -m src.pipeline2.main --config configs/pipeline2/experiments/01_eval_fixed512_faiss_dense_norerank_qwen25.yaml
```

Outputs are written under:

- `data/eval/runs/pipeline2/<eval_run_id>/per_question.jsonl`
- `data/eval/runs/pipeline2/<eval_run_id>/per_question.csv`
- `data/eval/runs/pipeline2/<eval_run_id>/summary_by_experiment.csv`
- `data/eval/runs/pipeline2/<eval_run_id>/eval_manifest.json`

Pipeline 2 automatic metrics:

- retrieval: `hit_at_k`, `recall_at_k`, `context_precision_at_k`, `mrr_at_k`, `ndcg_at_k`, `raw_duplicate_rate`
- answer quality: `literal_exact_match`, `canonical_exact_match`, `strict_numeric_accuracy`, `tolerant_numeric_accuracy`, `relative_error`, `numeric_parse_success`, `non_empty_answer_rate`, `abstention_rate`, `answer_relevancy_score`
- efficiency: `total_latency_ms`, `total_tokens`, `estimated_cost`
- reliability: `pipeline_success_rate`, `eval_success_rate`, `generation_failure_rate`, `run_valid`

Retrieval evaluation is source-file-level for the OfficeQA/Treasury configs and uses the explicit `evaluation.retrieval_eval_field` setting; it does not choose the retrieved ID field by checking gold overlap. `literal_exact_match` only trims, lowercases, and collapses whitespace. `canonical_exact_match` may canonicalize numeric/yes-no forms. `strict_numeric_accuracy` requires exact numeric equality after scale normalization, while `tolerant_numeric_accuracy` keeps the tolerance-based diagnostic. `numeric_accuracy` is kept as a backward-compatible alias for strict numeric correctness.

Citation correctness is structural only: citations are checked as retrieved source metadata, not as verified support for each answer sentence. `hallucination_rate` is emitted as null unless a detector is explicitly implemented. `answer_relevancy_score` is a deterministic lexical-overlap baseline between question and answer content words. It is useful as a cheap diagnostic only; it is not a semantic correctness metric. RAGAS support is preserved as optional code, but it is not part of the main automatic benchmark metric set.

## Helper Commands

```bash
python scripts/list_configs.py
python scripts/run_config.py configs/pipeline1/experiments/01_fixed512_faiss_dense_norerank_qwen25.yaml
python scripts/compare_runs.py
python scripts/benchmark_pipeline1.py --config configs/pipeline1/experiments/01_fixed512_faiss_dense_norerank_qwen25.yaml
python scripts/test_ollama.py --base-url http://localhost:11434
```

## Preserved Docker Support

Docker files are kept under `infra/docker/` for future server or GB10-style runs:

- `infra/docker/Dockerfile`
- `infra/docker/docker-compose.yml`
- `infra/docker/.dockerignore`
- `infra/docker/docker_host_ollama.yaml`

Build and run from the repository root:

```bash
docker compose -f infra/docker/docker-compose.yml build
docker compose -f infra/docker/docker-compose.yml up --abort-on-container-exit
```

The compose file mounts:

- `data/raw` as read-only input
- `data/runs` for Pipeline 1 outputs
- `data/processed` for reusable chunk, embedding, and index caches

It sets `OLLAMA_BASE_URL=http://host.docker.internal:11434` so the container can call Ollama on the host.

## Data Format

`data/raw/treasury_bulletins_1939_1963.jsonl` is the active Pipeline 1 knowledge base for the bundled experiments. It requires:

- `document_id` or `id`
- `cleaned_context`

`data/raw/questions_only.jsonl` is the active Pipeline 1 question file. It requires:

- `uid`
- `question`

`data/raw/qa_test.jsonl` and `data/raw/ground_truth_contexts.jsonl` are used only by Pipeline 2 evaluation.

## Cleanup Before Full Runs

Generated artifacts can be removed before a clean benchmark:

- `__pycache__/`
- `.pytest_cache/`
- `.tmp_pytest*/`
- `*.egg-info/`
- `data/processed/*`
- `data/runs/pipeline1/*`
- `data/eval/runs/pipeline2/*`

Keep:

- external copies of the required `data/raw/*.jsonl` files
- `configs/*`
- `src/*`
- `tests/*`
- `infra/docker/*`
- `.venv` if it is your active environment

## Tests

```bash
python -m pytest -p no:cacheprovider --basetemp=.tmp_pytest_clean tests/unit tests/integration
```
