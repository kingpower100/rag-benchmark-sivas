
# SIVAS ERP RAG Benchmark

This repository contains the SIVAS ERP Benchmark Dataset v1.0 workflow:

- Pipeline 1: baseline RAG execution.
- Pipeline 2: offline evaluation of Pipeline 1 outputs.

## Active Dataset

The active raw files are:

- `data/raw/kb_documents_fixed.jsonl`
- `data/raw/questions_fixed.jsonl`
- `data/raw/qa_ground_truth_fixed.jsonl`

Pipeline 1 uses only `kb_documents_fixed.jsonl` and `questions_fixed.jsonl`. Pipeline 2 uses `qa_ground_truth_fixed.jsonl` for offline evaluation.

## Active Pipeline 1 Config

- `configs/pipeline1/experiments/11_sivas_fixed512_faiss_dense_mistralsmall_baseline.yaml`

Baseline run command:

```bash
python -m src.pipeline1.main --config configs/pipeline1/experiments/11_sivas_fixed512_faiss_dense_mistralsmall_baseline.yaml
```

Expected Pipeline 1 output:

- `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_mistralsmall_baseline/results.jsonl`
- `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_mistralsmall_baseline/run_manifest.json`
- `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_mistralsmall_baseline/logs.txt`

Pipeline 1 requires the configured local generation service and embedding dependencies at runtime. Cleanup and static checks must not start services, load models, build indexes, or execute retrieval/generation.

## Pipeline 2 Evaluation

Pipeline 2 defaults are SIVAS-first:

- `configs/pipeline2/base_eval.yaml`
- `qa_path: data/raw/qa_ground_truth_fixed.jsonl`
- `pipeline1_results_path: data/runs/pipeline1/11_sivas_fixed512_faiss_dense_mistralsmall_baseline/results.jsonl`

Run evaluation only after Pipeline 1 has produced `results.jsonl`:

```bash
python -m src.pipeline2.main --config configs/pipeline2/base_eval.yaml
```

Expected Pipeline 2 output:

- `data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_mistralsmall_baseline_eval/per_question.jsonl`
- `data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_mistralsmall_baseline_eval/summary_by_experiment.csv`
- `data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_mistralsmall_baseline_eval/eval_manifest.json`

## Docker Services (pgvector and Elasticsearch backends)

The pgvector and Elasticsearch retrieval backends require two Docker services.
The FAISS baseline needs no external services.

### Start services

```bash
cd infra/docker
docker compose up -d postgres elasticsearch
docker compose ps
```

### Verify health

```bash
# PostgreSQL
docker exec rag-benchmark-postgres pg_isready -U rag -d rag
docker exec rag-benchmark-postgres psql -U rag -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker exec rag-benchmark-postgres psql -U rag -d rag -c "\dx"

# Elasticsearch
curl -s http://localhost:9200/_cluster/health | python3 -m json.tool
```

### Environment variables

```bash
export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
export ELASTICSEARCH_URL="http://localhost:9200"
```

### Quick service check (no data modification)

```bash
python scripts/check_backend_services.py
```

### Initialize schema and build indexes

```bash
# Schema/index creation (idempotent):
python scripts/init_pgvector.py
python scripts/init_elasticsearch.py --host http://localhost:9200 --index rag_benchmark_chunks

# Index documents (requires a pgvector or ES YAML config):
python scripts/index_pgvector.py      --config configs/pipeline1/experiments/<pgvector_config>.yaml
python scripts/index_elasticsearch.py --config configs/pipeline1/experiments/<es_bm25_config>.yaml
```

See `infra/docker/README.md` for full details, teardown instructions, and troubleshooting.

## Data Format

`kb_documents_fixed.jsonl` requires:

- `doc_key` or `doc_id`
- `text`
- `kategorie`
- `wissensart`
- `titel`
- `quellpfad`
- `doc_name`

`questions_fixed.jsonl` requires:

- `question_id`
- `frage`

`qa_ground_truth_fixed.jsonl` is evaluation-only and must not be used by Pipeline 1.

**********
cd ~/SIVAS/rag-benchmark-sivas
source .venv/bin/activate

export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
export ELASTICSEARCH_URL="http://localhost:9201"
echo $PGVECTOR_DSN
echo $ELASTICSEARCH_URL