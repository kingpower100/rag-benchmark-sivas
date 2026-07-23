
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

## Benchmark Experiment Groups

Benchmark configs are grouped by pipeline under `configs/pipeline*/experiments/`.

- `91-96`: Prompt Benchmarking, with `91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml` as the baseline and prompt V1-V5 variants through `96_sivas_fixed512_faiss_dense_mistralsmall_prompt_v5.yaml`.
- `97-98`: LLM Benchmarking, holding Prompt V4 fixed and varying the orchestration model with Qwen2.5 and Llama 3.1.

Smoke tests live separately under `configs/pipeline*/smoke/` and are excluded from benchmark leaderboards.

## Dense Retrieval Strategies

Pipeline 1 supports three dense retrieval strategies with distinct routing behavior:

- `dense`: embeds the question and searches the complete configured dense index. It does not use orchestration category predictions for filtering.
- `category_aware_dense`: uses the orchestration LLM's validated category as a hard routing filter. Valid categories search only that category. Invalid or missing categories, and insufficient category results when `fallback_to_global=true`, fall back to global retrieval. It does not run a global validation probe or compute support metrics.
- `adaptive_category_aware_dense`: uses evidence-aware category routing. After orchestration predicts and validates a category against the KB taxonomy, the retriever runs a small global probe across all categories. The probe is global, uses `category_routing_validation.probe_fetch_k` as both the requested probe size and raw candidate cap, and is separate from final retrieval. If probe support meets all configured thresholds, Pipeline 1 performs a second retrieval restricted to the predicted category. Otherwise, final retrieval is global.

Adaptive routing is configured under `retrieval.category_routing_validation`:

```yaml
retrieval:
  retriever_type: "adaptive_category_aware_dense"
  category_routing_validation:
    enabled: true
    probe_fetch_k: 20
    minimum_category_share: 0.60
    minimum_category_count: 3
    minimum_margin: 2
```

The adaptive probe computes `predicted_category_count`, `predicted_category_share`, `strongest_competing_category`, `competing_category_count`, `support_margin`, total probe candidates, and average score by category. Probe candidate IDs, categories, and scores are persisted in aligned order. Dense FAISS and pgvector probe scores use the existing backend similarity ranking semantics: higher values rank earlier.

Per-question adaptive diagnostics include the predicted category, validation status, probe fetch size, probe candidates and scores, thresholds, routing decision, decision reason, final retrieval mode, fallback reason, and final chunk IDs. The run manifest includes whether routing validation was enabled, thresholds, probe fetch size, accepted count, rejected count, invalid-category count, and global-fallback count.

## Active Pipeline 1 Config

- `configs/pipeline1/experiments/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml`

Baseline run command:

```bash
python -m src.pipeline1.main --config configs/pipeline1/experiments/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml
```

Expected Pipeline 1 output:

- `data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl`
- `data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/run_manifest.json`
- `data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/logs.txt`

Pipeline 1 requires the configured local generation service and embedding dependencies at runtime. Cleanup and static checks must not start services, load models, build indexes, or execute retrieval/generation.

## Pipeline 2 Evaluation

Pipeline 2 defaults are SIVAS-first:

- `configs/pipeline2/base_eval.yaml`
- `qa_path: data/raw/qa_ground_truth_fixed.jsonl`
- `pipeline1_results_path: data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl`

Retrieval metrics can be computed independently at two levels:

- Document-level metrics evaluate source-document discovery and ranking using the original SIVAS retrieval evidence.
- Chunk-level metrics evaluate retrieval of human-validated evidence-bearing production chunks. These labels are derived from canonical evidence spans, but they are specific to the exact production chunking configuration used to generate the chunks.

The original SIVAS raw dataset remains unchanged. The chunk-level benchmark under `data/ground_truth/chunk_level/` is a derived, human-validated extension and must not be reused across different chunk layouts.

For official benchmark configurations, chunk-level retrieval evaluation is mandatory.
Official Pipeline 2 YAML files must enable chunk evaluation, use
`missing_question_policy: error`, and reference the annotation package generated
for the exact production chunk boundaries used by the corresponding Pipeline 1 run.
Pipeline 2 treats detectable annotation-package mismatches as hard failures.

Official derived chunk annotation packages:

- `data/ground_truth/chunk_level/B00_sivas_character2048_overlap0`
- `data/ground_truth/chunk_level/E00-G_sentence512_overlap200`
- `data/ground_truth/chunk_level/C01_sentence256_overlap100`
- `data/ground_truth/chunk_level/C02_sentence1024_overlap400`
- `data/ground_truth/chunk_level/E91-E98_fixed512_overlap64`

B00 is the SIVAS-compatible adaptive category-aware pgvector baseline. It uses SIVAS character chunking, Mistral Embed, pgvector, and `adaptive_category_aware_dense`: category predictions are validated through a global pgvector probe, accepted predictions use category-restricted retrieval, and rejected, missing, or invalid predictions use global retrieval. It should not be described as exact SIVAS production reproduction unless the original SIVAS confidence and threshold logic is separately implemented and verified.

### Official C-Series Experiments

C00-C02 evaluate sentence chunking under fixed global dense retrieval:

- C00: sentence 512 tokens / 200-token overlap baseline.
- C01: sentence 256 tokens / 100-token overlap.
- C02: sentence 1024 tokens / 400-token overlap.
- C05: original SIVAS character chunking with 2048-character ceiling and zero overlap.

C03 is a parent-context retrieval ablation, not a chunk-size experiment. It keeps C00's sentence 512/200 chunking, `retriever_type: dense`, FAISS cosine backend, embedding model, reranker setting, orchestration-disabled state, generation model, and prompts fixed. The only introduced variable relative to C00 is `parent_context.enabled: true`, which expands retrieved child chunks to their deepest-containing Markdown parent sections before generation.

C05 isolates the SIVAS character-based chunking strategy inside the controlled local RAG framework. It reuses B00's `sivas_character` chunking configuration only; it does not inherit B00's Mistral embeddings, pgvector backend, adaptive category-aware retrieval, Mistral orchestration, or Mistral generation. C05 uses the `B00_sivas_character2048_overlap0` chunk-level annotation package because it produces the same chunk inventory.

Regenerate derived packages only from canonical validated evidence spans:

```bash
python scripts/build_chunk_annotation_packages.py
```

This command prepares annotation packages only; it is not a benchmark dry run.

Example combined configuration:

```yaml
retrieval_evaluation:
  document_level:
    enabled: true

  chunk_level:
    enabled: true
    ground_truth_path: data/ground_truth/chunk_level/E00-G_sentence512_overlap200/gold_chunk_annotations_E00-G_sentence512_overlap200.jsonl
    missing_question_policy: error
```

Run evaluation only after Pipeline 1 has produced `results.jsonl`:

```bash
python -m src.pipeline2.main --config configs/pipeline2/experiments/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml
```

Before official experiments, use the locked benchmark environment from
`requirements-lock.txt`. FAISS requires the pinned NumPy 1.x ABI:

```text
numpy==1.26.4
faiss-cpu==1.8.0.post1
bert-score==0.3.13
```

Install and repair dependencies with the active interpreter, not bare `pip`:

```bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements-lock.txt -c constraints.txt
python -m pip install --no-deps -e .
python scripts/check_benchmark_environment.py
```

The logical E91-E98 experiment names map to numeric config filenames in
`configs/official_experiment_mapping.yaml`. E91 is Prompt V0 and explicitly pins
`src/pipeline1/prompts/orchestration_prompt.txt`.

Expected Pipeline 2 output:

- `data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/per_question.jsonl`
- `data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/summary_by_experiment.csv`
- `data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/eval_manifest.json`

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
python scripts/index_pgvector.py      --config configs/pipeline1/smoke/smoke_pgvector_dense.yaml
python scripts/index_elasticsearch.py --config configs/pipeline1/smoke/smoke_elasticsearch_bm25.yaml
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
