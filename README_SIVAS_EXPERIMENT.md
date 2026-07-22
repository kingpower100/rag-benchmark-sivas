# SIVAS ERP RAG Benchmark — Remote Server Experiment Guide

> **WARNING: Never use synthetic, fake, or dummy data for benchmark results.**
> The Pipeline 1 output directory must be empty before the first real run.
> Any `results.jsonl` file in `data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/`
> that was not produced by a full Pipeline 1 run on the DGX server is invalid
> and must be deleted before you start.

---

## 1. Project Purpose

This repository implements a RAG (Retrieval-Augmented Generation) benchmark over the SIVAS ERP internal documentation corpus. It measures retrieval quality and answer quality for 96 German-language questions across five ERP categories (Technik, Vertrieb, Materialwirtschaft, Einkauf, Service).

**Two pipelines:**

| Pipeline | Role | Entry point |
|---|---|---|
| Pipeline 1 | RAG execution | `python -m src.pipeline1.main` |
| Pipeline 2 | Offline evaluation | `python -m src.pipeline2.main` |

---

## 2. Dataset Files

All three files live in `data/raw/`. They must not be modified.

| File | Used by | Rows | Description |
|---|---|---|---|
| `kb_documents_fixed.jsonl` | Pipeline 1 only | 65 | SIVAS knowledge-base documents |
| `questions_fixed.jsonl` | Pipeline 1 only | 96 | German ERP questions |
| `qa_ground_truth_fixed.jsonl` | Pipeline 2 only | 96 | Reference answers + retrieval evidence |

Pipeline 1 must **never** read `qa_ground_truth_fixed.jsonl`.  
Pipeline 2 reads `questions_fixed.jsonl` only for three-way ID alignment validation, not for metric computation.

---

## 3. Benchmark Configuration Semantics

### Retrieval

`retrieval.fetch_k` is the maximum number of raw backend candidates requested for one retrieval call. It is a hard cap: the runtime must not expand candidate depth beyond this value. `retrieval.top_k` is the maximum number of final contexts after reranking and chunk-ID deduplication. Deduplication, category constraints, metadata filtering, or malformed candidates may produce fewer than `top_k` final contexts.

For category-aware retrieval, global fallback is controlled only by `retrieval.fallback_to_global`. When it is `true`, invalid categories or insufficient category-scoped results may fall back to global retrieval. When it is `false`, no global retrieval is allowed for category-aware routing; the run records the disabled-fallback reason in retrieval diagnostics.

### Retrieval Evaluation Granularity

Pipeline 2 supports two independent retrieval evaluation levels.

Document-level relevance uses the original SIVAS source-document annotations. Each retrieved chunk is mapped to its source-document identifier and matched against the gold-relevant source documents. Therefore, document Hit@k, Recall@k, MRR@k, nDCG@k and Precision@k measure source-document discovery and ranking, not exact answer-passage localization.

Document Hit@k is the proportion of questions for which at least one of the top-k retrieved chunks originates from a gold-relevant source document. A document-level hit does not guarantee that the retrieved chunk contains the exact answer evidence.

Chunk-level relevance uses human-validated evidence-bearing production chunk IDs. The chunk annotations are derived from canonical evidence spans, but the generated chunk labels are specific to one production chunking configuration. Do not reuse a chunk annotation package for a different chunking strategy, chunk size, overlap, tokenizer, or boundary policy. Chunk-level evaluation compares `retrieved_chunk_ids` from Pipeline 1 against `gold_relevant_chunk_ids` from the configured annotation JSONL.

The original SIVAS raw dataset remains unchanged. The chunk-level benchmark is a derived, human-validated extension under `data/ground_truth/chunk_level/`.

Official Pipeline 2 configurations must enable chunk-level retrieval evaluation with
`missing_question_policy: error`. The currently supported official annotation
packages are:

- `B00_sivas_character2048_overlap0`
- `E00-G_sentence512_overlap200`
- `C01_sentence256_overlap100`
- `C02_sentence1024_overlap400`
- `E91-E98_fixed512_overlap64`

Pipeline 2 treats detectable annotation-package incompatibility as a hard failure.
Regenerate derived packages only from canonical evidence spans:

```bash
python scripts/build_chunk_annotation_packages.py
```

This command is data preparation, not a benchmark dry run.

Document-level and chunk-level evaluation can be enabled independently:

```yaml
retrieval_evaluation:
  document_level:
    enabled: true

  chunk_level:
    enabled: true
    ground_truth_path: data/ground_truth/chunk_level/E00-G_sentence512_overlap200/gold_chunk_annotations_E00-G_sentence512_overlap200.jsonl
    missing_question_policy: error
```

Identifier normalization currently compares source-document basenames case-insensitively after whitespace normalization and maps known chunk suffixes such as `.md_chunk_17` back to `.md`. This preserves existing official behavior, but duplicate basenames across different full paths are reported in evaluation manifests because future evidence-level evaluation should use full paths or stable document IDs.

### Sentence Chunking

Sentence chunking preserves sentence boundaries wherever possible. Official configs must state both `chunk_size_unit` and `chunk_overlap_unit`; do not describe a setting such as `512/200` without the units. Supported units are `tokens`, `words`, `sentences`, and `characters`. Token units use the configured `tokenizer_name` as a tiktoken encoding. A single sentence larger than the configured chunk size is emitted as one oversized sentence chunk so the chunker always makes forward progress.

Changing chunk units or tokenizer changes chunk IDs, chunk caches, embeddings, and FAISS index cache keys. Do not reuse old official outputs after changing these fields.

### SIVAS Character Chunking

`sivas_character_v2` preserves source text exactly. It locates boundaries with `(?<=[.!?;:])\s+|\n\n|\n(?=#{1,6}\s)|\n(?=-\s)`, assigns each matched separator to exactly one contiguous source span, and emits chunks by slicing the original document text. For each document, concatenating chunk text must reconstruct the source text exactly.

This version changes B00 chunk text and character offsets relative to the older normalized implementation. Delete or bypass old B00 chunk and embedding caches, rebuild B00 chunks, and rebuild pgvector rows before using B00 results. Do not reuse existing pgvector rows generated with the older SIVAS chunks; the pgvector manifest now includes chunk content and offset fingerprints in addition to chunk IDs.

### Configuration Reference

| Field | Type | Runtime effect | Valid values | Default | Backend scope | Deprecated |
|---|---|---|---|---|---|---|
| `retrieval.fallback_to_global` | boolean | Enables or forbids global fallback after invalid category or insufficient category results | `true`, `false` | `true` | `category_aware_dense` | No |
| `retrieval.fetch_k` | integer | Hard maximum raw candidates requested from the retriever backend | `>= top_k` | required | all retrievers | No |
| `index.dense_dim` | integer | Validated against generated embedding dimension; pgvector/Elasticsearch also use it for vector field dimensions | `> 0` | `384` | all vector backends | No |
| `index.index_name` | string | Names external Elasticsearch indexes; isolates FAISS cache identity when `index.type: faiss` | non-empty string | `sivas_fixed512_bge_small` | FAISS, Elasticsearch | No |
| `orchestration.prompt_path` | string | Selects the orchestration prompt file loaded at runtime | existing file path | default prompt path | orchestration | No |
| `orchestration.prompt_version` | string | Validation label; must match `prompt_path` stem when both are set | matching prompt label | `null` | orchestration | No |
| `orchestration.tasks` | list | No behavioral control in current fixed orchestration workflow | remove field | fixed default | orchestration | Yes |
| `generation.configurable` | boolean | No runtime effect | remove field | `false` | generation | Yes |
| `generation.temperature` | number | Passed to the generation provider | provider-supported number | `0.0` | generation | No |
| `generation.max_tokens` | integer | Passed to the generation provider as answer token budget | `> 0` | `512` | generation | No |
| `chunking.tokenizer_name` | string | Used by `fixed_token` chunking, token-based sentence chunking, and prompt/context token budgeting | valid tiktoken encoding | `cl100k_base` | chunking/generation budgeting | No |
| `chunking.max_chunk_chars` | integer | Enforced by `sivas_character` and `table_aware`; otherwise used for chunk diagnostics and oversized policy checks | `> 0` | `8000` | chunker-dependent | No |
| `chunking.max_chunk_tokens` | integer | Enforced by `table_aware`; otherwise used for chunk diagnostics and oversized policy checks | `> 0` | `1800` | chunker-dependent | No |
| `retrieval.bm25.enabled` | boolean | Removed; BM25 activation is controlled only by `retrieval.retriever_type` | remove field | n/a | retrieval | Yes |
| `bert_score.max_length` | integer | Removed; this framework's official BERTScore wrapper did not consume it | remove field | n/a | Pipeline 2 | Yes |
| `parent_context.parent_unit` | string | Selects parent context unit | `markdown_section` | `markdown_section` | parent context | No |
| `parent_context.deduplicate` | boolean | Controls whether repeated parent sections are deduplicated | `true`, `false` | `true` | parent context | No |
| `parent_context.missing_parent_policy` | string | Controls missing-parent handling | `use_child`, `error` | `use_child` | parent context | No |
| `parent_context.unique_parent_top_k` | integer | Maximum selected parent contexts | `> 0` | `5` | parent context | No |
| `parent_context.max_parent_tokens` | integer | Token limit used during parent selection; oversized parents prefer a deeper fitting section when available | `> 0` | `1800` | parent context | No |
| `parent_context.mapping_policy` | string | Removed; current mapping policy is fixed by the parent-store implementation | remove field | n/a | parent context | Yes |
| `parent_context.score_policy` | string | Removed; parent score/provenance behavior is fixed to child-trigger provenance | remove field | n/a | parent context | Yes |
| `parent_context.preserve_child_provenance` | boolean | Removed; child provenance is mandatory for auditability | remove field | n/a | parent context | Yes |
| `parent_context.oversized_parent_policy` | string | Removed; current behavior always prefers the deepest fitting parent section | remove field | n/a | parent context | Yes |

---

## 4. Active Configs

| Config | Path |
|---|---|
| Pipeline 1 experiment | `configs/pipeline1/experiments/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml` |
| Pipeline 2 evaluation | `configs/pipeline2/base_eval.yaml` |

Pipeline 1 config inherits from `configs/pipeline1/base.yaml` via `extends:`.

Key Pipeline 1 settings:
- Chunking: fixed-token, 512 tokens, 64 overlap
- Embedding: `BAAI/bge-small-en-v1.5` via sentence-transformers, CUDA required
- Index: FAISS cosine
- Retrieval: `category_aware_dense`, top-k=5
- Orchestration LLM: `mistral-small` via Ollama
- Generation LLM: `qwen2.5:7b` via Ollama
- Output: `data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/`
- Resume: enabled (`resume: true`, `overwrite: false`)

Key Pipeline 2 settings:
- Reads: `data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl`
- Ground truth: `data/raw/qa_ground_truth_fixed.jsonl`
- Retrieval eval field: `retrieved_file_names`
- Retrieval evaluation: document level enabled by default; chunk level disabled unless `retrieval_evaluation.chunk_level.enabled: true`
- Metrics at k: 1, 3, 5
- Output: `data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/`

---

## 5. Remote Server Setup

### 5.1 System requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 | Dockerfile target |
| Python | 3.11 | `python3.11` must be available |
| CUDA | 12.1 | For embedding GPU acceleration |
| GPU VRAM | 8 GB | For `qwen2.5:7b` via Ollama |
| RAM | 16 GB | For FAISS index + document loading |
| Disk | 10 GB free | Models + index + outputs |
| Ollama | Latest | Must be installed separately |

### 5.2 System packages (Ubuntu 22.04)

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip build-essential curl ca-certificates
```

### 5.3 Ollama installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 6. Dependency Installation

### Python version

Python **3.11** is required. The `requirements-lock.txt` pins are tested on 3.11.

### Critical version constraint: NumPy + FAISS

`faiss-cpu==1.8.0.post1` was compiled against NumPy 1.x. **NumPy 2.x causes an `ImportError` at runtime.** The lock file pins `numpy==1.26.4` (the last stable 1.x release). Do not upgrade numpy independently.

### FAISS: CPU vs GPU

The pinned package is `faiss-cpu`. If the server has a GPU and you want GPU-accelerated FAISS indexing, replace with:

```
faiss-gpu==1.7.4   # for CUDA 11.x
```

or build from source for CUDA 12.x. For the baseline experiment, `faiss-cpu` is sufficient — the performance bottleneck is Ollama generation, not FAISS search over 65 documents.

### PyTorch — system dependency

**torch is not pinned by this project.** The remote GPU server already provides a
CUDA-enabled PyTorch build. Installing the project should not downgrade or replace
that build.

Recommended workflow on a GPU server that already has the correct torch/CUDA stack:

```bash
# Install project without pulling its own torch
python -m pip install -e . --no-deps

# Install remaining project dependencies (torch is not in the lock file)
python -m pip install -r requirements-lock.txt -c constraints.txt
```

If you are setting up a fresh environment that does not yet have torch, install the
CUDA-compatible wheel **before** the project dependencies:

```bash
# Example for CUDA 12.8 (adjust the index URL to match the server's CUDA version)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# Then install the project
python -m pip install -r requirements-lock.txt -c constraints.txt
python -m pip install --no-deps -e .
```

Check https://pytorch.org/get-started/locally/ for the correct index URL for your
CUDA version. Do not add `+cuXXX` build tags to requirements files.

### Standard install commands (CPU / CI / local development)

```bash
# Clone
git clone <repository-url> rag_benchmark
cd rag_benchmark

# Create virtualenv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install torch first (without project pulling it in)
# On a GPU server: torch is already present — skip this step.
# On CPU / CI: the sentence-transformers and bert-score packages will pull
# in a compatible CPU torch automatically via their own dependencies.
python -m pip install -r requirements-lock.txt -c constraints.txt

# Install project in editable mode (makes src/ importable)
python -m pip install --no-deps -e .

# Verify
python scripts/check_benchmark_environment.py
```

### Full dependency list (from requirements-lock.txt)

```
pydantic==2.8.2
pydantic-core==2.20.1
pyyaml==6.0.2
jsonlines==4.0.0
# torch is NOT listed here — it is a system/environment dependency.
# Install the correct CUDA wheel for your server before running pip install.
sentence-transformers==3.0.1
transformers==4.44.2
huggingface-hub==0.24.6
tokenizers==0.19.1
safetensors==0.4.5
faiss-cpu==1.8.0.post1
numpy==1.26.4          # pinned — do NOT upgrade to 2.x
pandas==2.2.2
tqdm==4.66.5
tiktoken==0.7.0
requests==2.32.3
elasticsearch==8.15.1
pytest==8.3.3
```

### Embedding model download

The embedding model `BAAI/bge-small-en-v1.5` downloads automatically on first Pipeline 1 run from Hugging Face Hub. To pre-download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

Set cache location to match the config:

```bash
export SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers
export HF_HOME=/models/huggingface
```

---

## 6. Ollama Setup

```bash
# Start Ollama server (keep running in background or screen/tmux)
ollama serve &

# Pull required models
ollama pull mistral-small    # orchestration LLM (~12 GB)
ollama pull qwen2.5:7b       # generation LLM (~5 GB)

# Verify both are available
ollama list
```

Expected output of `ollama list` must show both `mistral-small` and `qwen2.5:7b` before running Pipeline 1.

Test Ollama is reachable:

```bash
python scripts/test_ollama.py --base-url http://localhost:11434
```

---

## 7. Running the First Experiment

### Step 1: Verify output directory is empty

```bash
ls data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/
```

Must be empty (or contain only `.gitkeep`). If `results.jsonl` exists from a previous session, verify it is a real run (check `experiment_id` in the first line). Any synthetic file must be deleted:

```bash
# Only if the file is synthetic/incomplete:
rm data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl
```

### Step 2: Run Pipeline 1

```bash
python -m src.pipeline1.main \
  --config configs/pipeline1/experiments/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml
```

Expected runtime: 30–120 minutes depending on GPU availability and Ollama generation speed.

Pipeline 1 supports resume (`resume: true` in the config). If interrupted, rerun the same command and it will skip already-processed questions.

Expected output files:

```
data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/
├── results.jsonl          # one row per question (96 rows for full SIVAS)
├── results.csv            # same data as CSV
├── run_manifest.json      # run metadata and config hash
├── events.jsonl           # per-question event log
└── logs.txt               # runtime log
```

### Step 3: Verify Pipeline 1 output

```bash
# Must output 96
wc -l data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl

# Inspect first row
head -n 1 data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl | python -m json.tool | head -20

# Check experiment_id is NOT synthetic
python -c "
import json
r = json.loads(open('data/runs/pipeline1/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0/results.jsonl').readline())
eid = r.get('experiment_id', '')
ans = r.get('generated_answer', '')
print('experiment_id:', eid)
print('answer[:80]:', ans[:80])
assert eid == '91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0', f'Wrong experiment_id: {eid}'
assert 'Synthetische' not in ans, 'Synthetic answer detected!'
print('OK: Real Pipeline 1 output confirmed.')
"
```

### Step 4: Run Pipeline 2

```bash
python -m src.pipeline2.main \
  --config configs/pipeline2/base_eval.yaml
```

Expected runtime: under 2 minutes (no model inference in Pipeline 2).

### Step 5: Verify Pipeline 2 output

```bash
# Per-question results (must be 96 rows)
wc -l data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/per_question.jsonl

# Summary metrics
cat data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/summary_metrics.json | python -m json.tool

# Leaderboard
cat data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/leaderboard.csv

# Category breakdown
cat data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/summary_by_category.csv

# Audit verdict
python -c "
import json
r = json.loads(open('data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/eval_manifest.json').read())
print('final_verdict:', r['final_verdict'])
print('strict_audit_pass:', r['strict_audit_pass'])
print('total_questions:', r['total_questions'])
"
```

---

## 8. Expected Output Files

### Pipeline 2 output directory

```
data/eval/runs/pipeline2/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0_eval/
├── per_question.jsonl          # per-question metric rows
├── per_question.csv            # same as CSV
├── per_question_metrics.jsonl  # metrics-only rows (no raw text)
├── summary_metrics.json        # aggregated metrics JSON
├── summary_by_experiment.csv   # experiment-level summary
├── summary_by_category.csv     # per-SIVAS-category breakdown
├── summary_by_category.json    # same as JSON
├── leaderboard.csv             # sorted leaderboard
├── leaderboard.md              # human-readable leaderboard
├── eval_manifest.json          # full audit manifest with hashes
└── audit_report.json           # final verdict + detailed audit
```

### Key metrics to check

| Metric | Field in summary_metrics.json |
|---|---|
| Hit@5 | `mean_hit_at_5` |
| Recall@5 | `mean_recall_at_5` |
| MRR@5 | `mean_mrr_at_5` |
| NDCG@5 | `mean_ndcg_at_5` |
| Category Accuracy | `mean_category_accuracy` |
| Embedding Similarity | `mean_embedding_similarity` |
| Pipeline Success Rate | `pipeline_success_rate` |

A valid real run requires `final_verdict: "valid"` in `eval_manifest.json`.

---

## 9. Docker Alternative

If running inside Docker:

```bash
# Build image
docker compose -f infra/docker/docker-compose.yml build

# Run Pipeline 1
docker compose -f infra/docker/docker-compose.yml up pipeline1
```

The compose file mounts:
- `data/raw/` as read-only input
- `data/runs/` as writable output
- `data/processed/` as writable cache

Ollama must be running on the **host** machine. The container reaches it via `http://host.docker.internal:11434`.

---

## 10. Troubleshooting

### `ImportError: numpy.core.multiarray failed to import` (FAISS)

**Cause:** NumPy 2.x installed. FAISS 1.8.x requires NumPy 1.x.

**Fix:**
```bash
python -m pip uninstall -y numpy faiss-cpu faiss-gpu
python -m pip install --no-cache-dir -r requirements-lock.txt -c constraints.txt
python -m pip check
python scripts/check_benchmark_environment.py
```

### `ConnectionRefusedError` or `Ollama not reachable`

**Cause:** Ollama is not running or listening on the wrong port.

**Fix:**
```bash
ollama serve
python scripts/test_ollama.py --base-url http://localhost:11434
```

### `Model 'mistral-small' not found` or `'qwen2.5:7b' not found`

**Fix:**
```bash
ollama pull mistral-small
ollama pull qwen2.5:7b
ollama list
```

### `QA rows have empty answer fields for 96 IDs`

**Cause:** Pipeline 2 did not recognize `referenzantwort` as the answer field. This is fixed in Phase 1 — verify you have the latest code.

**Check:**
```bash
grep "referenzantwort" src/pipeline2/metrics/answer_metrics.py
```

Must output a line inside `resolve_ground_truth_answer()`.

### Pipeline 1 output row count is less than 96

**Cause:** Pipeline 1 was interrupted before completion, or `resume: false` with a partial previous run.

**Fix:** Delete the partial results and rerun, or set `resume: true` (already the default in the experiment config) and rerun to pick up where it stopped:
```bash
python -m src.pipeline1.main \
  --config configs/pipeline1/experiments/91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0.yaml
```

### `final_verdict: invalid` in eval_manifest.json

Check `audit_report.json` for the specific failure. Common causes:
- `exact_set_equality: false` — Pipeline 1 didn't process all 96 questions
- `critical_leakage_found: true` — gold terms found in Pipeline 1 artifacts (not expected; investigate)
- `generation_failure_rate` exceeds threshold — too many questions returned empty answers

### CUDA not available (embedding is slow)

Pipeline 1 config has `require_cuda: true`. If no GPU is available:
1. Set `require_cuda: false` in the experiment config
2. Set `device: "cpu"`
3. Embedding will run on CPU — expect ~10x slower chunking and embedding phase

---

## 11. Data Integrity Rules

1. **Never use synthetic data for benchmark results.** Any `results.jsonl` with `experiment_id` containing `synthetic`, `dryrun`, `fake`, or `dummy` is invalid.
2. **Do not modify `data/raw/` files.** They are the official fixed SIVAS dataset.
3. **Do not copy `qa_ground_truth_fixed.jsonl` into Pipeline 1 inputs.** It must remain evaluation-only.
4. **A real run is valid only if** `eval_manifest.json` shows `final_verdict: "valid"` and `strict_audit_pass: true`.
5. **Pipeline 2 `overwrite: true` will delete previous eval outputs** when rerun. Save important results elsewhere before re-evaluating.
