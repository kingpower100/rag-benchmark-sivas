# SIVAS ERP RAG Benchmark — Remote Server Experiment Guide

> **WARNING: Never use synthetic, fake, or dummy data for benchmark results.**
> The Pipeline 1 output directory must be empty before the first real run.
> Any `results.jsonl` file in `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/`
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

## 3. Active Configs

| Config | Path |
|---|---|
| Pipeline 1 experiment | `configs/pipeline1/experiments/11_sivas_fixed512_faiss_dense_qwen25.yaml` |
| Pipeline 2 evaluation | `configs/pipeline2/base_eval.yaml` |

Pipeline 1 config inherits from `configs/pipeline1/base.yaml` via `extends:`.

Key Pipeline 1 settings:
- Chunking: fixed-token, 512 tokens, 64 overlap
- Embedding: `BAAI/bge-small-en-v1.5` via sentence-transformers, CUDA required
- Index: FAISS cosine
- Retrieval: `category_aware_dense`, top-k=5
- Orchestration LLM: `mistral-small` via Ollama
- Generation LLM: `qwen2.5:7b` via Ollama
- Output: `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/`
- Resume: enabled (`resume: true`, `overwrite: false`)

Key Pipeline 2 settings:
- Reads: `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/results.jsonl`
- Ground truth: `data/raw/qa_ground_truth_fixed.jsonl`
- Retrieval eval field: `retrieved_file_names`
- Metrics at k: 1, 3, 5
- Output: `data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/`

---

## 4. Remote Server Setup

### 4.1 System requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 | Dockerfile target |
| Python | 3.11 | `python3.11` must be available |
| CUDA | 12.1 | For embedding GPU acceleration |
| GPU VRAM | 8 GB | For `qwen2.5:7b` via Ollama |
| RAM | 16 GB | For FAISS index + document loading |
| Disk | 10 GB free | Models + index + outputs |
| Ollama | Latest | Must be installed separately |

### 4.2 System packages (Ubuntu 22.04)

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip build-essential curl ca-certificates
```

### 4.3 Ollama installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 5. Dependency Installation

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

### Install commands

```bash
# Clone
git clone <repository-url> rag_benchmark
cd rag_benchmark

# Create virtualenv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all pinned dependencies
pip install -r requirements-lock.txt

# Install project in editable mode (makes src/ importable)
pip install --no-deps -e .

# Verify
python -c "import faiss; import sentence_transformers; print('Dependencies OK')"
```

### Full dependency list (from requirements-lock.txt)

```
pydantic==2.8.2
pydantic-core==2.20.1
pyyaml==6.0.2
jsonlines==4.0.0
torch==2.4.1
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
ls data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/
```

Must be empty (or contain only `.gitkeep`). If `results.jsonl` exists from a previous session, verify it is a real run (check `experiment_id` in the first line). Any synthetic file must be deleted:

```bash
# Only if the file is synthetic/incomplete:
rm data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/results.jsonl
```

### Step 2: Run Pipeline 1

```bash
python -m src.pipeline1.main \
  --config configs/pipeline1/experiments/11_sivas_fixed512_faiss_dense_qwen25.yaml
```

Expected runtime: 30–120 minutes depending on GPU availability and Ollama generation speed.

Pipeline 1 supports resume (`resume: true` in the config). If interrupted, rerun the same command and it will skip already-processed questions.

Expected output files:

```
data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/
├── results.jsonl          # one row per question (96 rows for full SIVAS)
├── results.csv            # same data as CSV
├── run_manifest.json      # run metadata and config hash
├── events.jsonl           # per-question event log
└── logs.txt               # runtime log
```

### Step 3: Verify Pipeline 1 output

```bash
# Must output 96
wc -l data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/results.jsonl

# Inspect first row
head -n 1 data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/results.jsonl | python -m json.tool | head -20

# Check experiment_id is NOT synthetic
python -c "
import json
r = json.loads(open('data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/results.jsonl').readline())
eid = r.get('experiment_id', '')
ans = r.get('generated_answer', '')
print('experiment_id:', eid)
print('answer[:80]:', ans[:80])
assert eid == '11_sivas_fixed512_faiss_dense_qwen25', f'Wrong experiment_id: {eid}'
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
wc -l data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/per_question.jsonl

# Summary metrics
cat data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/summary_metrics.json | python -m json.tool

# Leaderboard
cat data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/leaderboard.csv

# Category breakdown
cat data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/summary_by_category.csv

# Audit verdict
python -c "
import json
r = json.loads(open('data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/eval_manifest.json').read())
print('final_verdict:', r['final_verdict'])
print('strict_audit_pass:', r['strict_audit_pass'])
print('total_questions:', r['total_questions'])
"
```

---

## 8. Expected Output Files

### Pipeline 2 output directory

```
data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/
├── per_question.jsonl          # per-question metric rows
├── per_question.csv            # same as CSV
├── per_question_metrics.jsonl  # metrics-only rows (no raw text)
├── summary_metrics.json        # aggregated metrics JSON
├── summary_by_experiment.csv   # experiment-level summary
├── summary_by_category.csv     # per-SIVAS-category breakdown
├── summary_by_difficulty.csv   # per-difficulty breakdown
├── summary_by_difficulty.json  # same as JSON
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
| ROUGE-L | `mean_rouge_l` |
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
pip install "numpy==1.26.4"
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
  --config configs/pipeline1/experiments/11_sivas_fixed512_faiss_dense_qwen25.yaml
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
