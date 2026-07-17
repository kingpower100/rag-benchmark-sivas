# B00 Remote Setup — SIVAS pgvector Reference Baseline

This document is structured for a **split local/remote workflow**:

- All code changes are made **locally** (on this machine).
- All execution (indexing, Ollama, Mistral API, PostgreSQL, GPU) runs **remotely**.
- No remote command in this document has been executed for you.

## B00 LLM configuration summary

| Component | Provider | Model | API key required? |
|---|---|---|---|
| Embedding | Mistral API | `mistral-embed` | Yes — `MISTRAL_API_KEY` |
| Orchestration | Local Ollama | `mistral-small` | No |
| Generation | Local Ollama | `mistral-small` *(temporary — see note below)* | No |

Only embedding calls the Mistral API.
Orchestration and generation run against the local Ollama server (`http://localhost:11434`).

> **Temporary implementation note:**
> The original SIVAS reference system uses Mistral Medium for answer generation.
> For development and framework validation, this implementation temporarily uses
> Mistral Small as the local generation model.
> The framework remains fully configurable and can be switched back to Mistral Medium
> by changing a single line in the B00 Pipeline 1 YAML:
> `generation.model_name: "mistral-medium"`
> No code changes are required.

---

## A. Local development workflow

The following changes are already applied to the local repository:

### New source files

| File | Purpose |
|---|---|
| `src/pipeline1/chunking/sivas_character_chunker.py` | Exact SIVAS partner regex chunker, 2048-char ceiling, no overlap |
| `src/pipeline1/embedding/mistral_embedder.py` | Mistral Embed API provider (reads `MISTRAL_API_KEY` from env) |
| `src/pipeline1/generation/mistral_generator.py` | Mistral Chat Completions provider (reads `MISTRAL_API_KEY` from env) |
| `tests/unit/test_sivas_character_chunker.py` | 37 unit tests for the SIVAS chunker (all mocked, no network) |
| `tests/unit/test_mistral_embedder.py` | 16 unit tests for the Mistral embedder (all mocked, no API calls) |
| `configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml` | B00 Pipeline 1 config |
| `configs/pipeline2/final_experiments/B00_sivas_pgvector_reference_eval.yaml` | B00 Pipeline 2 evaluation config |
| `configs/pipeline3/final_experiments/B00_sivas_pgvector_reference_eval.yaml` | B00 Pipeline 3 evaluation config |
| `docs/B00_REMOTE_SETUP.md` | This file |

### Modified source files

| File | Change |
|---|---|
| `src/pipeline1/schemas/config_schema.py` | Added `"sivas_character"` to `ChunkingConfig.strategy`; added `"mistral"` to `EmbeddingConfig.provider`, `OrchestrationConfig.provider`, `GenerationConfig.provider`; `chunk_overlap` remains a required field with no default (B00 YAML sets it explicitly to 0) |
| `src/pipeline1/stages/chunking_stage.py` | Added dispatch for `sivas_character` strategy |
| `src/pipeline1/embedding/factory.py` | Added dispatch for `provider: "mistral"` → `MistralEmbedder` |
| `src/pipeline1/generation/factory.py` | Added dispatch for `provider: "mistral"` → `MistralGenerator` |

### Local validation results

```bash
python -c "from src.pipeline1.schemas.config_schema import PipelineConfig; \
  PipelineConfig.from_yaml('configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml')"
```
**Result: PASSED**
- experiment_id: B00_sivas_pgvector_reference
- strategy: sivas_character / max_chunk_chars: 2048 / chunk_overlap: 0 (required field, explicit)
- embedding.provider: mistral / model: mistral-embed / dim: 1024 / metric: cosine
- index.type: pgvector / dsn_env: PGVECTOR_DSN / schema: rag / table: chunk_embeddings
- retriever_type: category_aware_dense / top_k: 5 / fetch_k: 20
- orchestration: ollama / mistral-small / http://localhost:11434 (no Mistral API key)
- generation: ollama / mistral-small / http://localhost:11434 (no Mistral API key)
  (temporary — original SIVAS system uses mistral-medium; change model_name to restore)
- No FAISS index or retriever configured

```bash
python -c "from src.pipeline2.schemas.eval_config_schema import EvalConfig; \
  EvalConfig.from_yaml('configs/pipeline2/final_experiments/B00_sivas_pgvector_reference_eval.yaml')"
```
**Result: PASSED**
- eval_run_id: B00_sivas_pgvector_reference_eval
- retrieval.ks: [1, 3, 5] / k: 5
- embedding_similarity: sentence_transformers / intfloat/multilingual-e5-large
- bert_score: enabled / bert-base-multilingual-cased

```bash
python -c "from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config; \
  Pipeline3Config.from_yaml('configs/pipeline3/final_experiments/B00_sivas_pgvector_reference_eval.yaml')"
```
**Result: PASSED**
- run_id: B00_sivas_pgvector_reference
- judge.model: qwen2.5:14b / max_context_chars: 6000 (matches base — fair comparison)
- ragas: faithfulness / answer_relevancy / context_recall (all enabled)
- llm_judge metrics: correctness / faithfulness / completeness / hallucination / context_relevance
- weights sum: 1.0

### Local test results

```bash
python -m pytest tests/unit/test_sivas_character_chunker.py \
                 tests/unit/test_mistral_embedder.py -v
```
**Result: 53/53 passed** (no network calls, fully mocked)

---

## B. Push changes to Git

Run these commands locally before SSHing to the remote server.

```bash
# Check what will be committed
git status
git branch --show-current

# Stage all B00 files
git add \
  src/pipeline1/chunking/sivas_character_chunker.py \
  src/pipeline1/embedding/mistral_embedder.py \
  src/pipeline1/generation/mistral_generator.py \
  src/pipeline1/schemas/config_schema.py \
  src/pipeline1/stages/chunking_stage.py \
  src/pipeline1/embedding/factory.py \
  src/pipeline1/generation/factory.py \
  tests/unit/test_sivas_character_chunker.py \
  tests/unit/test_mistral_embedder.py \
  configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml \
  configs/pipeline2/final_experiments/B00_sivas_pgvector_reference_eval.yaml \
  configs/pipeline3/final_experiments/B00_sivas_pgvector_reference_eval.yaml \
  docs/B00_REMOTE_SETUP.md

git diff --staged --stat

git commit -m "Add SIVAS-compatible B00 pgvector baseline

- SivasCharacterChunker: exact partner regex, 2048-char ceiling, no overlap
- MistralEmbedder: mistral-embed API, batching, retry, no key logging
- MistralGenerator: Mistral chat completions provider
- Schema: sivas_character strategy, mistral provider; chunk_overlap required (no default)
- B00 P1 config: mistral-embed + pgvector + category-aware dense
  orchestration=ollama/mistral-small, generation=ollama/mistral-small (temporary; original: mistral-medium)
- B00 P2 eval config: extends base_eval.yaml; all standard metrics
- B00 P3 eval config: extends base_pipeline3.yaml; matches base judge+RAGAS setup
- 53 new unit tests (53/53 pass)"

git push
```

---

## C. Remote server setup

SSH into the remote server, then run the following commands.

### C1. Pull the latest changes

```bash
cd /path/to/rag-benchmark
git status
git branch --show-current
git pull
```

### C2. Activate the Python environment

```bash
source .venv/bin/activate
```

### C3. Install updated dependencies

```bash
pip install -e .
python -m pip check
```

No new third-party packages are required. `MistralEmbedder` uses `requests`,
which is already listed in `pyproject.toml`.  Orchestration and generation use
the existing `OllamaGenerator` — no Mistral client library is needed.

### C4. Pull required Ollama models for B00

B00 uses local Ollama for orchestration (`mistral-small`) and generation
(`mistral-small` — temporary; original SIVAS system uses `mistral-medium`).

```bash
ollama pull mistral-small    # both orchestration and generation (temporary configuration)
```

To restore the original SIVAS generation model later:

```bash
ollama pull mistral-medium
# Then set generation.model_name: "mistral-medium" in B00_sivas_pgvector_reference.yaml
```

Verify:

```bash
ollama list | grep mistral-small
```

If `ollama pull` returns a 404 or "model not found" error, the Ollama registry may use
different names. See **Section L → Mistral model ID mapping** for versioned alternatives.

---

## D. Remote secrets

Set these environment variables **in the remote shell session only**.
Never write secrets into YAML files or source code.

```bash
# Required: Mistral Embed API (embedding only — orchestration/generation use local Ollama)
export MISTRAL_API_KEY="<your-mistral-api-key>"

# Required: PostgreSQL connection string
export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
```

Adjust the DSN to match your PostgreSQL host, port, user, password, and database.

Verify without displaying the secret:

```bash
test -n "$MISTRAL_API_KEY" && echo "MISTRAL_API_KEY is set" || echo "MISTRAL_API_KEY MISSING"
test -n "$PGVECTOR_DSN"    && echo "PGVECTOR_DSN is set"    || echo "PGVECTOR_DSN MISSING"
```

If the project supports `.env`, create it locally (it is already in `.gitignore`
as confirmed in the repository — `.env` and `.env.*` are excluded):

```bash
# .env (never commit this file)
MISTRAL_API_KEY=<your-key>
PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag
```

---

## E. Remote PostgreSQL + pgvector startup

### E1. Start the PostgreSQL container

```bash
# Compose file: infra/docker/docker-compose.yml
# Service name: postgres  (image: pgvector/pgvector:pg16, container: rag-benchmark-postgres)

docker compose -f infra/docker/docker-compose.yml up -d postgres

# Check container status
docker compose -f infra/docker/docker-compose.yml ps postgres

# Wait for healthy
docker compose -f infra/docker/docker-compose.yml logs postgres --tail 20
```

### E2. Verify the pgvector extension

```bash
psql "$PGVECTOR_DSN" -c "
SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';
"
```

Expected output: one row with `vector | <version>`.

### E3. Initialize schema, table, and HNSW index

```bash
PGVECTOR_DSN="$PGVECTOR_DSN" python scripts/init_pgvector.py
```

Optional overrides (defaults match B00):

```bash
PGVECTOR_DSN="$PGVECTOR_DSN" \
  PG_SCHEMA=rag \
  PG_TABLE=chunk_embeddings \
  DENSE_DIM=1024 \
  INDEX_TYPE=hnsw \
  HNSW_M=16 \
  HNSW_EF_CONSTRUCTION=64 \
  python scripts/init_pgvector.py
```

### E4. Verify the table

```bash
psql "$PGVECTOR_DSN" -c "\d rag.chunk_embeddings"
```

Expected: columns `chunk_id TEXT`, `embedding vector(1024)`, `category TEXT`.

---

## F. Remote smoke tests

### F1. Smoke test — pgvector infrastructure (sentence_transformers embedder)

The existing smoke config uses `BAAI/bge-m3` (sentence_transformers) to validate
that the pgvector backend is reachable and the HNSW index responds.
It does **not** validate the Mistral API adapter.

```bash
# Index smoke documents (uses BAAI/bge-m3 on GPU)
PGVECTOR_DSN="$PGVECTOR_DSN" \
  python scripts/index_pgvector.py \
    --config configs/pipeline1/smoke/smoke_pgvector_dense.yaml

# Execute smoke test
PGVECTOR_DSN="$PGVECTOR_DSN" \
  python scripts/run_config.py \
    configs/pipeline1/smoke/smoke_pgvector_dense.yaml
```

This smoke test verifies:
- PostgreSQL connection and vector extension
- Vector insertion via `psycopg2`
- pgvector HNSW retrieval (category-aware SQL WHERE clause)
- Pipeline 1 output format compatibility

It does **not** verify the Mistral Embed API.

### F2. Minimal Mistral Embed verification

Run a single embedding call to confirm the API key and network path are working
before launching full indexing.

```bash
python -c "
import os
from src.pipeline1.embedding.mistral_embedder import MistralEmbedder

# Verifies MISTRAL_API_KEY is set (raises EnvironmentError if not)
embedder = MistralEmbedder()
print('Constructor OK — MISTRAL_API_KEY found')

result = embedder.encode_query('SIVAS ERP Wissensmanagement Test')
print(f'Embedding shape: {result.shape}')    # expected: (1024,)
print(f'Embedding dtype: {result.dtype}')    # expected: float32
print(f'L2 norm: {(result**2).sum()**0.5:.4f}')  # expected: near 1.0
"
```

If this prints `Embedding shape: (1024,)` without error, the Mistral API path is working.

---

## G. Remote B00 indexing

Index all B00 documents into PostgreSQL using Mistral Embed:

```bash
PGVECTOR_DSN="$PGVECTOR_DSN" \
  python scripts/index_pgvector.py \
    --config configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml
```

This command:
- Loads `data/raw/kb_documents_fixed.jsonl`
- Applies `SivasCharacterChunker` (exact SIVAS regex, 2048-char ceiling, no overlap)
- Calls Mistral Embed API in batches of 32 to produce 1024-dim embeddings
- Upserts all chunk embeddings into `rag.chunk_embeddings` via pgvector
- Is idempotent — safe to re-run

Locate the output (number of indexed chunks):

```bash
psql "$PGVECTOR_DSN" -c "SELECT COUNT(*) FROM rag.chunk_embeddings;"
```

---

## H. Remote B00 Pipeline 1 execution

```bash
PGVECTOR_DSN="$PGVECTOR_DSN" \
  python scripts/run_config.py \
    configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml
```

Output location:

```
data/runs/pipeline1/B00_sivas_pgvector_reference/results.jsonl
data/runs/pipeline1/B00_sivas_pgvector_reference/results.csv   (if runtime.save_csv: true)
```

---

## I. Remote Pipeline 2 evaluation

Entry point confirmed by inspection of `src/pipeline2/main.py`:

```python
# src/pipeline2/main.py
parser.add_argument("--config", required=True, ...)
EvaluationOrchestrator().run(args.config)
```

Exact command:

```bash
python -m src.pipeline2.main \
  --config configs/pipeline2/final_experiments/B00_sivas_pgvector_reference_eval.yaml
```

Pipeline 2 reads:

```
data/runs/pipeline1/B00_sivas_pgvector_reference/results.jsonl
data/raw/questions_fixed.jsonl
data/raw/qa_ground_truth_fixed.jsonl
```

Output: `data/eval/runs/pipeline2/B00_sivas_pgvector_reference_eval/`

---

## J. Remote Pipeline 3 evaluation

Entry point confirmed by inspection of `src/pipeline3/main.py`:

```python
# src/pipeline3/main.py
parser.add_argument("--config", required=True, ...)
Pipeline3Orchestrator().run(args.config)
```

Exact command:

```bash
python -m src.pipeline3.main \
  --config configs/pipeline3/final_experiments/B00_sivas_pgvector_reference_eval.yaml
```

Pipeline 3 requires Ollama for the LLM judge and optionally CUDA for RAGAS
embeddings (see `base_pipeline3.yaml`: `require_cuda: true`).

Pipeline 3 reads:

```
data/runs/pipeline1/B00_sivas_pgvector_reference/results.jsonl
data/raw/questions_fixed.jsonl
data/raw/qa_ground_truth_fixed.jsonl
```

Ensure Ollama has the required models before running:

```bash
ollama pull qwen2.5:14b          # LLM judge (base_pipeline3.yaml: judge.model)
ollama pull qwen2.5:7b-instruct  # RAGAS LLM (base_pipeline3.yaml: ragas.llm_model)
```

Output: `data/eval/runs/pipeline3/B00_sivas_pgvector_reference/`

---

## K. Full remote execution sequence

All commands below are verified against the repository.
Run them in order. Each step assumes the previous step succeeded.

```bash
# ── Step 1: SSH to the remote server ─────────────────────────────────────────
ssh user@remote-host

# ── Step 2: Change to repository directory ───────────────────────────────────
cd /path/to/rag-benchmark

# ── Step 3: Check branch and working tree ────────────────────────────────────
git status
git branch --show-current

# ── Step 4: Pull the latest code ─────────────────────────────────────────────
git pull

# ── Step 5: Activate the environment ─────────────────────────────────────────
source .venv/bin/activate

# ── Step 6: Install updated dependencies ─────────────────────────────────────
pip install -e .

# ── Step 7: Check dependencies ───────────────────────────────────────────────
python -m pip check

# ── Step 8: Export MISTRAL_API_KEY (embedding only) ──────────────────────────
export MISTRAL_API_KEY="<your-mistral-api-key>"
test -n "$MISTRAL_API_KEY" && echo "MISTRAL_API_KEY is set" || echo "MISTRAL_API_KEY MISSING"

# ── Step 9: Export PostgreSQL DSN ────────────────────────────────────────────
export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
test -n "$PGVECTOR_DSN" && echo "PGVECTOR_DSN is set" || echo "PGVECTOR_DSN MISSING"

# ── Step 10: Start PostgreSQL/pgvector ───────────────────────────────────────
# Compose file: infra/docker/docker-compose.yml — service name: postgres
docker compose -f infra/docker/docker-compose.yml up -d postgres
docker compose -f infra/docker/docker-compose.yml ps postgres

# ── Step 11: Verify the vector extension ─────────────────────────────────────
psql "$PGVECTOR_DSN" -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
# Expected: one row  vector | <version>

# ── Step 12: Initialize the database (idempotent) ────────────────────────────
PGVECTOR_DSN="$PGVECTOR_DSN" python scripts/init_pgvector.py
# Verifies: rag schema, chunk_embeddings table, HNSW index (dim=1024, m=16, ef=64)

# ── Step 13: Verify Ollama is running ────────────────────────────────────────
curl -s http://localhost:11434/api/tags | python -c "import sys,json; d=json.load(sys.stdin); print('Ollama OK —', len(d.get('models',[])), 'models')"

# ── Step 14: Pull B00 Ollama models ──────────────────────────────────────────
ollama pull mistral-small    # orchestration + generation (temporary; see TEMPORARY IMPLEMENTATION NOTE)

# Pull Pipeline 3 judge/RAGAS models (if not already present)
ollama pull qwen2.5:14b          # LLM judge
ollama pull qwen2.5:7b-instruct  # RAGAS LLM

# ── Step 15: Verify Ollama models ────────────────────────────────────────────
ollama list
# Must show: mistral-small (used for both orchestration and generation), qwen2.5:14b, qwen2.5:7b-instruct

# ── Step 16: pgvector smoke test (validates infra with sentence_transformers) ─
PGVECTOR_DSN="$PGVECTOR_DSN" \
  python scripts/index_pgvector.py \
    --config configs/pipeline1/smoke/smoke_pgvector_dense.yaml
PGVECTOR_DSN="$PGVECTOR_DSN" \
  python scripts/run_config.py \
    configs/pipeline1/smoke/smoke_pgvector_dense.yaml
# Note: smoke config uses BAAI/bge-m3 (sentence_transformers), not Mistral API.

# ── Step 17: Mistral Embed API smoke test ────────────────────────────────────
python -c "
from src.pipeline1.embedding.mistral_embedder import MistralEmbedder
embedder = MistralEmbedder()
result = embedder.encode_query('SIVAS ERP Wissensmanagement Test')
print('Embedding shape:', result.shape)    # expected: (1024,)
print('dtype:', result.dtype)             # expected: float32
print('L2 norm:', round(float((result**2).sum()**0.5), 4))  # expected: ~1.0
"

# ── Step 18: Index B00 documents ─────────────────────────────────────────────
# Loads kb_documents_fixed.jsonl → SivasCharacterChunker → MistralEmbedder → pgvector
PGVECTOR_DSN="$PGVECTOR_DSN" \
  python scripts/index_pgvector.py \
    --config configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml

# ── Step 19: Execute Pipeline 1 B00 ──────────────────────────────────────────
PGVECTOR_DSN="$PGVECTOR_DSN" \
  python scripts/run_config.py \
    configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml

# ── Step 20: Verify Pipeline 1 output exists ─────────────────────────────────
ls -lh data/runs/pipeline1/B00_sivas_pgvector_reference/results.jsonl
wc -l  data/runs/pipeline1/B00_sivas_pgvector_reference/results.jsonl
# Expected: 96 lines (one per question)

# ── Step 21: Execute Pipeline 2 B00 evaluation ───────────────────────────────
# Entry point: src/pipeline2/main.py → EvaluationOrchestrator().run(config)
python -m src.pipeline2.main \
  --config configs/pipeline2/final_experiments/B00_sivas_pgvector_reference_eval.yaml

# ── Step 22: Verify Pipeline 2 outputs ───────────────────────────────────────
ls data/eval/runs/pipeline2/B00_sivas_pgvector_reference_eval/
# Expected files: per_question_metrics.jsonl, summary.json, summary.csv, manifest.json

# ── Step 23: Execute Pipeline 3 B00 evaluation ───────────────────────────────
# Entry point: src/pipeline3/main.py → Pipeline3Orchestrator().run(config)
# Judge: qwen2.5:14b  RAGAS LLM: qwen2.5:7b-instruct  RAGAS emb: multilingual-e5-large
python -m src.pipeline3.main \
  --config configs/pipeline3/final_experiments/B00_sivas_pgvector_reference_eval.yaml

# ── Step 24: Verify Pipeline 3 outputs ───────────────────────────────────────
ls data/eval/runs/pipeline3/B00_sivas_pgvector_reference/
# Expected files: per_question_results.csv, semantic_summary.csv, judge_raw_outputs.jsonl,
#                 judge_failures.jsonl, pipeline3_report.json, manifest.json
```

---

## L. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `EnvironmentError: MISTRAL_API_KEY not set` | Key not exported in shell | `export MISTRAL_API_KEY="<key>"` |
| `EnvironmentError: Mistral API authentication failed` | Wrong or expired key | Regenerate key at console.mistral.ai |
| `ValidationError: model_name not in ALLOWED_ORCHESTRATION_MODELS` | Versioned Mistral model ID not in allow-list | Add `"mistral-small-latest"` to `ALLOWED_ORCHESTRATION_MODELS` in `config_schema.py` |
| `404 Not Found` from Mistral chat API | Model name not available on account | Change `model_name` to `"mistral-small-latest"` / `"mistral-medium-latest"` in B00 YAML |
| `OperationalError: could not connect to PostgreSQL` | PostgreSQL container not running | `docker compose -f infra/docker/docker-compose.yml up -d postgres` |
| `ERROR: type "vector" does not exist` | pgvector extension not installed | `PGVECTOR_DSN="$PGVECTOR_DSN" python scripts/init_pgvector.py` |
| `FileNotFoundError: data/runs/pipeline1/B00_sivas_pgvector_reference/results.jsonl` | Pipeline 1 not run yet | Run step 9 before steps 10/11 |
| `index_pgvector.py: config index.type='pgvector' — expected 'pgvector'` | Wrong config passed | Confirm `--config configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml` |

### Mistral model ID mapping

The `model_name` strings in B00 are repository aliases.  If the Mistral API
rejects them, replace with the versioned IDs:

| B00 alias | Mistral API ID (verify at docs.mistral.ai) |
|---|---|
| `mistral-small` | `mistral-small-latest` |
| `mistral-medium` | `mistral-medium-latest` |

To add a versioned ID to the orchestration allow-list, edit `config_schema.py`:

```python
ALLOWED_ORCHESTRATION_MODELS = frozenset({
    "mistral-small",
    "mistral-small-latest",   # ← add if needed
    "qwen2.5:7b",
    "llama3.1:8b",
})
```
