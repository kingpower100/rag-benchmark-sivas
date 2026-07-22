# Infrastructure Overview

This document describes the complete infrastructure required to run the RAG Benchmark framework with all supported retrieval backends. It is intended to allow another researcher — or the original author — to reproduce the environment from scratch on a Linux server.

## Architecture

```
Laptop (development)
        │
        │  git push / git pull
        ▼
GitHub (source of truth)
        │
        │  git pull / scp / rsync
        ▼
Remote Server (schmidhuber18)
        │
        ├── Ollama               – LLM inference (orchestration + generation)
        ├── PostgreSQL + pgvector – dense vector store (HNSW, cosine, 1024-dim)
        ├── Elasticsearch        – BM25 sparse retrieval
        ├── Kibana               – visual inspection of ES indices
        └── RAG Benchmark Framework (Python, CUDA)
```

### Component Roles

| Component | Role |
|---|---|
| **Laptop** | Code development, config authoring, result analysis |
| **GitHub** | Version-controlled source; sync point between laptop and server |
| **schmidhuber18** | GPU server running all inference, indexing, and retrieval |
| **Ollama** | Serves open-weight LLMs locally for orchestration and answer generation |
| **PostgreSQL + pgvector** | Persists dense chunk embeddings; supports HNSW approximate nearest-neighbour search |
| **Elasticsearch** | Hosts the inverted BM25 index for sparse keyword retrieval |
| **Kibana** | Web UI for inspecting Elasticsearch indices and debugging indexed documents |
| **RAG Benchmark Framework** | Python application (Pipelines 1–3) that orchestrates chunking, indexing, retrieval, and evaluation |

---

# Docker Services

All services are defined in `infra/docker/docker-compose.yml` and managed with Docker Compose.

## PostgreSQL + pgvector

| Property | Value |
|---|---|
| Image | `pgvector/pgvector:pg16` |
| Container name | `rag-benchmark-postgres` |
| Host port | `5432` |
| Container port | `5432` |
| Database | `rag` |
| User / Password | `rag` / `rag` |
| Volume | `pgdata` (named volume, persisted across restarts) |

**Purpose:** Stores dense chunk embeddings (1024-dimensional, cosine metric) in a PostgreSQL table with the `pgvector` extension. The framework uses HNSW indexing for fast approximate nearest-neighbour retrieval at query time. The schema (`rag.chunk_embeddings`) is initialised by `scripts/init_pgvector.py`.

**Health check:** `pg_isready -U rag -d rag` — the container is considered healthy only after PostgreSQL accepts connections.

---

## Elasticsearch

| Property | Value |
|---|---|
| Image | `docker.elastic.co/elasticsearch/elasticsearch:8.15.1` |
| Container name | `rag-benchmark-elasticsearch` |
| Host port | `9201` |
| Container port | `9200` |
| Discovery mode | `single-node` |
| Security | Disabled (`xpack.security.enabled=false`) |
| JVM heap | `1 GB` (`-Xms1g -Xmx1g`) |
| Volume | `esdata` (named volume) |

**Purpose:** Hosts the inverted BM25 index (`rag_benchmark_chunks`) with a German-language analyser (k1=1.5, b=0.75). The BM25 retriever queries Elasticsearch directly at retrieval time. The index is populated by `scripts/index_elasticsearch.py`.

> **Note:** The host port is `9201` (not the default `9200`) to avoid conflicts with any locally installed Elasticsearch instance.

---

## Kibana

| Property | Value |
|---|---|
| Image | `docker.elastic.co/kibana/kibana:8.15.1` |
| Container name | `rag-benchmark-kibana` |
| Host port | `5601` |
| Container port | `5601` |
| Depends on | `elasticsearch` (service_healthy) |
| Restart policy | `unless-stopped` |

**Purpose:** Provides a web-based UI for browsing and querying Elasticsearch indices. Useful for verifying that documents are indexed correctly and for debugging retrieval behaviour. Access is via SSH tunnel from the local machine (see [Kibana section](#kibana-1) below).

---

## Ollama

| Property | Value |
|---|---|
| Container name | `ollama` |
| Host port | `11434` |
| Container port | `11434` |

**Purpose:** Runs open-weight LLMs locally to serve the orchestration and generation steps of the RAG pipeline. The framework connects to Ollama via `OLLAMA_BASE_URL=http://localhost:11434` (or `http://host.docker.internal:11434` from inside a container).

**Installed models:**

| Model | Usage |
|---|---|
| `mistral-small` | Primary generation model (experiments 11–15, 18) |
| `qwen2.5:7b` | Compact generation model |
| `qwen2.5:14b` | Larger generation model |
| `llama3.1:8b` | Alternative generation model (experiment 17) |
| `qwen2.5:7b-instruct` | Instruction-tuned variant |

To verify installed models on the server:

```bash
ollama list
```

---

# Environment Variables

These variables must be exported in the shell before running any indexing or retrieval command. They are **not** stored in config files to keep configurations portable across environments.

| Variable | Example value | Purpose |
|---|---|---|
| `PGVECTOR_DSN` | `postgresql://rag:rag@localhost:5432/rag` | PostgreSQL connection string for the pgvector backend |
| `ELASTICSEARCH_URL` | `http://localhost:9201` | Base URL for the Elasticsearch REST API |

Export them in the current shell session:

```bash
export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
export ELASTICSEARCH_URL="http://localhost:9201"
```

Or add them to `~/.bashrc` / `~/.bash_profile` for persistence across sessions.

The framework reads these at runtime via `os.environ`. The config files reference the variable names (`dsn_env: "PGVECTOR_DSN"`, `host_env: "ELASTICSEARCH_URL"`), not the values.

---

# Starting the Infrastructure

Navigate to the Docker Compose directory and start all services in detached mode:

```bash
cd infra/docker
docker compose up -d
```

Verify that all services are running and healthy:

```bash
docker compose ps
```

Expected output (all services should show `healthy` or `running`):

```
NAME                           IMAGE                                                   STATUS
rag-benchmark-elasticsearch    docker.elastic.co/elasticsearch/elasticsearch:8.15.1   Up (healthy)
rag-benchmark-kibana           docker.elastic.co/kibana/kibana:8.15.1                 Up
rag-benchmark-postgres         pgvector/pgvector:pg16                                 Up (healthy)
```

> Elasticsearch may take 30–60 seconds to become healthy on first start due to JVM warm-up.

---

# Stopping the Infrastructure

**Stop containers (preserve data volumes):**

```bash
docker compose stop
```

This halts all running containers but leaves named volumes (`pgdata`, `esdata`) intact. Data is preserved and services resume immediately on the next `docker compose up -d`.

**Remove containers (preserve data volumes):**

```bash
docker compose down
```

This stops and removes containers and their networks, but named volumes are kept by default. Use this when you want a clean container state without losing indexed data.

**Remove containers and all data (destructive):**

```bash
docker compose down -v
```

This also deletes named volumes. All indexed data (PostgreSQL embeddings, Elasticsearch index) will be lost and must be rebuilt from scratch.

---

# PostgreSQL Verification

Connect to the PostgreSQL container:

```bash
docker exec -it rag-benchmark-postgres psql -U rag -d rag
```

**Verify the pgvector extension is installed:**

```sql
SELECT extname FROM pg_extension;
```

Expected output includes:

```
  extname
----------
 plpgsql
 vector
```

**Verify the embeddings table exists and is populated:**

```sql
SELECT COUNT(*) FROM rag.chunk_embeddings;
```

Expected output after a successful indexing run:

```
 count
-------
  1234
(1 row)
```

The exact count depends on the corpus and chunking configuration.

**Inspect a sample row:**

```sql
SELECT id, chunk_id, kategorie, LEFT(content, 80) AS content_preview
FROM rag.chunk_embeddings
LIMIT 5;
```

Exit the psql session with `\q`.

---

# pgAdmin

pgAdmin (desktop or web) can be used to inspect the PostgreSQL database visually. Connect with the following settings:

| Field | Value |
|---|---|
| Server Name | `RAG pgvector` |
| Host | `localhost` |
| Port | `5432` |
| Database | `rag` |
| Username | `rag` |
| Password | `rag` |

**Navigating to the embeddings table:**

```
Servers
  └── RAG pgvector
        └── Databases
              └── rag
                    └── Schemas
                          └── rag
                                └── Tables
                                      └── chunk_embeddings
```

Right-click `chunk_embeddings` → **View/Edit Data** → **All Rows** to inspect stored chunks and their metadata. The `embedding` column stores the raw vector and is displayed as a byte array; use the `content` and `kategorie` columns to verify correctness.

---

# Elasticsearch Verification

**Check that Elasticsearch is reachable:**

```bash
curl http://localhost:9201
```

Expected response: JSON with `"cluster_name"`, `"status": "green"` or `"yellow"`, and version information.

**List all indices:**

```bash
curl http://localhost:9201/_cat/indices?v
```

Expected output after indexing:

```
health status index                  ...  docs.count  ...
yellow open   rag_benchmark_chunks   ...       1234   ...
```

> Status `yellow` is normal for a single-node cluster (no replicas can be assigned).

**Count documents in the benchmark index:**

```bash
curl http://localhost:9201/rag_benchmark_chunks/_count
```

Expected response:

```json
{"count":1234,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0}}
```

**Inspect a sample document:**

```bash
curl "http://localhost:9201/rag_benchmark_chunks/_search?size=1&pretty"
```

---

# Kibana

Kibana runs on the remote server and is accessed from the local machine via an SSH tunnel.

**Step 1 — Open an SSH tunnel:**

```bash
ssh -L 5601:localhost:5601 lfgaier@schmidhuber18.imla.hs-offenburg.de
```

Keep this terminal open. The tunnel forwards local port 5601 to port 5601 on the remote server.

**Step 2 — Open Kibana in a browser:**

```
http://localhost:5601
```

**Step 3 — Create a Data View (first time only):**

1. Navigate to **Management** → **Stack Management** → **Data Views**.
2. Click **Create data view**.
3. Set **Index pattern** to `rag_benchmark_chunks`.
4. Set **Name** to `rag_benchmark_chunks`.
5. Click **Save data view to Kibana**.

**Step 4 — Inspect indexed chunks:**

1. Navigate to **Discover** (left sidebar).
2. Select the `rag_benchmark_chunks` data view.
3. Browse documents using the search bar and field filters.
4. Use the `content`, `kategorie`, and `chunk_id` fields to verify that chunks are correctly indexed and the German analyser is applied.

---

# Pipeline Integration

The framework supports multiple retrieval backends, selectable via `index.type` and `retrieval.retriever_type` in the experiment YAML config.

| Backend | Config key | Storage | Infrastructure required |
|---|---|---|---|
| **FAISS** | `index.type: faiss` | Local `.faiss` files under `data/processed/` | None (no external service) |
| **PostgreSQL + pgvector** | `index.type: pgvector` | `rag.chunk_embeddings` table in PostgreSQL | PostgreSQL container + `$PGVECTOR_DSN` |
| **Elasticsearch BM25** | `retrieval.retriever_type: bm25` | `rag_benchmark_chunks` ES index | Elasticsearch container + `$ELASTICSEARCH_URL` |
| **Hybrid** | `retriever_type: hybrid` | pgvector (dense) + Elasticsearch (sparse) | Both PostgreSQL and Elasticsearch containers |

### FAISS

Used by experiments 11–18 (the primary benchmark series). Embeddings are stored as local `.faiss` index files and loaded into memory at retrieval time. No external service is required; the framework manages file paths via `data/processed/`.

### PostgreSQL + pgvector

The dense retrieval backend for experiments that require persistent vector storage. Embeddings are inserted into `rag.chunk_embeddings` during indexing (`scripts/index_pgvector.py`) and retrieved via HNSW approximate nearest-neighbour queries at runtime.

### Elasticsearch

The BM25 sparse retrieval backend. Chunks are indexed as plain text with a German-language analyser. At retrieval time, the `ElasticsearchBM25Retriever` issues a BM25 query against the `rag_benchmark_chunks` index and returns the top-k documents by BM25 score.

### Hybrid

Combines pgvector (dense) and Elasticsearch (sparse) retrieval. Results from both backends are merged and re-ranked before passing to the generation step. Both PostgreSQL and Elasticsearch must be running and populated.

---

# Smoke Tests

Before running benchmark experiments, use the smoke-test configurations to validate each retrieval backend independently.

## 99a — pgvector Dense Smoke Test

```
configs/pipeline1/smoke/smoke_pgvector_dense.yaml
```

**Purpose:** Verifies that `CategoryAwareDenseRetriever` in pgvector mode can connect to PostgreSQL, execute SQL `WHERE`-clause category filters, and produce Pipeline 1 output compatible with Pipeline 2 evaluation.

**Prerequisites:**

```bash
# 1. Initialise schema and HNSW index
python scripts/init_pgvector.py

# 2. Index documents
python scripts/index_pgvector.py \
  --config configs/pipeline1/smoke/smoke_pgvector_dense.yaml

# 3. Run the smoke test
export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
python -m src.pipeline1.main \
  --config configs/pipeline1/smoke/smoke_pgvector_dense.yaml
```

## 99b — Elasticsearch BM25 Smoke Test

```
configs/pipeline1/smoke/smoke_elasticsearch_bm25.yaml
```

**Purpose:** Verifies that `ElasticsearchBM25Retriever` can connect to a running Elasticsearch instance, query the pre-built BM25 index, and produce Pipeline 1 output compatible with Pipeline 2 evaluation.

**Prerequisites:**

```bash
# 1. Create and populate the BM25 index
python scripts/init_elasticsearch.py \
  --host "$ELASTICSEARCH_URL" \
  --index rag_benchmark_chunks

python scripts/index_elasticsearch.py \
  --config configs/pipeline1/smoke/smoke_elasticsearch_bm25.yaml

# 2. Run the smoke test
export ELASTICSEARCH_URL="http://localhost:9201"
python -m src.pipeline1.main \
  --config configs/pipeline1/smoke/smoke_elasticsearch_bm25.yaml
```

> These configs are **infrastructure validation** tools, not benchmark experiments. They are not part of the numbered experiment series (11–18) and their results are not included in benchmark comparisons.

---

# Benchmark Experiments

Benchmark experiments begin after all smoke tests pass.

**Current experiment series: 11–18 (orchestration experiments)**

These experiments benchmark different LLM prompting strategies and generation models using the FAISS dense retrieval backend. They share the same ERP corpus (SIVAS), chunking strategy (fixed-512), and embedding model.

| Experiment | Description |
|---|---|
| 11 | Baseline — mistral-small, default prompt |
| 12 | mistral-small, prompt V1 |
| 13 | mistral-small, prompt V2 |
| 14 | mistral-small, prompt V3 |
| 15 | mistral-small, prompt V4 |
| 16 | qwen2.5, prompt V4 |
| 17 | llama3.1, prompt V4 |
| 18 | mistral-small, prompt V5 |

**Planned future experiments** will benchmark retrieval backend differences using the same dataset and evaluation pipeline (Pipelines 2–3):

- FAISS (baseline, already covered by experiments 11–18)
- PostgreSQL + pgvector (dense, persistent)
- Elasticsearch BM25 (sparse)
- Hybrid (dense + sparse fusion)

---

# Troubleshooting

## Docker Compose file not found

**Symptom:** `docker compose up` fails with `no configuration file provided`.

**Fix:** Ensure you are in the correct directory before running Docker Compose commands.

```bash
cd infra/docker
docker compose up -d
```

## Port 9200 already in use

**Symptom:** Elasticsearch container fails to start; port conflict on `9200`.

**Fix:** The compose file maps Elasticsearch to host port `9201`, not `9200`. If another service is using `9201`, either stop it or update the host port in `docker-compose.yml` and the `ELASTICSEARCH_URL` environment variable accordingly.

## PostgreSQL connection refused

**Symptom:** `psycopg2.OperationalError: could not connect to server`.

**Fix:**
1. Verify the container is running: `docker compose ps`.
2. Verify `PGVECTOR_DSN` is exported and correct: `echo $PGVECTOR_DSN`.
3. Verify the port is accessible: `nc -zv localhost 5432`.
4. Check container logs: `docker logs rag-benchmark-postgres`.

## Elasticsearch health not green

**Symptom:** `curl http://localhost:9201/_cluster/health` returns `"status":"red"`.

**Fix:**
1. Check container logs: `docker logs rag-benchmark-elasticsearch`.
2. Allow more time for JVM warm-up (up to 60 seconds on first start).
3. Ensure the JVM heap setting (`-Xms1g -Xmx1g`) does not exceed available server RAM.
4. Verify `esdata` volume is not corrupted: `docker compose down -v` and restart (this will delete the index).

## Kibana not reachable

**Symptom:** `http://localhost:5601` times out or refuses connection.

**Fix:**
1. Verify the SSH tunnel is open: `ssh -L 5601:localhost:5601 lfgaier@schmidhuber18.imla.hs-offenburg.de`.
2. Verify Kibana is running on the server: `docker compose ps`.
3. Kibana requires Elasticsearch to be healthy before it starts — check ES health first.

## Missing PGVECTOR_DSN

**Symptom:** `KeyError: 'PGVECTOR_DSN'` or `ValueError: PGVECTOR_DSN not set`.

**Fix:** Export the variable before running the framework:

```bash
export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
```

## Missing ELASTICSEARCH_URL

**Symptom:** `KeyError: 'ELASTICSEARCH_URL'` or connection refused to Elasticsearch.

**Fix:** Export the variable before running the framework:

```bash
export ELASTICSEARCH_URL="http://localhost:9201"
```

## CUDA / PyTorch incompatibility

**Symptom:** `RuntimeError: CUDA error: no kernel image is available for execution on the device` or torch import errors.

**Background:** torch is treated as a system/environment dependency — it is **not** pinned
in `requirements.txt` or `requirements-lock.txt`. The project does not dictate which torch
version the server uses. If the server already has a CUDA-enabled torch, do not reinstall it.

**Fix:**
1. Verify GPU is recognised: `nvidia-smi`.
2. Verify PyTorch CUDA version matches the installed driver:
   `python -c "import torch; print(torch.version.cuda)"`.
3. If torch is missing or wrong for the driver, install the correct wheel manually:
   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu<version>
   ```
   See https://pytorch.org/get-started/locally/ for the correct index URL.
4. After installing the correct torch, install the project **without** letting pip
   overwrite it:
   ```bash
   python -m pip install -r requirements-lock.txt -c constraints.txt
   python -m pip install --no-deps -e .
   ```
   Do **not** run `pip install -r requirements.txt` after the GPU server already has
   the correct torch — `requirements-lock.txt` no longer pins torch so it will not
   downgrade it.

## Ollama model missing

**Symptom:** `ollama: model 'mistral-small' not found` or HTTP 404 from Ollama API.

**Fix:** Pull the required model:

```bash
ollama pull mistral-small
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b-instruct
```

Verify installed models: `ollama list`.

---

# Appendix — Useful Commands

## Docker

```bash
# Show running containers and health status
docker compose ps

# Stream logs from a specific service
docker logs -f rag-benchmark-elasticsearch

# Open a shell inside a container
docker exec -it rag-benchmark-postgres bash

# Restart a single service
docker compose restart kibana
```

## PostgreSQL

```bash
# Open interactive psql session
docker exec -it rag-benchmark-postgres psql -U rag -d rag

# List tables in the rag schema
\dt rag.*

# Count embeddings
SELECT COUNT(*) FROM rag.chunk_embeddings;

# Exit
\q
```

## Elasticsearch

```bash
# Cluster health
curl http://localhost:9201/_cluster/health?pretty

# List indices with document counts
curl http://localhost:9201/_cat/indices?v

# Count documents in benchmark index
curl http://localhost:9201/rag_benchmark_chunks/_count

# Sample document
curl "http://localhost:9201/rag_benchmark_chunks/_search?size=1&pretty"
```

## Ollama

```bash
# List installed models
ollama list

# Pull a model
ollama pull mistral-small

# Test a model (interactive)
ollama run mistral-small
```

## GPU

```bash
# Check GPU availability and VRAM usage
nvidia-smi

# Monitor GPU usage in real time
watch -n 1 nvidia-smi
```

## Git

```bash
# Pull latest changes from GitHub
git pull

# Check working tree status
git status

# Check branch and remote tracking
git branch -vv
```
