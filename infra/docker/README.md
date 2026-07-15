# Docker Infrastructure

Manages two external services used by the pgvector and Elasticsearch retrieval backends.

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | Dense vector index (pgvector backend) |
| `elasticsearch` | `elasticsearch:8.15.1` | 9200 | BM25 lexical index + optional dense backend |

The `pipeline1` application container is also defined here for Docker-based runs, but is
not required for local development — you can run the pipeline directly with Python while
only the two service containers are running.

---

## Quick start

```bash
cd infra/docker

# Start background services only (recommended for local development):
docker compose up -d postgres elasticsearch

# Check status:
docker compose ps
```

---

## Health verification

### PostgreSQL + pgvector

```bash
# Connectivity
docker exec rag-benchmark-postgres pg_isready -U rag -d rag

# Ensure the vector extension is available:
docker exec rag-benchmark-postgres psql -U rag -d rag \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Confirm extension is installed:
docker exec rag-benchmark-postgres psql -U rag -d rag -c "\dx"
# Expected: a row for "vector" in the output.
```

### Elasticsearch

```bash
curl -s http://localhost:9200/_cluster/health | python3 -m json.tool
# Expected: "status" is "green" or "yellow" (single-node is always yellow).
# Red means the cluster has not finished bootstrapping — wait 30 s and retry.
```

---

## Environment variables

Copy `.env.example` to `.env` and adjust if your credentials differ:

```bash
cp .env.example .env
```

Then export before running scripts:

```bash
export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
export ELASTICSEARCH_URL="http://localhost:9200"
```

`.env` is git-ignored; `.env.example` is committed as a template.

---

## Schema and index initialization

Run once per fresh database / index (idempotent — safe to re-run):

```bash
# From project root:
python scripts/init_pgvector.py
python scripts/init_elasticsearch.py --host http://localhost:9200 --index rag_benchmark_chunks
```

Then index documents:

```bash
python scripts/index_pgvector.py      --config configs/pipeline1/smoke/smoke_pgvector_dense.yaml
python scripts/index_elasticsearch.py --config configs/pipeline1/smoke/smoke_elasticsearch_bm25.yaml
```

For a quick service connectivity check without touching data:

```bash
python scripts/check_backend_services.py
```

---

## Service details

### PostgreSQL

- **Image**: `pgvector/pgvector:pg16` — PostgreSQL 16 with the `pgvector` extension pre-installed.
- **Credentials**: user=`rag`, password=`rag`, database=`rag`.
- **Data volume**: `pgdata` (persists across `docker compose down`; removed by `docker compose down -v`).
- **Health check**: `pg_isready -U rag -d rag` every 5 s, up to 10 retries.

### Elasticsearch

- **Image**: `docker.elastic.co/elasticsearch/elasticsearch:8.15.1` — matches the project's Python client `elasticsearch==8.15.1`.
- **Security**: disabled (`xpack.security.enabled=false`) — suitable for local/CI use only.
- **Memory**: 1 GB heap (`ES_JAVA_OPTS=-Xms1g -Xmx1g`). Reduce to `-Xms512m -Xmx512m` on machines with < 4 GB RAM.
- **Data volume**: `esdata` (persists across `docker compose down`).
- **Health check**: HTTP GET `/_cluster/health`, every 10 s, up to 20 retries.

---

## Teardown

```bash
# Stop containers, keep volumes (data survives):
docker compose down

# Stop containers AND delete all data volumes:
docker compose down -v
```
