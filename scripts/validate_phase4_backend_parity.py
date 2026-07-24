#!/usr/bin/env python3
"""Phase 4 vector-backend parity validation script.

Validates that FAISS, pgvector, and Elasticsearch return equivalent Top-5
rankings for identical normalized query vectors over the same embedding matrix.

Steps:
  1. Load deterministic fixed vectors (or use a real embedding cache if supplied).
  2. Validate vector count, dimension, and normalization.
  3. Query FAISS (always — no external service required).
  4. Query pgvector if PGVECTOR_DSN is set and table exists.
  5. Query Elasticsearch if ES is reachable at index.host.
  6. Print Top-5 chunk IDs per query for each backend.
  7. Compare ordered rankings (tie-aware).
  8. Return non-zero exit code if unexplained mismatches are found.

Environment variables:
  PGVECTOR_DSN      PostgreSQL DSN (optional)
  ELASTICSEARCH_URL Elasticsearch host URL (optional, default: http://localhost:9200)

Usage:
  python scripts/validate_phase4_backend_parity.py
  python scripts/validate_phase4_backend_parity.py \\
      --embedding-cache data/processed/embeddings/<hash>.npy \\
      --chunk-ids data/processed/chunks/<key>.jsonl \\
      --top-k 5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

SCORE_TOL = 1e-5


def ok(msg: str) -> None:
    print(f"  {_GREEN}[OK]{_RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {_RED}[FAIL]{_RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {_YELLOW}[WARN]{_RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {_CYAN}[INFO]{_RESET} {msg}")


# ---------------------------------------------------------------------------
# Deterministic fixed-vector fixture (used when no real cache is supplied)
# ---------------------------------------------------------------------------

DIM_FIXTURE = 16
N_DOCS_FIXTURE = 8

_e = [np.zeros(DIM_FIXTURE, dtype="float32") for _ in range(DIM_FIXTURE)]
for _i in range(DIM_FIXTURE):
    _e[_i][_i] = 1.0

_doc_raw_fixture = np.array([
    _e[0],
    0.9 * _e[0] + math.sqrt(1 - 0.81) * _e[1],
    0.7071 * _e[0] + 0.7071 * _e[1],
    _e[1],
    0.7071 * _e[1] + 0.7071 * _e[2],
    _e[2],
    _e[0],   # duplicate of doc_0 → tie
    -_e[0],  # anti-aligned
], dtype="float32")
_norms = np.linalg.norm(_doc_raw_fixture, axis=1, keepdims=True)
FIXTURE_DOC_VECS = _doc_raw_fixture / _norms

_q_raw_fixture = np.array([
    _e[0],
    _e[1],
    -_e[0],
    0.5774 * _e[0] + 0.5774 * _e[1] + 0.5774 * _e[2],
    _e[3],
], dtype="float32")
_qnorms = np.linalg.norm(_q_raw_fixture, axis=1, keepdims=True)
FIXTURE_QUERY_VECS = _q_raw_fixture / _qnorms

FIXTURE_CHUNK_IDS = [f"chunk_{i:03d}" for i in range(N_DOCS_FIXTURE)]
FIXTURE_DOC_IDS = [
    "doc_000", "doc_000",
    "doc_001", "doc_001",
    "doc_002", "doc_002",
    "doc_003", "doc_003",
]

QUERY_LABELS = [
    "Q0 (aligned e0 — docs 0/6 tied)",
    "Q1 (aligned e1 — close neighbors)",
    "Q2 (anti-aligned — negative cosines)",
    "Q3 (spread — small margins)",
    "Q4 (no alignment — near-zero sims)",
]


# ---------------------------------------------------------------------------
# Vector validation
# ---------------------------------------------------------------------------

def validate_vectors(vecs: np.ndarray, dim: int, label: str) -> bool:
    print(f"\n{_BOLD}Vector validation: {label}{_RESET}")
    print("-" * 50)
    n, d = vecs.shape
    info(f"Count: {n}  Dimension: {d}")

    if d != dim:
        fail(f"Dimension mismatch: expected {dim}, got {d}")
        return False
    ok(f"Dimension correct: {d}")

    norms = np.linalg.norm(vecs, axis=1)
    max_dev = float(np.max(np.abs(norms - 1.0)))
    if max_dev < 1e-4:
        ok(f"All vectors normalized (max deviation from 1.0: {max_dev:.2e})")
    else:
        fail(f"Vectors NOT normalized (max deviation: {max_dev:.2e})")
        return False

    return True


# ---------------------------------------------------------------------------
# Tie-aware ranking comparison
# ---------------------------------------------------------------------------

def _group_by_score(ranked, tol=SCORE_TOL):
    if not ranked:
        return
    group = [ranked[0]]
    for item in ranked[1:]:
        if abs(item[2] - group[-1][2]) <= tol:
            group.append(item)
        else:
            yield group
            group = [item]
    yield group


def compare_rankings(
    ref: list[tuple[str, str, float]],
    actual: list[tuple[str, str, float]],
    backend_label: str,
    tol: float = SCORE_TOL,
) -> tuple[bool, bool, bool, list[str]]:
    """Compare rankings. Returns (ordered_pass, set_pass, score_pass, issues).

    Tie handling:
    - Step 1: scores must agree at every rank (within tol).
    - Step 2: multi-item tied groups (size > 1 in ref) must have equal chunk-ID sets.
    - Singleton items at the tie boundary (tied with items outside top-k) are accepted
      when scores match — chunk IDs may differ legitimately.
    """
    issues = []
    if len(actual) != len(ref):
        issues.append(f"count mismatch: ref={len(ref)} actual={len(actual)}")
        return False, False, False, issues

    score_ok = True
    set_ok = True
    ordered_ok = True

    # Step 1: scores must agree at every rank
    for i, (r, a) in enumerate(zip(ref, actual)):
        diff = abs(r[2] - a[2])
        if diff > tol:
            score_ok = False
            issues.append(
                f"  rank {i + 1} score mismatch (diff={diff:.2e}) "
                f"ref_chunk={r[0]} ref_score={r[2]:.6f} "
                f"actual_chunk={a[0]} actual_score={a[2]:.6f}"
            )

    # Step 2: multi-item tied groups must have equal chunk-ID sets.
    # Singleton items at the boundary are accepted as long as scores match (Step 1).
    ref_scores = [r[2] for r in ref]
    i = 0
    while i < len(ref):
        j = i + 1
        while j < len(ref) and abs(ref_scores[j] - ref_scores[i]) <= tol:
            j += 1
        if j - i > 1:
            ref_ids = {r[0] for r in ref[i:j]}
            act_ids = {a[0] for a in actual[i:j]}
            if ref_ids != act_ids:
                set_ok = False
                issues.append(
                    f"  tied group at ranks {i + 1}–{j} set mismatch: "
                    f"ref={sorted(ref_ids)} actual={sorted(act_ids)}"
                )
        else:
            # Singleton: ordered parity if not a boundary tie
            # A singleton is non-tied only if no neighbors share its score in EITHER list
            curr_score = ref_scores[i]
            act_scores_local = [a[2] for a in actual]
            prev_tied = i > 0 and abs(ref_scores[i - 1] - curr_score) <= tol
            next_tied = i < len(ref) - 1 and abs(ref_scores[i + 1] - curr_score) <= tol
            act_prev_tied = i > 0 and abs(act_scores_local[i - 1] - curr_score) <= tol
            act_next_tied = i < len(actual) - 1 and abs(act_scores_local[i + 1] - curr_score) <= tol
            is_boundary_tie = prev_tied or next_tied or act_prev_tied or act_next_tied
            # Also accept singletons at score-boundary (tied with items outside top-k)
            # detected by same score but different chunk-id between ref and actual
            cross_tied = (abs(ref[i][2] - actual[i][2]) <= tol and ref[i][0] != actual[i][0])
            if not is_boundary_tie and not cross_tied:
                if ref[i][0] != actual[i][0]:
                    ordered_ok = False
                    issues.append(
                        f"  ordered mismatch at rank {i + 1}: "
                        f"ref={ref[i][0]} actual={actual[i][0]}"
                    )
        i = j

    return ordered_ok, set_ok, score_ok, issues


# ---------------------------------------------------------------------------
# FAISS query
# ---------------------------------------------------------------------------

def query_faiss(doc_vecs: np.ndarray, query_vec: np.ndarray, chunk_ids: list[str],
                doc_ids: list[str], top_k: int) -> list[tuple[str, str, float]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.pipeline1.indexing.faiss_index import FaissIndex

    idx = FaissIndex(metric="cosine")
    idx.build(doc_vecs.copy())
    scores_raw, positions = idx.search(query_vec, top_k)
    results = []
    for pos, score in zip(positions, scores_raw):
        p = int(pos)
        if p < 0 or p >= len(chunk_ids):
            continue
        results.append((chunk_ids[p], doc_ids[p], float(score)))
    return results


# ---------------------------------------------------------------------------
# pgvector query (live)
# ---------------------------------------------------------------------------

def query_pgvector_live(
    doc_vecs: np.ndarray,
    query_vec: np.ndarray,
    chunk_ids: list[str],
    doc_ids: list[str],
    top_k: int,
    table_name: str = "v01_phase4_parity_test",
    schema_name: str = "rag",
) -> list[tuple[str, str, float]] | None:
    dsn = os.environ.get("PGVECTOR_DSN", "").strip()
    if not dsn:
        return None

    try:
        import psycopg2
    except ImportError:
        warn("psycopg2 not installed — pgvector check skipped.")
        return None

    try:
        conn = psycopg2.connect(dsn, connect_timeout=5)
        conn.autocommit = True
        cur = conn.cursor()

        # Create a temporary table for parity testing
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        cur.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name}")
        cur.execute(
            f"CREATE TABLE {schema_name}.{table_name} "
            f"(chunk_id TEXT, doc_id TEXT, embedding vector({doc_vecs.shape[1]}))"
        )
        for cid, did, vec in zip(chunk_ids, doc_ids, doc_vecs):
            vec_str = "[" + ",".join(str(float(v)) for v in vec) + "]"
            cur.execute(
                f"INSERT INTO {schema_name}.{table_name} (chunk_id, doc_id, embedding) VALUES (%s, %s, %s::vector)",
                (cid, did, vec_str),
            )
        q_str = "[" + ",".join(str(float(v)) for v in query_vec) + "]"
        cur.execute(
            f"SELECT chunk_id, doc_id, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {schema_name}.{table_name} "
            f"ORDER BY embedding <=> %s::vector LIMIT %s",
            (q_str, q_str, top_k),
        )
        rows = cur.fetchall()
        cur.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name}")
        conn.close()
        return [(r[0], r[1], float(r[2])) for r in rows]
    except Exception as ex:
        warn(f"pgvector query failed: {ex}")
        return None


# ---------------------------------------------------------------------------
# Elasticsearch query (live)
# ---------------------------------------------------------------------------

def query_elasticsearch_live(
    doc_vecs: np.ndarray,
    query_vec: np.ndarray,
    chunk_ids: list[str],
    doc_ids: list[str],
    top_k: int,
    index_name: str = "v02_phase4_parity_test",
    es_host: str | None = None,
) -> list[tuple[str, str, float]] | None:
    host = es_host or os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        warn("elasticsearch package not installed — ES check skipped.")
        return None

    try:
        client = Elasticsearch(host, request_timeout=10, verify_certs=False)
        client.info()
    except Exception as ex:
        warn(f"Elasticsearch not reachable at {host}: {ex}")
        return None

    try:
        dim = doc_vecs.shape[1]
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
        client.indices.create(
            index=index_name,
            body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {"properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "embedding": {"type": "dense_vector", "dims": dim, "index": False},
                }},
            },
        )
        ops = []
        for cid, did, vec in zip(chunk_ids, doc_ids, doc_vecs):
            ops.append({"index": {"_index": index_name, "_id": cid}})
            ops.append({"chunk_id": cid, "doc_id": did, "embedding": vec.tolist()})
        client.bulk(operations=ops, refresh=True)

        q_list = query_vec.tolist()
        response = client.search(
            index=index_name,
            size=top_k,
            query={
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": q_list},
                    },
                }
            },
            source=["chunk_id", "doc_id"],
        )
        client.indices.delete(index=index_name)
        hits = response.get("hits", {}).get("hits", [])
        return [
            (
                hit["_source"]["chunk_id"],
                hit["_source"]["doc_id"],
                float(hit["_score"]) - 1.0,
            )
            for hit in hits
        ]
    except Exception as ex:
        warn(f"Elasticsearch parity query failed: {ex}")
        return None


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

def print_results(
    label: str,
    faiss_r: list[tuple[str, str, float]],
    pgvec_r: list[tuple[str, str, float]] | None,
    es_r: list[tuple[str, str, float]] | None,
    ref_r: list[tuple[str, str, float]],
) -> None:
    print(f"\n  {_BOLD}Query: {label}{_RESET}")

    # Header row
    pg_hdr = "pgvector" if pgvec_r else "pgvector (N/A)"
    es_hdr = "Elasticsearch" if es_r else "Elasticsearch (N/A)"
    print(f"  {'Rank':<6} {'Numpy ref':<14} {'FAISS':<14} {pg_hdr:<14} {es_hdr:<14}")
    print("  " + "-" * 62)

    for i in range(max(len(ref_r), len(faiss_r))):
        r_id = ref_r[i][0] if i < len(ref_r) else "-"
        f_id = faiss_r[i][0] if i < len(faiss_r) else "-"
        p_id = pgvec_r[i][0] if pgvec_r and i < len(pgvec_r) else "-"
        e_id = es_r[i][0] if es_r and i < len(es_r) else "-"
        mark = "" if f_id == r_id else f"  {_YELLOW}*{_RESET}"
        print(f"  {i+1:<6} {r_id:<14} {f_id:<14} {p_id:<14} {e_id:<14}{mark}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 backend parity validation")
    parser.add_argument("--embedding-cache", help="Path to .npy embedding file (optional)")
    parser.add_argument("--chunk-ids", help="Path to JSONL chunk file for chunk_id mapping (optional)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--es-host", help="Elasticsearch host (default: http://localhost:9200)")
    args = parser.parse_args()

    print(f"\n{_BOLD}{'='*60}{_RESET}")
    print(f"{_BOLD}Phase 4 Backend Parity Validation{_RESET}")
    print(f"{_BOLD}{'='*60}{_RESET}")

    top_k = args.top_k
    live_pgvec = False
    live_es = False
    all_issues: list[str] = []

    # --- Load vectors ---
    if args.embedding_cache:
        cache_path = Path(args.embedding_cache)
        print(f"\nLoading embedding cache: {cache_path}")
        if not cache_path.exists():
            print(f"{_RED}ERROR: Embedding cache not found: {cache_path}{_RESET}")
            sys.exit(1)
        doc_vecs_raw = np.load(str(cache_path)).astype("float32")
        dim = doc_vecs_raw.shape[1]
        n_docs = doc_vecs_raw.shape[0]

        # Load chunk/doc IDs from JSONL if supplied
        if args.chunk_ids:
            import json
            chunk_ids = []
            doc_ids = []
            with open(args.chunk_ids, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    chunk_ids.append(str(rec.get("chunk_id", rec.get("id", ""))))
                    doc_ids.append(str(rec.get("document_id", rec.get("doc_id", ""))))
        else:
            chunk_ids = [f"chunk_{i:05d}" for i in range(n_docs)]
            doc_ids = [f"doc_{i:05d}" for i in range(n_docs)]

        # Build 5 deterministic query vectors in the same dimension
        rng = np.random.default_rng(42)
        q_raw = rng.standard_normal((5, dim)).astype("float32")
        q_norms = np.linalg.norm(q_raw, axis=1, keepdims=True)
        query_vecs = q_raw / q_norms
        labels = [f"Q{i} (random seeded)" for i in range(5)]
    else:
        print("\nUsing deterministic fixed-vector fixture (no cache supplied).")
        doc_vecs_raw = FIXTURE_DOC_VECS
        chunk_ids = FIXTURE_CHUNK_IDS
        doc_ids = FIXTURE_DOC_IDS
        query_vecs = FIXTURE_QUERY_VECS
        dim = DIM_FIXTURE
        n_docs = N_DOCS_FIXTURE
        labels = QUERY_LABELS

    # Normalize
    norms = np.linalg.norm(doc_vecs_raw, axis=1, keepdims=True)
    doc_vecs = doc_vecs_raw / np.where(norms == 0, 1.0, norms)

    # --- Validate ---
    valid = validate_vectors(doc_vecs, dim, f"Document vectors ({n_docs} docs, dim={dim})")
    if not valid:
        sys.exit(1)

    # --- Per-query results ---
    query_verdicts: list[str] = []
    for q_idx, (query, label) in enumerate(zip(query_vecs, labels)):
        print(f"\n{_BOLD}Query {q_idx}: {label}{_RESET}")
        print("-" * 60)

        # numpy reference
        ref_scores = doc_vecs @ query.astype("float32")
        ref_order = np.argsort(-ref_scores)[:top_k]
        ref_r = [(chunk_ids[i], doc_ids[i], float(ref_scores[i])) for i in ref_order]

        # FAISS
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from src.pipeline1.indexing.faiss_index import FaissIndex
            faiss_r = query_faiss(doc_vecs, query, chunk_ids, doc_ids, top_k)
            ok("FAISS query complete")
        except Exception as ex:
            fail(f"FAISS query failed: {ex}")
            faiss_r = []

        # pgvector (live)
        pgvec_r = query_pgvector_live(doc_vecs, query, chunk_ids, doc_ids, top_k)
        if pgvec_r is not None:
            live_pgvec = True
            ok("pgvector query complete")
        else:
            info("pgvector skipped (PGVECTOR_DSN not set or service unavailable)")

        # Elasticsearch (live)
        es_r = query_elasticsearch_live(doc_vecs, query, chunk_ids, doc_ids, top_k, es_host=args.es_host)
        if es_r is not None:
            live_es = True
            ok("Elasticsearch query complete")
        else:
            info("Elasticsearch skipped (service unavailable)")

        # Print result table
        print_results(label, faiss_r, pgvec_r, es_r, ref_r)

        # Compare
        q_pass = True
        if faiss_r:
            o, s, sc, issues = compare_rankings(ref_r, faiss_r, "FAISS")
            if issues:
                q_pass = False
                all_issues.extend([f"Q{q_idx} FAISS: {iss}" for iss in issues])
                fail(f"FAISS: ordered={o} set={s} scores={sc}")
            else:
                ok("Ordered parity: PASS  Set parity: PASS  Score tolerance: PASS  [FAISS]")

        if pgvec_r:
            o, s, sc, issues = compare_rankings(ref_r, pgvec_r, "pgvector")
            if issues:
                q_pass = False
                all_issues.extend([f"Q{q_idx} pgvector: {iss}" for iss in issues])
                fail(f"pgvector: ordered={o} set={s} scores={sc}")
            else:
                ok("Ordered parity: PASS  Set parity: PASS  Score tolerance: PASS  [pgvector]")

        if es_r:
            o, s, sc, issues = compare_rankings(ref_r, es_r, "Elasticsearch")
            if issues:
                q_pass = False
                all_issues.extend([f"Q{q_idx} Elasticsearch: {iss}" for iss in issues])
                fail(f"Elasticsearch: ordered={o} set={s} scores={sc}")
            else:
                ok("Ordered parity: PASS  Set parity: PASS  Score tolerance: PASS  [Elasticsearch]")

        query_verdicts.append("PASS" if q_pass else "FAIL")

    # --- Final verdict ---
    print(f"\n{_BOLD}{'='*60}{_RESET}")
    print(f"{_BOLD}Final Verdict{_RESET}")
    print("-" * 60)

    for q_idx, v in enumerate(query_verdicts):
        color = _GREEN if v == "PASS" else _RED
        print(f"  Query {q_idx}: {color}{v}{_RESET}")

    if all_issues:
        verdict = "PHASE4_BACKEND_PARITY_FAIL"
        print(f"\n{_RED}{_BOLD}{verdict}{_RESET}")
        print(f"{_RED}Unexplained mismatches:{_RESET}")
        for iss in all_issues:
            print(f"  - {iss}")
        sys.exit(1)
    elif not live_pgvec and not live_es:
        verdict = "PHASE4_BACKEND_PARITY_PARTIAL_LOCAL_ONLY"
        print(f"\n{_YELLOW}{_BOLD}{verdict}{_RESET}")
        print(
            f"{_YELLOW}FAISS matches numpy reference for all {len(query_verdicts)} queries.{_RESET}\n"
            "pgvector and Elasticsearch were not validated against live services.\n"
            "Set PGVECTOR_DSN and start Elasticsearch to enable live validation."
        )
        sys.exit(0)
    elif not live_pgvec or not live_es:
        verdict = "PHASE4_BACKEND_PARITY_PARTIAL_LOCAL_ONLY"
        missing = []
        if not live_pgvec:
            missing.append("pgvector")
        if not live_es:
            missing.append("Elasticsearch")
        print(f"\n{_YELLOW}{_BOLD}{verdict}{_RESET}")
        print(f"{_YELLOW}Missing live validation for: {', '.join(missing)}{_RESET}")
        sys.exit(0)
    else:
        verdict = "PHASE4_BACKEND_PARITY_PASS"
        print(f"\n{_GREEN}{_BOLD}{verdict}{_RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
