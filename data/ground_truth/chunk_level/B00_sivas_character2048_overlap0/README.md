# B00_sivas_character2048_overlap0

**Valid only for: `B00_sivas_character2048_overlap0`**

This ground-truth package is valid exclusively for the B00 experiment that uses
the SIVAS character-based chunking strategy. Using it with any other chunking
configuration will produce incorrect metrics.

> **Warning:** The archived directory `_invalid_archived/B00_sentence2048_overlap0/`
> was generated with the wrong strategy (sentence/token-based, 2048 tokens) and
> must NOT be used. It has been moved out of the active namespace.

## Chunk Configuration

| Parameter          | Value |
|--------------------|-------|
| Config ID          | `B00_sivas_character2048_overlap0` |
| Strategy           | `sivas_character` |
| Maximum chunk size | 2048 characters |
| Overlap            | 0 |
| Size unit          | characters (NOT tokens) |
| Tokenizer          | none |

## Partner-Defined Boundary Regex

```
(?<=[.!?;:])\s+|\n\n|\n(?=#{1,6}\s)|\n(?=-\s)
```

Splits on: sentence-final punctuation followed by whitespace, double newlines,
markdown heading newlines, and list-item newlines.

## Evidence Spans

| Metric               | Value |
|----------------------|-------|
| Questions            | 96 |
| Gold evidence spans  | 799 |
| Mapped spans         | 799 |
| Unmapped spans       | 0 |
| Total chunks         | 854 |
| Unique gold-relevant | 508 |

## Production Equivalence

Result: **PASS — exact match with production ERP11 ground truth**

Verified: generated `gold_relevant_chunk_ids` match production ERP11
ground truth for all 96 questions.

## Mapping Policy

Policy: `any_overlap` — a chunk is gold-relevant if it overlaps the evidence
span by at least 1 character.

The mapping is deterministic: documents and spans are sorted before processing,
and the primary chunk is selected by maximum overlap (ties broken by chunk index).

## Supported Metrics (Pipeline 2)

- Chunk Hit@k
- Chunk Recall@k
- Chunk MRR
- Chunk nDCG@k

## Files

| File | Purpose |
|------|---------|
| `gold_chunk_annotations_B00_sivas_character2048_overlap0.jsonl` | **Primary runtime file** — one record per question |
| `gold_span_chunk_mappings_B00_sivas_character2048_overlap0.jsonl` | Span-level traceability |
| `chunk_mapping_summary_B00_sivas_character2048_overlap0.json` | Aggregate statistics |
| `final_annotation_validation.json` | Validation report |
| `integration_package.json` | File manifest with SHA-256 hashes |
| `README.md` | This file |

## Pipeline 2 Integration

```yaml
retrieval_evaluation:
  chunk_level:
    enabled: true
    ground_truth_path: ground_truth/chunk_level/B00_sivas_character2048_overlap0/gold_chunk_annotations_B00_sivas_character2048_overlap0.jsonl
    chunk_config_id: B00_sivas_character2048_overlap0
    strict_chunk_id_matching: true
```
