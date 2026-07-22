# SIVAS Chunk-Level Retrieval Ground Truth — C02_sentence1024_overlap400

This folder contains the finalized artifacts required to integrate chunk-level
ground truth into the RAG benchmarking framework for the **C02_sentence1024_overlap400**
chunking configuration.

## Runtime File

Pipeline 2 should load:

```
gold_chunk_annotations_C02_sentence1024_overlap400.jsonl
```

Each record contains one question's `gold_primary_chunk_ids` and
`gold_relevant_chunk_ids` for this specific chunk configuration.

## Chunk Configuration

| Parameter       | Value                  |
|-----------------|------------------------|
| Config ID       | `C02_sentence1024_overlap400`          |
| Strategy        | sentence       |
| Chunk size      | 1024 tokens |
| Chunk overlap   | 400 tokens |
| Tokenizer       | cl100k_base      |
| Mapping policy  | any_overlap               |

> **Warning:** Chunk IDs in retrieval output must exactly match the chunk IDs
> in the ground-truth file. A different chunking configuration requires its
> own derived ground-truth file.

## Dataset Statistics

| Metric                             | Value    |
|------------------------------------|----------|
| Questions                          | 96  |
| Gold evidence spans (total)        | 799 |
| Mapped evidence spans              | 799      |
| Unmapped evidence spans            | 0    |
| Total chunks                       | 750 |
| Documents with chunks              | 65    |
| Evidence mapped to one chunk       | 108     |
| Evidence mapped to multiple chunks | 691   |
| Unique relevant chunks             | 493 |
| Unique primary chunks              | 340  |
| Mean relevant chunks per evidence  | 2.1977           |
| Max relevant chunks per evidence   | 6            |
| Validation status                  | **PASS**  |

## Files in This Directory

| File | Purpose |
|------|---------|
| `gold_chunk_annotations_C02_sentence1024_overlap400.jsonl` | **Primary runtime file.** One record per question. |
| `gold_span_chunk_mappings_C02_sentence1024_overlap400.jsonl` | Per-span traceability artifact. |
| `chunk_mapping_summary_C02_sentence1024_overlap400.json` | Aggregate mapping statistics. |
| `final_annotation_validation.json` | Validation report. |
| `integration_package.json` | File manifest with SHA-256 hashes. |
| `README.md` | This file. |

## Configuration Example

```yaml
retrieval_evaluation:
  chunk_level:
    enabled: true
    ground_truth_path: ground_truth/chunk_level/C02_sentence1024_overlap400/gold_chunk_annotations_C02_sentence1024_overlap400.jsonl
    chunk_config_id: C02_sentence1024_overlap400
    strict_chunk_id_matching: true
```

## Derivation

These ground-truth labels were derived **deterministically** by remapping the
canonical gold evidence spans to the `C02_sentence1024_overlap400` chunk corpus. No human
re-annotation was performed. The source spans are at:

```
chunk_level_annotation/annotations/gold_evidence_spans.jsonl
```

The remapping was produced by `chunk_level_annotation/scripts/remap_new_configs.py`
using the chunking and overlap logic from `map_05.py`.
