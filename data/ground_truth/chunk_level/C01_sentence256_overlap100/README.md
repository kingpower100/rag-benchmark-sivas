# SIVAS Chunk-Level Retrieval Ground Truth — C01_sentence256_overlap100

This folder contains the finalized artifacts required to integrate chunk-level
ground truth into the RAG benchmarking framework for the **C01_sentence256_overlap100**
chunking configuration.

## Runtime File

Pipeline 2 should load:

```
gold_chunk_annotations_C01_sentence256_overlap100.jsonl
```

Each record contains one question's `gold_primary_chunk_ids` and
`gold_relevant_chunk_ids` for this specific chunk configuration.

## Chunk Configuration

| Parameter       | Value                  |
|-----------------|------------------------|
| Config ID       | `C01_sentence256_overlap100`          |
| Strategy        | sentence       |
| Chunk size      | 256 tokens |
| Chunk overlap   | 100 tokens |
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
| Total chunks                       | 3139 |
| Documents with chunks              | 65    |
| Evidence mapped to one chunk       | 27     |
| Evidence mapped to multiple chunks | 772   |
| Unique relevant chunks             | 1537 |
| Unique primary chunks              | 381  |
| Mean relevant chunks per evidence  | 4.7322           |
| Max relevant chunks per evidence   | 10            |
| Validation status                  | **PASS**  |

## Files in This Directory

| File | Purpose |
|------|---------|
| `gold_chunk_annotations_C01_sentence256_overlap100.jsonl` | **Primary runtime file.** One record per question. |
| `gold_span_chunk_mappings_C01_sentence256_overlap100.jsonl` | Per-span traceability artifact. |
| `chunk_mapping_summary_C01_sentence256_overlap100.json` | Aggregate mapping statistics. |
| `final_annotation_validation.json` | Validation report. |
| `integration_package.json` | File manifest with SHA-256 hashes. |
| `README.md` | This file. |

## Configuration Example

```yaml
retrieval_evaluation:
  chunk_level:
    enabled: true
    ground_truth_path: ground_truth/chunk_level/C01_sentence256_overlap100/gold_chunk_annotations_C01_sentence256_overlap100.jsonl
    chunk_config_id: C01_sentence256_overlap100
    strict_chunk_id_matching: true
```

## Derivation

These ground-truth labels were derived **deterministically** by remapping the
canonical gold evidence spans to the `C01_sentence256_overlap100` chunk corpus. No human
re-annotation was performed. The source spans are at:

```
chunk_level_annotation/annotations/gold_evidence_spans.jsonl
```

The remapping was produced by `chunk_level_annotation/scripts/remap_new_configs.py`
using the chunking and overlap logic from `map_05.py`.
