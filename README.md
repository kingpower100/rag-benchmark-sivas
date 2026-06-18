
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

## Active Pipeline 1 Config

- `configs/pipeline1/experiments/11_sivas_fixed512_faiss_dense_qwen25.yaml`

Baseline run command:

```bash
python -m src.pipeline1.main --config configs/pipeline1/experiments/11_sivas_fixed512_faiss_dense_qwen25.yaml
```

Expected Pipeline 1 output:

- `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/results.jsonl`
- `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/run_manifest.json`
- `data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/logs.txt`

Pipeline 1 requires the configured local generation service and embedding dependencies at runtime. Cleanup and static checks must not start services, load models, build indexes, or execute retrieval/generation.

## Pipeline 2 Evaluation

Pipeline 2 defaults are SIVAS-first:

- `configs/pipeline2/base_eval.yaml`
- `qa_path: data/raw/qa_ground_truth_fixed.jsonl`
- `pipeline1_results_path: data/runs/pipeline1/11_sivas_fixed512_faiss_dense_qwen25/results.jsonl`

Run evaluation only after Pipeline 1 has produced `results.jsonl`:

```bash
python -m src.pipeline2.main --config configs/pipeline2/base_eval.yaml
```

Expected Pipeline 2 output:

- `data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/per_question.jsonl`
- `data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/summary_by_experiment.csv`
- `data/eval/runs/pipeline2/11_sivas_fixed512_faiss_dense_qwen25_eval/eval_manifest.json`

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


