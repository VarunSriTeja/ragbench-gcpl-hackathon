# Option B — RAG System with Retrieval Benchmarking

This repository is a working prototype for **Option B** from the GCPL BT AI Hackathon brief.

## What this project does

- Ingests a curated dataset of RAG engineering notes
- Chunks the documents using **two strategies**:
  - `fixed`
  - `markdown`
- Generates embeddings for dense retrieval
- Supports **dense** and optional **hybrid** retrieval
- Supports optional **cross-encoder reranking**
- Produces grounded answers from retrieved context
- Benchmarks retrieval quality on **10 evaluation queries**

## Dataset

The dataset is a curated, compact knowledge base built from public material about:

- RAG fundamentals
- chunking strategies
- hybrid retrieval
- vector search metrics
- reranking
- evaluation design

Source notes live under [data/raw](data/raw).

## Project layout

- [src/ragbench](src/ragbench) — implementation
- [data/raw](data/raw) — curated documents
- [data/eval/questions.json](data/eval/questions.json) — evaluation set
- [docs/architecture.md](docs/architecture.md) — architecture and design notes

## Quick start

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the semantic benchmark

```bash
PYTHONPATH=src python -m ragbench evaluate --backend sentence-transformers --mode hybrid --reranker auto
```

This compares the default chunking configurations and prints aggregate metrics such as:

- `precision@k`
- `recall@k`
- `mrr`
- answer grounding hit rate

If model downloads are blocked, fall back to:

```bash
PYTHONPATH=src python -m ragbench evaluate --backend tfidf --mode dense
```

### 3. Ask a grounded question

```bash
PYTHONPATH=src python -m ragbench ask "Why is chunking important in RAG?" --backend sentence-transformers --mode hybrid --reranker auto
```

### 4. Optional: enable OpenAI answer generation

If you want an LLM-generated answer instead of the local extractive fallback:

```bash
export OPENAI_API_KEY=...
export RAGBENCH_USE_OPENAI=1
PYTHONPATH=src python -m ragbench ask "What does hybrid search combine?"
```

## Evaluation scope

Mandatory brief requirements covered:

- ingest and chunk documents
- generate embeddings
- implement vector search retrieval
- generate answers grounded in retrieved context
- benchmark retrieval performance
- test **two chunking strategies**
- include **10 test queries with expected answers**

Bonus coverage:

- hybrid retrieval
- reranking
- local extractive fallback when no LLM API is present

## Recommended demo flow

1. Explain the dataset and why it was curated.
2. Show the two chunking strategies.
3. Run `evaluate` with `sentence-transformers`, hybrid retrieval, and reranking.
4. Run `ask` with a few questions.
5. Highlight hallucination controls, trade-offs, and next steps.

## Submission materials

- Architecture notes: [docs/architecture.md](/architecture.md)
- LaTeX source: [docs/submission_ready.tex](/submission_ready.tex)
- Final PDF: [artifacts/submission_ready_from_latex.pdf](artifacts/submission_ready_from_latex.pdf)
- Latest benchmark JSON: [artifacts/eval_results_semantic.json](artifacts/eval_results_semantic.json)
