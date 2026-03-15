from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from ragbench.chunking import chunk_documents
from ragbench.dataset import load_documents, load_questions
from ragbench.generation import GroundedAnswerGenerator
from ragbench.retrieval import SearchEngine
from ragbench.types import EvalRecord


def _dedupe_document_ids(ids: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def evaluate_configuration(
    chunking_strategy: str,
    retrieval_mode: str = "dense",
    embedding_backend: str = "auto",
    reranker_backend: str = "none",
    top_k: int = 5,
) -> dict[str, object]:
    documents = load_documents()
    questions = load_questions()
    chunks = chunk_documents(documents, strategy=chunking_strategy)
    engine = SearchEngine.build(
        chunks,
        embedding_backend=embedding_backend,
        enable_hybrid=(retrieval_mode == "hybrid"),
        reranker_backend=reranker_backend,
    )
    generator = GroundedAnswerGenerator()

    records: list[EvalRecord] = []
    for question in questions:
        results = engine.search(
            question.question,
            top_k=top_k,
            mode=retrieval_mode,
            rerank=(reranker_backend != "none"),
            rerank_k=max(top_k * 3, top_k),
        )
        retrieved_documents = _dedupe_document_ids([result.chunk.document_id for result in results])
        relevant = set(question.relevant_documents)
        retrieved_relevant = [doc_id for doc_id in retrieved_documents if doc_id in relevant]
        precision = len(retrieved_relevant) / top_k
        recall = len(retrieved_relevant) / max(len(relevant), 1)

        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_documents, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break

        answer = generator.answer(question.question, results)
        answer_lower = answer.lower()
        answer_hit = 1.0 if all(term.lower() in answer_lower for term in question.expected_answer[:2]) else 0.0

        records.append(
            EvalRecord(
                question_id=question.id,
                question=question.question,
                retrieved_documents=retrieved_documents,
                precision_at_k=precision,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                answer_hit=answer_hit,
            )
        )

    summary = {
        "chunking_strategy": chunking_strategy,
        "retrieval_mode": retrieval_mode,
        "embedding_backend": engine.backend.name,
        "reranker_backend": reranker_backend,
        "chunk_count": len(chunks),
        "top_k": top_k,
        "precision_at_k": round(mean(record.precision_at_k for record in records), 4),
        "recall_at_k": round(mean(record.recall_at_k for record in records), 4),
        "mrr": round(mean(record.reciprocal_rank for record in records), 4),
        "answer_hit_rate": round(mean(record.answer_hit for record in records), 4),
        "records": [asdict(record) for record in records],
    }
    return summary


def compare_chunking_strategies(
    retrieval_mode: str = "dense",
    embedding_backend: str = "auto",
    reranker_backend: str = "none",
    top_k: int = 5,
) -> list[dict[str, object]]:
    return [
        evaluate_configuration(
            "fixed",
            retrieval_mode=retrieval_mode,
            embedding_backend=embedding_backend,
            reranker_backend=reranker_backend,
            top_k=top_k,
        ),
        evaluate_configuration(
            "markdown",
            retrieval_mode=retrieval_mode,
            embedding_backend=embedding_backend,
            reranker_backend=reranker_backend,
            top_k=top_k,
        ),
    ]


def format_results_table(results: list[dict[str, object]]) -> str:
    lines = [
        "| chunking | retrieval | backend | reranker | chunks | precision@k | recall@k | mrr | answer hit |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {chunking_strategy} | {retrieval_mode} | {embedding_backend} | {reranker_backend} | {chunk_count} | {precision_at_k:.4f} | {recall_at_k:.4f} | {mrr:.4f} | {answer_hit_rate:.4f} |".format(**result)
        )
    return "\n".join(lines)


def save_results(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
