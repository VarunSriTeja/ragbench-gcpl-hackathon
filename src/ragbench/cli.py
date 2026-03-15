from __future__ import annotations

import argparse
from pathlib import Path

from ragbench.chunking import chunk_documents
from ragbench.dataset import load_documents
from ragbench.evaluation import compare_chunking_strategies, format_results_table, save_results
from ragbench.export_pdf import export_submission_pdf
from ragbench.generation import GroundedAnswerGenerator
from ragbench.retrieval import SearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG benchmark prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question")
    ask_parser.add_argument("question", help="Question to answer")
    ask_parser.add_argument("--chunking", choices=["fixed", "markdown"], default="markdown")
    ask_parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    ask_parser.add_argument("--backend", choices=["auto", "sentence-transformers", "tfidf"], default="auto")
    ask_parser.add_argument("--reranker", choices=["none", "auto", "cross-encoder", "heuristic"], default="none")
    ask_parser.add_argument("--top-k", type=int, default=4)

    eval_parser = subparsers.add_parser("evaluate", help="Run retrieval benchmark")
    eval_parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    eval_parser.add_argument("--backend", choices=["auto", "sentence-transformers", "tfidf"], default="auto")
    eval_parser.add_argument("--reranker", choices=["none", "auto", "cross-encoder", "heuristic"], default="none")
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--save-json", type=Path)

    pdf_parser = subparsers.add_parser("export-pdf", help="Export submission notes to a PDF")
    pdf_parser.add_argument("--output", type=Path, default=Path("artifacts/submission_summary.pdf"))

    return parser


def run_ask(args: argparse.Namespace) -> int:
    documents = load_documents()
    chunks = chunk_documents(documents, strategy=args.chunking)
    engine = SearchEngine.build(
        chunks,
        embedding_backend=args.backend,
        enable_hybrid=(args.mode == "hybrid"),
        reranker_backend=args.reranker,
    )
    results = engine.search(
        args.question,
        top_k=args.top_k,
        mode=args.mode,
        rerank=(args.reranker != "none"),
        rerank_k=max(args.top_k * 3, args.top_k),
    )

    print(f"Question: {args.question}\n")
    print("Top context:")
    for result in results:
        rerank_suffix = f", rerank={result.rerank_score:.4f}" if result.rerank_score is not None else ""
        print(f"- [{result.chunk.document_id}] {result.chunk.title} :: score={result.score:.4f}{rerank_suffix}")

    answer = GroundedAnswerGenerator().answer(args.question, results)
    print("\n" + answer)
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    results = compare_chunking_strategies(
        retrieval_mode=args.mode,
        embedding_backend=args.backend,
        reranker_backend=args.reranker,
        top_k=args.top_k,
    )
    table = format_results_table(results)
    print(table)
    if args.save_json:
        save_results(results, args.save_json)
        print(f"\nSaved JSON results to {args.save_json}")
    return 0


def run_export_pdf(args: argparse.Namespace) -> int:
    export_submission_pdf(args.output)
    print(f"Saved submission PDF to {args.output}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ask":
        return run_ask(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "export-pdf":
        return run_export_pdf(args)
    parser.error("Unknown command")
    return 1
