from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ragbench.embeddings import EmbeddingBackend, build_backend, cosine_similarity_matrix
from ragbench.reranking import BaseReranker, build_reranker
from ragbench.types import Chunk, RetrievalResult


TOKEN_RE = re.compile(r"\b\w+\b")


@dataclass(slots=True)
class SearchEngine:
    chunks: list[Chunk]
    backend: EmbeddingBackend
    dense_matrix: np.ndarray
    lexical_model: object | None = None
    lexical_corpus: list[list[str]] | None = None
    reranker: BaseReranker | None = None

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedding_backend: str = "auto",
        enable_hybrid: bool = False,
        reranker_backend: str = "none",
    ) -> "SearchEngine":
        backend = build_backend(embedding_backend)
        dense_matrix = backend.fit_transform([chunk.text for chunk in chunks])
        lexical_model = None
        lexical_corpus = None
        reranker = None
        if enable_hybrid:
            from rank_bm25 import BM25Okapi

            lexical_corpus = [TOKEN_RE.findall(chunk.text.lower()) for chunk in chunks]
            lexical_model = BM25Okapi(lexical_corpus)
        if reranker_backend != "none":
            reranker = build_reranker(reranker_backend)
        return cls(
            chunks=chunks,
            backend=backend,
            dense_matrix=dense_matrix,
            lexical_model=lexical_model,
            lexical_corpus=lexical_corpus,
            reranker=reranker,
        )

    def search(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "dense",
        alpha: float = 0.5,
        rerank: bool = False,
        rerank_k: int | None = None,
    ) -> list[RetrievalResult]:
        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode must be 'dense' or 'hybrid'")

        query_vector = self.backend.transform([question])
        dense_scores = cosine_similarity_matrix(query_vector, self.dense_matrix)[0]
        dense_ranking = np.argsort(-dense_scores)

        candidate_limit = max(top_k, rerank_k or top_k)
        candidate_limit = max(candidate_limit, top_k * 4, 10) if rerank else candidate_limit

        if mode == "dense" or self.lexical_model is None or self.lexical_corpus is None:
            results = [
                RetrievalResult(chunk=self.chunks[idx], score=float(dense_scores[idx]), dense_rank=rank + 1)
                for rank, idx in enumerate(dense_ranking[:candidate_limit])
            ]
            return self._maybe_rerank(question, results, top_k=top_k, rerank_k=rerank_k) if rerank else results[:top_k]

        tokens = TOKEN_RE.findall(question.lower())
        bm25_scores = np.asarray(self.lexical_model.get_scores(tokens), dtype=np.float32)
        lexical_ranking = np.argsort(-bm25_scores)

        dense_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(dense_ranking[:candidate_limit])}
        lexical_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(lexical_ranking[:candidate_limit])}
        candidate_ids = set(dense_rank_map) | set(lexical_rank_map)

        fused: list[tuple[int, float]] = []
        for idx in candidate_ids:
            dense_component = 0.0
            lexical_component = 0.0
            if idx in dense_rank_map:
                dense_component = 1.0 / (60 + dense_rank_map[idx])
            if idx in lexical_rank_map:
                lexical_component = 1.0 / (60 + lexical_rank_map[idx])
            score = alpha * dense_component + (1 - alpha) * lexical_component
            fused.append((idx, score))

        fused.sort(key=lambda item: item[1], reverse=True)
        results: list[RetrievalResult] = []
        for idx, score in fused[:candidate_limit]:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    dense_rank=dense_rank_map.get(idx),
                    lexical_rank=lexical_rank_map.get(idx),
                )
            )
        return self._maybe_rerank(question, results, top_k=top_k, rerank_k=rerank_k) if rerank else results[:top_k]

    def _maybe_rerank(
        self,
        question: str,
        results: list[RetrievalResult],
        top_k: int,
        rerank_k: int | None,
    ) -> list[RetrievalResult]:
        if self.reranker is None:
            return results[:top_k]

        candidate_count = min(len(results), rerank_k or max(top_k * 3, top_k))
        rerank_inputs = results[:candidate_count]
        rerank_scores = self.reranker.score(question, [result.chunk.text for result in rerank_inputs])

        base_rank_map = {result.chunk.id: rank + 1 for rank, result in enumerate(rerank_inputs)}
        rerank_order = sorted(
            range(len(rerank_inputs)),
            key=lambda index: rerank_scores[index],
            reverse=True,
        )
        rerank_rank_map = {
            rerank_inputs[index].chunk.id: rank + 1
            for rank, index in enumerate(rerank_order)
        }

        rescored: list[RetrievalResult] = []
        for result, rerank_score in zip(rerank_inputs, rerank_scores, strict=False):
            result.rerank_score = float(rerank_score)
            base_component = 1.0 / (60 + base_rank_map[result.chunk.id])
            rerank_component = 1.0 / (60 + rerank_rank_map[result.chunk.id])
            result.score = (0.3 * base_component) + (0.7 * rerank_component)
            rescored.append(result)

        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
