from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"\b\w+\b")
STOPWORDS = {
    "the", "a", "an", "is", "are", "what", "why", "how", "does", "do", "of", "for", "to", "in", "and", "or",
}


@dataclass(slots=True)
class BaseReranker:
    name: str

    def score(self, query: str, documents: list[str]) -> list[float]:
        raise NotImplementedError


class HeuristicReranker(BaseReranker):
    def __init__(self) -> None:
        super().__init__(name="heuristic")

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_terms = [token.lower() for token in TOKEN_RE.findall(query) if token.lower() not in STOPWORDS]
        query_term_set = set(query_terms)
        scores: list[float] = []
        for document in documents:
            doc_tokens = [token.lower() for token in TOKEN_RE.findall(document)]
            if not doc_tokens:
                scores.append(0.0)
                continue
            doc_term_set = set(doc_tokens)
            overlap = len(query_term_set & doc_term_set)
            density = overlap / max(len(doc_term_set), 1)
            phrase_bonus = 0.0
            lowered = document.lower()
            for term in query_terms:
                if term in lowered:
                    phrase_bonus += 0.05
            scores.append(float(overlap + density + phrase_bonus))
        return scores


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        super().__init__(name=model_name)
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [[query, document] for document in documents]
        scores = self.model.predict(pairs)
        return [float(score) for score in scores]


def build_reranker(kind: str = "auto") -> BaseReranker:
    if kind == "heuristic":
        return HeuristicReranker()
    if kind == "cross-encoder":
        return CrossEncoderReranker()
    if kind != "auto":
        raise ValueError(f"Unsupported reranker backend: {kind}")

    try:
        return CrossEncoderReranker()
    except Exception:
        return HeuristicReranker()
