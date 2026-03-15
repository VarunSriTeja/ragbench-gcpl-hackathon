from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EmbeddingBackend:
    name: str

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfBackend(EmbeddingBackend):
    def __init__(self) -> None:
        super().__init__(name="tfidf")
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), stop_words="english")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).toarray().astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype(np.float32)


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__(name=model_name)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


def build_backend(kind: str = "auto") -> EmbeddingBackend:
    if kind == "tfidf":
        return TfidfBackend()
    if kind == "sentence-transformers":
        return SentenceTransformerBackend()
    if kind != "auto":
        raise ValueError(f"Unsupported embedding backend: {kind}")

    try:
        return SentenceTransformerBackend()
    except Exception:
        return TfidfBackend()


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
    matrix_norm = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    return query_norm @ matrix_norm.T
