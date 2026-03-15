from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    id: str
    title: str
    source: str
    tags: list[str]
    content: str


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    title: str
    source: str
    strategy: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


@dataclass(slots=True)
class EvalQuestion:
    id: str
    question: str
    expected_answer: list[str]
    relevant_documents: list[str]


@dataclass(slots=True)
class EvalRecord:
    question_id: str
    question: str
    retrieved_documents: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    answer_hit: float
