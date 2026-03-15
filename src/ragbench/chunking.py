from __future__ import annotations

import re
from collections.abc import Iterable

from ragbench.types import Chunk, Document


WORD_RE = re.compile(r"\S+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _word_chunks(words: list[str], chunk_size: int, overlap: int) -> Iterable[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size)
        if start >= end:
            break
        yield start, end, " ".join(words[start:end])
        if end == len(words):
            break


def _fixed_chunks(text: str, chunk_size: int = 120, overlap: int = 30) -> list[dict[str, object]]:
    words = WORD_RE.findall(text)
    chunks: list[dict[str, object]] = []
    for index, (start, end, chunk_text) in enumerate(_word_chunks(words, chunk_size, overlap)):
        chunks.append(
            {
                "text": chunk_text,
                "chunk_index": index,
                "start_word": start,
                "end_word": end,
            }
        )
    return chunks


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def _markdown_chunks(text: str, chunk_size: int = 160, overlap: int = 25) -> list[dict[str, object]]:
    sections = _split_markdown_sections(text)
    chunks: list[dict[str, object]] = []
    chunk_index = 0
    for heading, body in sections:
        words = WORD_RE.findall(body)
        if len(words) <= chunk_size:
            chunks.append(
                {
                    "text": f"{heading}\n\n{body}".strip(),
                    "chunk_index": chunk_index,
                    "section": heading,
                }
            )
            chunk_index += 1
            continue

        for start, end, chunk_text in _word_chunks(words, chunk_size, overlap):
            chunks.append(
                {
                    "text": f"{heading}\n\n{chunk_text}".strip(),
                    "chunk_index": chunk_index,
                    "section": heading,
                    "start_word": start,
                    "end_word": end,
                }
            )
            chunk_index += 1
    return chunks


def chunk_document(document: Document, strategy: str) -> list[Chunk]:
    if strategy == "fixed":
        parts = _fixed_chunks(document.content)
    elif strategy == "markdown":
        parts = _markdown_chunks(document.content)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    chunks: list[Chunk] = []
    for part in parts:
        chunk_index = int(part["chunk_index"])
        chunks.append(
            Chunk(
                id=f"{document.id}:{strategy}:{chunk_index}",
                document_id=document.id,
                title=document.title,
                source=document.source,
                strategy=strategy,
                chunk_index=chunk_index,
                text=str(part["text"]),
                metadata={k: v for k, v in part.items() if k not in {"text", "chunk_index"}},
            )
        )
    return chunks


def chunk_documents(documents: list[Document], strategy: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy=strategy))
    return chunks
