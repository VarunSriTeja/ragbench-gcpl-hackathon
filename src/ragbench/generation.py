from __future__ import annotations

import os
import re
from collections import defaultdict

from ragbench.types import RetrievalResult


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"\b\w+\b")
STOPWORDS = {
    "the", "a", "an", "is", "are", "what", "why", "how", "does", "do", "of", "for", "to", "in", "and", "or",
}


class GroundedAnswerGenerator:
    def answer(self, question: str, results: list[RetrievalResult]) -> str:
        if os.getenv("RAGBENCH_USE_OPENAI") == "1" and os.getenv("OPENAI_API_KEY"):
            try:
                return self._openai_answer(question, results)
            except Exception:
                pass
        return self._extractive_answer(question, results)

    def _extractive_answer(self, question: str, results: list[RetrievalResult]) -> str:
        if not results:
            return "I do not know based on the retrieved context."

        query_terms = {token.lower() for token in TOKEN_RE.findall(question) if token.lower() not in STOPWORDS}
        ordered_query_terms = [token.lower() for token in TOKEN_RE.findall(question) if token.lower() not in STOPWORDS]
        query_bigrams = {
            " ".join((ordered_query_terms[index], ordered_query_terms[index + 1]))
            for index in range(len(ordered_query_terms) - 1)
        }
        sentence_scores: list[tuple[float, str, str]] = []
        seen_sentences: set[str] = set()
        best_overlap = 0
        best_bigram_hits = 0

        for rank, result in enumerate(results, start=1):
            base_weight = 1.0 / rank
            title_terms = {token.lower() for token in TOKEN_RE.findall(result.chunk.title)}
            for sentence in SENTENCE_RE.split(result.chunk.text.replace("\n", " ")):
                sentence = sentence.strip()
                if len(sentence) < 30:
                    continue
                normalized = sentence.lower()
                if normalized in seen_sentences:
                    continue
                seen_sentences.add(normalized)
                tokens = {token.lower() for token in TOKEN_RE.findall(sentence)}
                overlap = len(query_terms & tokens)
                bigram_hits = sum(1 for bigram in query_bigrams if bigram in normalized)
                title_overlap = len(query_terms & title_terms)
                best_overlap = max(best_overlap, overlap)
                best_bigram_hits = max(best_bigram_hits, bigram_hits)
                score = (overlap * 3.0) + (bigram_hits * 2.0) + (title_overlap * 0.5) + base_weight + max(result.score, 0.0)
                sentence_scores.append((score, sentence, result.chunk.title))

        sentence_scores.sort(key=lambda item: item[0], reverse=True)
        top_sentences = sentence_scores[:4]
        if not top_sentences or top_sentences[0][0] < 0.5:
            return "I do not know based on the retrieved context."

        if best_overlap == 0 and best_bigram_hits == 0:
            return "I do not know based on the retrieved context."

        grouped: dict[str, list[str]] = defaultdict(list)
        for _, sentence, title in top_sentences:
            grouped[title].append(sentence)

        lines = ["Grounded answer:"]
        for title, sentences in grouped.items():
            merged = " ".join(sentences[:2])
            lines.append(f"- {merged} [Source: {title}]")
        return "\n".join(lines)

    def _openai_answer(self, question: str, results: list[RetrievalResult]) -> str:
        from openai import OpenAI

        context_blocks = []
        for result in results:
            context_blocks.append(f"Source: {result.chunk.title}\n{result.chunk.text}")

        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("RAGBENCH_OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Answer only from the provided context. If the context is insufficient, say you do not know. "
                                "Cite source titles inline in square brackets."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Question: {question}\n\nContext:\n" + "\n\n".join(context_blocks),
                        }
                    ],
                },
            ],
        )
        return response.output_text.strip()
