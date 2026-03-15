---
title: Reranking and Two-Stage Retrieval
source: https://www.pinecone.io/learn/series/rag/rerankers/
tags: reranking, retrieval, rag
---

# Reranking and Two-Stage Retrieval

In a two-stage retrieval system, the first stage retrieves a broader candidate set and the second stage reranks that set more carefully. This helps balance speed and quality.

## Why reranking matters

Dense retrieval compresses documents into single vectors, which introduces information loss. As a result, relevant chunks may appear lower in the retrieved list than they should.

## Recall versus context limits

A tempting approach is to pass many retrieved chunks directly to the language model. This often hurts answer quality because long context windows increase noise, cost, and the chance that relevant facts are buried.

## Benefit of rerankers

A reranker scores a query together with each candidate document, which is usually more accurate than relying on vector similarity alone. The trade-off is latency, because reranking is slower than nearest-neighbor search.

## Practical lesson

A strong retrieval pipeline often maximizes recall in stage one, then trims to the most relevant evidence before answer generation.
