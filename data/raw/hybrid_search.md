---
title: Hybrid Search
source: https://docs.weaviate.io/weaviate/search/hybrid
tags: retrieval, hybrid, bm25, vector
---

# Hybrid Search

Hybrid search combines vector retrieval and keyword retrieval. The vector side captures semantic similarity, while the keyword side preserves exact-token matching for product names, acronyms, and rare phrases.

## Why hybrid search helps

Dense retrieval works well when users and documents use different wording for the same concept. Keyword retrieval works well when exact terms matter. Combining both often improves recall and ranking quality.

## Weighting behavior

Hybrid systems usually expose a control over the balance between lexical and semantic signals. In some systems, an alpha of one means pure vector search, while zero means pure keyword search.

## Fusion

A hybrid pipeline needs a fusion method to combine result lists. Relative-score fusion and rank-based fusion are common choices. Rank-based fusion is simple when the score scales from the two retrievers are not directly comparable.

## Operational value

Hybrid retrieval is especially useful for enterprise search, where users ask natural-language questions but also rely on exact internal terms. It is a practical way to reduce misses that come from using only one retrieval signal.
