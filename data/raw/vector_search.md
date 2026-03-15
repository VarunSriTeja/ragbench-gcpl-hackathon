---
title: Vector Search Concepts
source: https://qdrant.tech/documentation/concepts/search/
tags: retrieval, vector-search, similarity
---

# Vector Search Concepts

Vector search retrieves items whose embeddings are close to a query embedding. It is commonly implemented as nearest-neighbor search over dense vectors.

## Similarity metrics

Common metrics include cosine similarity, dot product, Euclidean distance, and Manhattan distance. Cosine similarity is a frequent default for text embeddings because it compares direction more than magnitude.

## Filtering and thresholds

Vector search systems often support payload filtering and score thresholds. Thresholding can remove low-confidence matches, while filtering allows retrieval inside a subset such as one document type or business unit.

## Approximate versus exact search

Approximate nearest-neighbor methods trade some exactness for speed at scale. Exact search can be slower but may be useful for debugging or evaluation.

## Grouping and document-level views

When a single document is split into multiple chunks, it can be useful to group results by document id. This reduces redundant chunk-level hits and makes document-level evaluation easier to interpret.
