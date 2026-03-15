---
title: Evaluation Design for RAG
source: https://www.pinecone.io/learn/retrieval-augmented-generation/
tags: evaluation, benchmark, metrics
---

# Evaluation Design for RAG

RAG quality should be measured with controlled experiments rather than intuition. A useful benchmark starts with a fixed set of user questions and expected answers.

## Ground truth

Each evaluation question should have a known relevant source or expected answer pattern. Without ground truth, retrieval experiments are difficult to compare fairly.

## Retrieval metrics

Precision@k measures how many of the returned items are relevant. Recall@k measures how much of the relevant material was found within the top-k results. Mean reciprocal rank rewards systems that place the first relevant hit early in the list.

## Qualitative review

Numbers alone are not enough. It is also important to inspect failure cases such as missed sources, partial grounding, redundant chunks, and hallucinated claims.

## Reproducibility

When comparing chunking or retrieval strategies, keep the dataset and test questions fixed. Change only one main variable at a time so the comparison stays interpretable.
