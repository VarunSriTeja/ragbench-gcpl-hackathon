---
title: RAG Fundamentals
source: https://www.pinecone.io/learn/retrieval-augmented-generation/
tags: rag, fundamentals, hallucination
---

# Retrieval-Augmented Generation

Retrieval-augmented generation improves model output by combining a user question with relevant external context before response generation. The usual pipeline has four stages: ingestion, retrieval, augmentation, and generation.

## Why teams use RAG

Foundation models have knowledge cutoffs, can miss domain-specific detail, and cannot directly access private company data used after training. They also may sound confident while being wrong. RAG addresses these issues by grounding responses in authoritative material that can be traced back to a source.

## Ingestion

During ingestion, documents are cleaned, chunked, embedded, and stored in a retrieval index. Chunking matters because long documents need to be broken into smaller units that still preserve meaning.

## Retrieval

A user query is embedded and compared against stored chunk embeddings. Retrieval can be dense, lexical, or hybrid. High-quality retrieval is essential because poor context leads to poor answers.

## Augmentation and generation

The retrieved context is added to the prompt. A well-behaved answering system should say it does not know when the supporting context is missing. This reduces hallucination risk and improves trust.

## Evaluation

Good RAG systems need ground-truth evaluation sets. A representative set of questions and expected answers is necessary to determine whether retrieval changes actually improve system quality.
