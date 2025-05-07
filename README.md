# Spanish Enhanced RAG System

## Table of Contents

1. [General Overview](#1-general-overview)  
2. [Component: research_embedding](#2-component-researchembedding)  
3. [Component: research_genlm](#3-component-researchgenlm)  
4. [Component: research_chunking](#4-component-researchchunking)  
5. [Component: research_advanced_rag](#5-component-researchadvanced_rag)  
6. [Infrastructure Setup (Docker)](#6-infrastructure-setup-docker)  
7. [API Documentation](#7-api-documentation)  
8. [Future Work](#8-future-work)  

---

## 1. General Overview

This project is a modular framework for developing and experimenting with an Enhanced Retrieval-Augmented Generation (RAG) pipeline optimized for Spanish-language queries. It incorporates state-of-the-art multilingual embeddings, generative language models, document chunking, and hybrid retrieval strategies.

The system is designed with modular research components to allow rapid iteration and testing across embedding strategies, chunking methods, generation configurations, and advanced retrieval logic.

---

## 2. Component: research_embedding

This module focuses on analyzing and optimizing text embedding strategies to improve vector representations of input documents. The current setup uses `intfloat/multilingual-e5-small` via the SentenceTransformers library with normalized cosine similarity.

Refer to [`research_embedding/README.md`](./research_embedding/README.md) for implementation details and future enhancements.

---

## 3. Component: research_genlm

This component explores the use of generative language models for response generation. The system currently uses `Qwen/Qwen2.5-0.5B-Instruct` served through Hugging Face’s Text Generation Inference (TGI). Prompt formatting, temperature tuning, and fallbacks are core research areas.

See [`research_genlm/README.md`](./research_genlm/README.md) for design rationale and usage.

---

## 4. Component: research_chunking

This component investigates document segmentation techniques. The goal is to identify chunking methods that balance retrieval accuracy and semantic coherence. The current system uses a sentence-based windowed approach, with additional strategies under experimentation.

Full details are in [`research_chunking/README.md`](./research_chunking/README.md).

---

## 5. Component: research_advanced_rag

This module focuses on enhancements such as hybrid retrieval, re-ranking, fallback response strategies, and query transformations. It aims to push beyond basic dense retrieval and improve the system’s adaptability to complex information needs.

Advanced techniques are documented in [`research_advanced_rag/README.md`](./research_advanced_rag/README.md).

## 6. Infrastructure Setup (Docker)

The system runs via Docker Compose, which orchestrates three primary services:

- `tei`: Text embedding server using SentenceTransformers  
- `tgi`: Text generation server using Hugging Face TGI  
- `rag`: FastAPI application coordinating retrieval and generation  

To build and start the system locally, run:

```bash
docker-compose up --build
```

---

## 7. API Documentation

### Upload Documents

**Endpoint:** `POST /upload`  
**Description:** Adds one or more documents to the knowledge base.

**Request Body:**

```json
{
  "texts": ["Documento 1", "Documento 2"]
}
```

---

### Generate Response

**Endpoint:** `POST /generate`  
**Description:** Generates a response to a query using the RAG system.

**Request Body:**

```json
{
  "query": "¿Qué es una red neuronal?"
}
```

**Response Fields:**

- **`response`**: The generated or fallback answer.  
- **`contexts`**: The top-k retrieved context segments.  
- **`scores`**: Cosine similarity scores associated with each context.  
- **`response_type`**: One of `generation`, `retrieval_fallback`, `no_context`, or `error_fallback`.

---

## 8. Future Work

- Integrate re-ranking using cross-encoder models for improved precision.  
- Benchmark additional embedding and generation models.  
- Add support for multi-hop retrieval and conversational query memory.  
- Extend metadata storage for better traceability and auditing.  
- Optimize for GPU-based deployments and scalable inference.

---

## 9. Alternative Models Considered
Several models were evaluated during development, but not adopted due to limitations related to hallucination, inference speed, or deployment constraints. Our goal was to strike a practical balance between generation quality, semantic relevance, and runtime efficiency.

LaBSE: Produced strong multilingual embeddings, but inference was significantly slower, and it underperformed on short-form queries during testing.

LLaMA2-7B: Delivered high-quality completions but required GPU resources we couldn’t support in our local or Docker-based setup. It also exhibited factual drift in longer responses.

Mistral-7B-Instruct: Promising for accurate answers, but prone to hallucination in Spanish when retrieval context was limited. Integration via TGI was also less stable at the time.

Ultimately, we selected Qwen2.5-0.5B-Instruct and multilingual-e5-small for their stable performance, reasonable accuracy, and low latency, especially in CPU-constrained environments.







