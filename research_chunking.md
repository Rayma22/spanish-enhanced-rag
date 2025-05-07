# research_chunking

## 1. Overview

The `research_chunking` module is focused on document segmentation strategies to enhance retrieval performance and maintain semantic coherence. Proper chunking improves context relevance and reduces hallucination risk in the generation step.

This module allows testing of different chunking methods such as sentence-based windows, fixed-length tokens, paragraph segmentation, and semantic boundary detection.

---

## 2. Current Implementation

The system currently uses a sentence-based sliding window approach implemented through a custom `SentenceWindowNodeParser`.

### Key Characteristics:

- **Chunk Unit:** Sentence
- **Window Size:** 3 (sliding window of 3 sentences per chunk)
- **Overlap:** Enabled (each chunk shares overlap with the previous)
- **Storage:** Chunked content is embedded and stored in the document vector index

This method preserves semantic flow between sentences while limiting the token count of each input to the language model.

---

## 3. Design Rationale

The following design choices were made to balance performance and coherence:

- **Sentence-based segmentation** ensures context is linguistically meaningful.
- **Windowing** prevents loss of important transitional content.
- **Overlap** allows questions to align with the right chunk even if they span sentence boundaries.

This approach reduces false negatives in retrieval caused by fragmented ideas and maintains a manageable token size for generative models.

---

## 4. Alternative Chunking Strategies (Planned)

The module is designed to support experimentation with:

### Fixed-Length Token Chunking:
- Split text by token count (e.g., every 256 tokens)
- Easier to manage with token-limited LLMs

### Paragraph-Based Chunking:
- Preserve formatting and structure
- Useful for long-form academic or legal content

### Semantic Chunking:
- Use similarity scores or topic modeling to identify coherent segments
- Potentially combine with Named Entity Recognition (NER)

---

## 5. Evaluation Criteria

To evaluate chunking effectiveness, the following metrics are considered:

- **Context relevance** (retrieved chunk matches query intent)
- **Answer accuracy** (how well the chunk supports correct generation)
- **Retrieval precision and recall** (top-k chunk relevance)
- **Latency** (processing time for chunk generation)
- **Token economy** (total size of each chunk for LLM input)

---

## 6. Future Work

- Add configuration support to toggle chunking strategies at runtime
- Visualize chunk overlaps and boundary impacts
- Measure performance across different document domains (e.g., tech, news, medical)
- Integrate transformer-based segmenters (e.g., `textsplit`, `spacy-transformers`)
- Fine-tune retrieval indexes per chunking method for deeper comparison
