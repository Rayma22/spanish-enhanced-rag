# research_embedding

## 1. Overview

The `research_embedding` component is dedicated to investigating and improving the document and query embedding process within the RAG system. This step is foundational for semantic retrieval, as the quality and structure of vector representations directly affect the accuracy of document similarity matching.

This module allows experimentation with different embedding models, normalization strategies, and performance benchmarking.

---

## 2. Current Implementation

The system currently uses the `intfloat/multilingual-e5-small` model via the `sentence-transformers` library to generate multilingual embeddings. These embeddings are normalized (L2 norm), and cosine similarity is used to retrieve top-k relevant documents.

### Model Used:
- **Name:** `intfloat/multilingual-e5-small`
- **Framework:** SentenceTransformers
- **Vector Dimensionality:** 384
- **Similarity Function:** Cosine Similarity (via scikit-learn)

### Workflow Summary:
1. Input documents are encoded using the SentenceTransformer model.
2. Embeddings are normalized and stored on disk.
3. Query is embedded and compared against stored vectors.
4. Top-k most similar documents are retrieved based on cosine similarity.

---

## 3. Key Code References

```python
self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
self.document_embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
```

Embeddings are persisted in:
- `index_storage/embeddings.npy`
- `index_storage/documents.json`

---

## 4. Potential Enhancements

This module is designed for extensibility. Below are ideas for further exploration:

### Alternative Models:
- `sentence-transformers/all-MiniLM-L6-v2`
- `LaBSE`
- `gte-small`
- `e5-large-v2`

### Evaluation Criteria:
- Retrieval accuracy (precision@k, recall@k)
- Embedding generation latency
- Memory and storage efficiency
- Multilingual generalization

### Advanced Ideas:
- Add support for Faiss or HNSW indexing
- Use query-specific dimensionality reduction (e.g., PCA)
- Train domain-specific embedding models on fine-tuned corpora
- Experiment with hybrid embeddings (concatenating sentence and document-level vectors)

---

## 5. Future Work

- Build benchmark scripts to compare model retrieval effectiveness
- Add model selector via configuration file or environment variable
- Introduce automatic evaluation with labeled query-document relevance pairs
- Track retrieval performance under varying noise levels and query types
