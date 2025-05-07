# research_genlm

## 1. Overview

The `research_genlm` module focuses on the generative component of the RAG pipeline. It is responsible for generating answers based on contextually retrieved documents using a transformer-based language model.

This module allows experimentation with different language models, prompt engineering techniques, generation parameters, and fallback logic to improve the quality, relevance, and factuality of the generated responses.

---

## 2. Current Implementation

The system uses the `Qwen/Qwen2.5-0.5B-Instruct` model served via Hugging Face’s Text Generation Inference (TGI). This setup provides efficient and scalable generation while maintaining support for instruction-following behavior.

### Key Features:
- **Model:** Qwen2.5-0.5B-Instruct
- **Serving:** Hugging Face TGI (containerized)
- **Prompt Template:** ChatML-style with system, user, and assistant roles
- **Generation Parameters:**  
  - `temperature`: 0.3  
  - `max_tokens`: 150  
  - `top_p`: 0.92  
  - `top_k`: 40  
  - `repetition_penalty`: 1.2  

---

## 3. Prompt Structure

The prompt format ensures that the model only generates answers grounded in the retrieved documents.

### Example Prompt:
```
<|im_start|>system
Es muy importante que tu respuesta se base exclusivamente en la información proporcionada...
<|im_end|>

<|im_start|>user
Contexto:
- Documento 1
- Documento 2

Basándote exclusivamente en el contexto anterior, responde a esta pregunta: ¿Qué es una red neuronal?
<|im_end|>

<|im_start|>assistant
```

---

## 4. Response Quality Controls

The system applies several safeguards to prevent hallucinations and irrelevant outputs:

- **Prompt reinforcement** that prohibits introducing new information
- **Heuristic filters** to detect overly short or generic responses
- **Fallback to direct retrieval-based response** if generation quality is poor

Additionally, token filtering removes model artifacts such as `<|`, `<pad>`, `endoftext`, etc.

---

## 5. Key Code References

```python
outputs = self.generator(
    prompt,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.3,
    top_p=0.92,
    top_k=40,
    repetition_penalty=1.2,
    num_return_sequences=1,
    return_full_text=True
)
```

If the generated output is flagged as low quality by internal checks, the system uses a fallback function to create a direct response using retrieved contexts.

---

## 6. Future Work

- Benchmark multiple open-source models (e.g., Mistral, LLaMA2, Falcon)
- Introduce conversational memory to support multi-turn dialogue
- Apply LoRA or QLoRA fine-tuning on Spanish academic datasets
- Integrate prompt variants (e.g., zero-shot vs. few-shot)
- Experiment with structured response formats (e.g., JSON output for QA pairs)
