import os
import time
import logging
import sys
import gc
import json
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set up storage directories
INDEX_STORAGE_DIR = "/app/index_storage"
logger.info(f"Checking if directory exists: {INDEX_STORAGE_DIR}")
if os.path.exists(INDEX_STORAGE_DIR):
    logger.info(f"Directory {INDEX_STORAGE_DIR} exists")
else:
    logger.info(f"Directory {INDEX_STORAGE_DIR} does not exist, creating it")
    try:
        os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)
        logger.info(f"Successfully created directory {INDEX_STORAGE_DIR}")
    except Exception as e:
        logger.error(f"Error creating directory {INDEX_STORAGE_DIR}: {e}")

# FastAPI imports
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# For embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

class EnhancedSpanishRAG:
    """
    Enhanced RAG implementation for Spanish optimized for the Qwen2.5-0.5B-Instruct model
    with a relevance_boost of 0.7
    """
    def __init__(
        self,
        embedding_model_name: str = "intfloat/multilingual-e5-small",
        k: int = 3,
        threshold: float = 0.7,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        relevance_boost: float = 0.7
    ):
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Store configuration
        self.k = k
        self.threshold = threshold
        self.relevance_boost = relevance_boost
        self.documents = []
        self.document_embeddings = None
        
        # Check for GPU
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            logger.info("GPU not available. Models will run on CPU (slower).")
            self.device = torch.device("cpu")
        
        # Load the generative model
        self.model_name = model_name
        self.load_model()
        
        # Initialize documents path
        self.documents_path = os.path.join(INDEX_STORAGE_DIR, "documents.json")
        self.embeddings_path = os.path.join(INDEX_STORAGE_DIR, "embeddings.npy")
        
        # Load existing documents and embeddings if available
        self.load_documents_and_embeddings()
    
    def load_documents_and_embeddings(self):
        """Load existing documents and embeddings from storage"""
        try:
            if os.path.exists(self.documents_path) and os.path.exists(self.embeddings_path):
                logger.info("Loading existing documents and embeddings")
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                self.document_embeddings = np.load(self.embeddings_path)
                logger.info(f"Loaded {len(self.documents)} documents from storage")
            else:
                logger.info("No existing documents found")
        except Exception as e:
            logger.error(f"Error loading documents and embeddings: {e}")
            self.documents = []
            self.document_embeddings = None
    
    def save_documents_and_embeddings(self):
        """Save documents and embeddings to storage"""
        try:
            logger.info(f"Saving {len(self.documents)} documents and embeddings")
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False)
            np.save(self.embeddings_path, self.document_embeddings)
            logger.info("Documents and embeddings saved successfully")
        except Exception as e:
            logger.error(f"Error saving documents and embeddings: {e}")
    
    def load_model(self) -> bool:
        """Load the language model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            start_time = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Ensure we have a pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure model parameters based on device
            if self.cuda_available:
                logger.info("Using GPU acceleration with float16")
                # Load model with GPU optimization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # Load model on CPU
                logger.info("Using CPU for model inference")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.cuda_available else "cpu"
            )
            
            # Calculate loading time
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_documents(self, texts: List[str]) -> Dict[str, Any]:
        """Add documents to the knowledge base"""
        start_time = time.time()
        logger.info(f"Adding {len(texts)} documents to the knowledge base")
        
        # If we have existing documents, extend them
        if self.documents and self.document_embeddings is not None:
            existing_docs_count = len(self.documents)
            logger.info(f"Extending existing {existing_docs_count} documents")
            
            # Generate embeddings for new documents
            new_embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True
            )
            
            # Combine with existing embeddings
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
            self.documents.extend(texts)
        else:
            # Start fresh
            logger.info("Creating new document index")
            self.documents = texts
            
            # Generate embeddings for all documents
            self.document_embeddings = self.embedding_model.encode(
                self.documents,
                normalize_embeddings=True
            )
        
        # Save to storage
        self.save_documents_and_embeddings()
        
        process_time = time.time() - start_time
        logger.info(f"Documents added in {process_time:.2f} seconds")
        
        return {
            "message": "Documents added successfully", 
            "count": len(self.documents),
            "processing_time": process_time
        }
    
    def retrieve(self, query: str) -> tuple:
        """Retrieve relevant documents with their similarity scores"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        )
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            self.document_embeddings
        )[0]
        
        # Get top k documents with scores above threshold
        indices = np.argsort(similarities)[-self.k:][::-1]
        
        # Filter by threshold
        filtered_docs = []
        filtered_scores = []
        
        for idx in indices:
            score = similarities[idx]
            if score >= self.threshold:
                filtered_docs.append(self.documents[idx])
                filtered_scores.append(score)
        
        return filtered_docs, filtered_scores
    
    def direct_response(self, query: str, contexts: List[str], scores: List[float]) -> str:
        """Create a direct response from retrieved documents without using a model"""
        if not contexts:
            return "Lo siento, no tengo suficiente información para responder a esta pregunta."
        
        # Check for question patterns
        query_lower = query.lower()
        
        # Handle definition questions
        if any(x in query_lower for x in ["qué es", "que es", "definición", "definicion", "explica", "describe"]):
            if scores[0] > 0.9:
                return contexts[0]
            
            # Combine information
            response = contexts[0]
            if len(contexts) > 1:
                additional_info = ". ".join([c for c in contexts[1:] if c != contexts[0]])
                if additional_info:
                    response += f" {additional_info}"
            return response
        
        # Handle how-to/process questions
        elif any(x in query_lower for x in ["cómo", "como", "funciona", "cuál es el proceso", "cual es el proceso"]):
            if scores[0] > 0.85:
                return contexts[0]
            return f"{contexts[0]}. " + ". ".join([c for c in contexts[1:] if c != contexts[0]])
        
        # Handle purpose/utility questions
        elif any(x in query_lower for x in ["para qué", "para que", "cuál es el propósito", "cual es el proposito"]):
            if scores[0] > 0.85:
                return contexts[0]
            return f"{contexts[0]}. " + ". ".join([c for c in contexts[1:] if c != contexts[0]])
        
        # Default approach - combine information from all contexts
        return ". ".join(contexts)
    
    def _get_prompt_for_qwen(self, contexts: List[str], query: str) -> str:
        """Get the appropriate prompt template for Qwen model"""
        # Format the context text
        context_text = "\n".join([f"- {ctx}" for ctx in contexts])
        
        # Create prompt for Qwen model
        return f"""<|im_start|>system
Eres un asistente preciso que responde preguntas basándose únicamente en la información proporcionada.
<|im_end|>

<|im_start|>user
Contexto:
{context_text}

Basándote exclusivamente en el contexto anterior, responde a esta pregunta: {query}
<|im_end|>

<|im_start|>assistant
"""
    
    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """Extract the actual answer from the generated text"""
        if "<|im_start|>assistant" in generated_text:
            answer = generated_text.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in answer:
                answer = answer.split("<|im_end|>")[0].strip()
        else:
            answer = generated_text.replace(prompt, "").strip()
        
        return answer
    
    def is_good_response(self, response: str, query: str, contexts: List[str]) -> bool:
        """Check if a generated response is of good quality"""
        # Check for empty or very short responses
        if not response or len(response) < 15:
            return False
        
        # Check if response is just repeating the question
        if query.lower() in response.lower():
            return False
        
        # Check for model control tokens
        bad_markers = ["<|", "|>", "endoftext", "</s>", "[INST]", "[/INST]", "<s>", 
                      "usuario:", "assistente:", "humano:", "<pad>", "<|im_"]
        if any(marker in response.lower() for marker in bad_markers):
            return False
            
        # Check for content relevance - should contain key terms from contexts
        key_terms = set()
        for context in contexts:
            words = context.lower().split()
            key_terms.update([word for word in words if len(word) > 5])
        
        # Count how many key terms from the context appear in the response
        term_matches = sum(1 for term in key_terms if term in response.lower())
        
        # If we have key terms but almost none appear in the response, it might be off-topic
        if key_terms and term_matches < min(3, len(key_terms) / 8):
            return False
        
        return True
    
    def generate(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to a query using the Qwen model with a fixed relevance_boost of 0.7.
        
        Args:
            query: The user's query
            
        Returns:
            Dict with generation results
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        contexts, scores = self.retrieve(query)
        
        if not contexts:
            return {
                "query": query,
                "response": "Lo siento, no tengo suficiente información para responder a esta pregunta.",
                "contexts": [],
                "scores": [],
                "response_type": "no_context",
                "generation_time": time.time() - start_time
            }
        
        # Generation parameters - fixed with relevance_boost=0.7
        params = {
            "temperature": 0.3,
            "max_tokens": 150
        }
        
        try:
            # Get appropriate prompt with high relevance emphasis
            prompt = self._get_prompt_for_qwen(contexts, query)
            
            # Add additional instruction to maintain factuality due to high relevance_boost
            prompt = prompt.replace("system\nEres", "system\nEs muy importante que tu respuesta se base exclusivamente en la información proporcionada y no añadas datos que no estén presentes en el contexto. Eres")
            
            # Generate response
            outputs = self.generator(
                prompt,
                max_new_tokens=params["max_tokens"],
                do_sample=True,
                temperature=params["temperature"],
                top_p=0.92,
                top_k=40,
                repetition_penalty=1.2,
                num_return_sequences=1,
                return_full_text=True
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Extract the answer
            answer = self._extract_answer(generated_text, prompt)
            
            # Check if the response is good
            is_good = self.is_good_response(answer, query, contexts)
            
            if not is_good:
                # If not good, fallback to direct retrieval with high relevance boost
                direct_response = self.direct_response(query, contexts, scores)
                
                # Create a hybrid response for better results
                return {
                    "query": query,
                    "response": direct_response,
                    "contexts": contexts,
                    "scores": [float(s) for s in scores],
                    "response_type": "retrieval_fallback",
                    "generation_time": time.time() - start_time
                }
            
            return {
                "query": query,
                "response": answer,
                "contexts": contexts,
                "scores": [float(s) for s in scores],
                "response_type": "generation",
                "generation_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to direct retrieval
            return {
                "query": query,
                "response": self.direct_response(query, contexts, scores),
                "contexts": contexts,
                "scores": [float(s) for s in scores],
                "response_type": "error_fallback",
                "error": str(e),
                "generation_time": time.time() - start_time
            }

# Initialize the Spanish RAG system
rag_system = None

def initialize_rag():
    """Initialize the RAG system"""
    global rag_system
    if rag_system is None:
        logger.info("Initializing Spanish RAG system")
        rag_system = EnhancedSpanishRAG(
            embedding_model_name="intfloat/multilingual-e5-small",
            k=3,
            threshold=0.7,
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            relevance_boost=0.7
        )
        logger.info("Spanish RAG system initialized")

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_rag()

# Model for document upload requests
class UploadRequest(BaseModel):
    """Request model for document uploads"""
    texts: list[str]

# Model for generation requests
class GenerationRequest(BaseModel):
    """Request model for text generation"""
    query: str

@app.post("/upload")
async def upload_documents(req: UploadRequest):
    """
    Endpoint to add documents to the knowledge base
    """
    global rag_system
    
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="No texts were received for indexing.")
        
        # Initialize if not already done
        if rag_system is None:
            initialize_rag()
        
        # Add documents
        result = rag_system.add_documents(req.texts)
        
        return result
    except Exception as e:
        logger.error(f"Error while adding documents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error while adding documents: {e}"
        )

@app.post("/generate")
async def generate_text(req: GenerationRequest):
    """
    Endpoint to generate responses using the Spanish RAG system
    """
    global rag_system
    
    try:
        if not req.query:
            raise HTTPException(status_code=400, detail="No query was provided.")
        
        # Initialize if not already done
        if rag_system is None:
            initialize_rag()
        
        # Generate response
        result = rag_system.generate(req.query)
        
        return result
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error during generation: {e}"
        )

@app.get("/")
def read_root():
    """
    Root endpoint to check API status
    """
    # Check and report directory status
    dir_status = "exists" if os.path.exists(INDEX_STORAGE_DIR) else "does not exist"
    try:
        dir_contents = os.listdir(INDEX_STORAGE_DIR) if os.path.exists(INDEX_STORAGE_DIR) else []
    except:
        dir_contents = ["error listing contents"]
    
    return {
        "message": "Spanish RAG API is running successfully",
        "index_storage_dir": INDEX_STORAGE_DIR,
        "dir_status": dir_status,
        "dir_contents": dir_contents,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "relevance_boost": 0.7
    }

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request to {request.url.path} took {process_time:.4f}s")
    return response

@app.get("/debug/numpy")
def debug_numpy():
    try:
        import numpy as np
        return {"numpy_version": np.__version__}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)