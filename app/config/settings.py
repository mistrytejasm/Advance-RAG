import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = "data/uploads"
PROCESSED_DIR = "data/processed"

ALLOWED_EXTENSION = [
    ".pdf",
    ".docx",
    ".png",
    ".jpg"
]

MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "rag_db"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40

# --- Phase 3: Embedding & Vector DB ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-index"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 64    # chunks sent to model at once
PINECONE_BATCH_SIZE = 100    # vectors upserted to Pinecone at once

# --- Phase 4: Retrieval Pipeline ---
TOP_K = 20                    # vectors fetched from Pinecone per query
RERANK_TOP_K = 5              # final results returned after reranking
SIMILARITY_THRESHOLD = 0.6    # drop any vector result below this score
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # CPU-safe, fast

# --- Phase 5: Hybrid Search (Semantic + BM25) ---
HYBRID_INDEX_NAME = "hybrid_search_index"   # Atlas Search index name
VECTOR_WEIGHT = 0.7           # weight for semantic (Pinecone) score
BM25_WEIGHT = 0.3             # weight for keyword (BM25) score
BM25_TOP_K = 20               # candidates to fetch from Atlas BM25 search

# --- Phase 6: Query Understanding ---
# Weights used when QueryRouter selects the BM25_PRIORITY route
# (navigational queries: "which weeks cover X", "list all topics on Y")
VECTOR_WEIGHT_BM25_PRIORITY = 0.4
BM25_WEIGHT_BM25_PRIORITY = 0.6
# HYBRID_FILTERED route: multiply top_k by this factor so that page/section
# specific chunks are more likely to appear before the metadata filter runs.
# Example: top_k=20 → 20*4 = 80 candidates fetched from Pinecone/BM25.
FILTERED_TOP_K_MULTIPLIER = 4

# --- Phase 7: Answer Generation (LLM via Groq) ---
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.1"))   # Low = factual, deterministic
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))  # Max LLM response tokens
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))   # Hard request timeout
LLM_MAX_RETRIES     = int(os.getenv("LLM_MAX_RETRIES", "3"))        # Retries on transient errors
LLM_RETRY_DELAY     = float(os.getenv("LLM_RETRY_DELAY", "1.5"))    # Seconds between retries
# Maximum tokens fed to the LLM as retrieved context.
# Using tiktoken cl100k_base as a conservative approximation.
MAX_CONTEXT_TOKENS  = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
# Maximum number of chunks to include in context (hard cap, token budget wins first)
MAX_CONTEXT_CHUNKS  = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
# Minimum confidence proxy (avg rerank_score) to attempt LLM generation.
# Below this, return "insufficient information" without calling the LLM.
MIN_ANSWER_CONFIDENCE = float(os.getenv("MIN_ANSWER_CONFIDENCE", "0.0"))
# The exact phrase the LLM should return when context doesn't have the answer.
LLM_NO_ANSWER_PHRASE = "I don't have enough information in the provided context to answer this question."