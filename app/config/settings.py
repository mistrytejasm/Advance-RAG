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