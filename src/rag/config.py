import os
from pathlib import Path

# RAG Data Directory Structure (Format v2 Consolidated Primary Architecture)
RAG_DIR = Path(__file__).resolve().parent
DOMAINS_DATA_DIR = RAG_DIR / "data" / os.getenv("DOMAINS_DATA_FOLDER", "domains")
FAISS_INDEX_DIR = RAG_DIR / "data" / os.getenv("FAISS_INDEX_FOLDER", "faiss_index")
CHROMA_DATA_DIR = RAG_DIR / "data" / os.getenv("CHROMA_DATA_FOLDER", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "nexus_erp_knowledge")
VECTOR_DB_BACKEND = os.getenv("VECTOR_DB_BACKEND", "both").lower()

# PostgreSQL Knowledge Base Connection Settings (Dynamic from .env)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "db_nexus_rag")

if DB_PASSWORD:
    _DYNAMIC_DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    _DYNAMIC_DB_URL = f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("PGVECTOR_CONNECTION_STRING") or _DYNAMIC_DB_URL
USE_POSTGRES_KB = os.getenv("USE_POSTGRES_KB", "true").lower() in ("true", "1", "yes")

# FAISS Search Parameters
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "1"))
FAISS_SIMILARITY_THRESHOLD = float(os.getenv("FAISS_SIMILARITY_THRESHOLD", "0.20"))

# Local Embedding Model (HuggingFace SentenceTransformer, runs 100% locally on CPU)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# RAG LLM Models (Flagship Meta LLaMA with zero-downtime fallback chain)
DEFAULT_RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama-3.3-70b-versatile")
_PRIORITY_ENV = os.getenv("RAG_LLM_PRIORITY_MODELS", "")
if _PRIORITY_ENV:
    RAG_LLM_PRIORITY_MODELS = [m.strip() for m in _PRIORITY_ENV.split(",") if m.strip()]
else:
    RAG_LLM_PRIORITY_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "allam-2-7b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b"
    ]
