import os
from pathlib import Path
from typing import Dict, List, Optional
import src.config  # Automatically loads .env into os.environ

# =============================================================================
# PostgreSQL Knowledge Base Connection Settings (Dynamic from .env)
# =============================================================================
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

# If running inside Docker container and pointing to host localhost, bridge via host.docker.internal
IS_DOCKER = os.path.exists("/.dockerenv") or os.getenv("IS_DOCKER", "false").lower() in ("true", "1")
if IS_DOCKER and DATABASE_URL:
    if "@localhost:" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("@localhost:", "@host.docker.internal:")
    elif "@127.0.0.1:" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("@127.0.0.1:", "@host.docker.internal:")

USE_POSTGRES_KB = os.getenv("USE_POSTGRES_KB", "true").lower() in ("true", "1", "yes")

# Connection Pool Settings
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))

# =============================================================================
# RAG Data Directory Structure (PostgreSQL Primary + ChromaDB Vector Engine)
# =============================================================================
RAG_DIR = Path(__file__).resolve().parent
DOMAINS_DATA_DIR = RAG_DIR / "data" / os.getenv("DOMAINS_DATA_FOLDER", "domains")
CHROMA_DATA_DIR = RAG_DIR / "data" / os.getenv("CHROMA_DATA_FOLDER", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "nexus_erp_knowledge")
CHROMA_TOP_K = int(os.getenv("CHROMA_TOP_K", "5"))
CHROMA_SIMILARITY_THRESHOLD = float(os.getenv("CHROMA_SIMILARITY_THRESHOLD", "0.20"))

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

# =============================================================================
# RAG Prompts
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an Enterprise ERP Database Architect specializing in PostgreSQL and MySQL relational schemas.
You have access to a validated, production-grade enterprise schema knowledge base.

Rules:
1. Adhere strictly to the domain relationships, primary/foreign key hierarchies, and column data types provided in the retrieved schema context.
2. Preserve double-entry accounting integrity, financial cutoffs, audit trails, and referential integrity.
3. When extending or modifying the schema, maintain topological ordering so parent tables precede dependent child tables.
4. Output clean, syntactically correct SQL DDL with appropriate constraints and indices.
"""

RAG_CONTEXT_TEMPLATE = """### RETRIEVED ENTERPRISE ERP SCHEMA CONTEXT:
Domain: {domain_name} ({domain_id})
Available Tables: {tables}

{schema_ddl}

### ACTIVE SCHEMA SUMMARY LEDGER:
{summary}

### USER INSTRUCTION:
{user_query}

Generate the exact SQL schema DDL satisfying the instruction while respecting the architectural standards and relationships above.
"""

# =====================================================================
# Domain Priorities & Canonical Synonyms (Imported from utils)
# =====================================================================
from utils.domain_priority import (
    DOMAIN_PRIMARY_TABLES,
    TABLE_PRIMARY_COLUMNS_MAP,
)
from utils.schema_synonyms import (
    TABLE_SYNONYMS,
    NONEXISTENT_TABLE_PATTERNS,
)

__all__ = [
    "DATABASE_URL",
    "USE_POSTGRES_KB",
    "DB_POOL_SIZE",
    "DB_MAX_OVERFLOW",
    "DB_POOL_TIMEOUT",
    "DB_POOL_RECYCLE",
    "RAG_DIR",
    "DOMAINS_DATA_DIR",
    "CHROMA_DATA_DIR",
    "CHROMA_COLLECTION_NAME",
    "CHROMA_TOP_K",
    "CHROMA_SIMILARITY_THRESHOLD",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIMENSION",
    "DEFAULT_RAG_LLM_MODEL",
    "RAG_LLM_PRIORITY_MODELS",
    "RAG_SYSTEM_PROMPT",
    "RAG_CONTEXT_TEMPLATE",
    "DOMAIN_PRIMARY_TABLES",
    "TABLE_PRIMARY_COLUMNS_MAP",
    "TABLE_SYNONYMS",
    "NONEXISTENT_TABLE_PATTERNS",
]
