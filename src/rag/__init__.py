"""
Enterprise RAG Knowledge Base and Vector Search Package.
"""
from src.config import get_logger

logger = get_logger(__name__)

from src.rag.config import (
    DATABASE_URL,
    USE_POSTGRES_KB,
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DOMAINS_DATA_DIR,
    CHROMA_DATA_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_TOP_K,
    CHROMA_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    DEFAULT_RAG_LLM_MODEL,
    RAG_LLM_PRIORITY_MODELS,
    RAG_SYSTEM_PROMPT,
    RAG_CONTEXT_TEMPLATE,
)
from src.rag.schema import (
    TableSchema,
    RAGQueryRequest,
    RAGSchemaResponse,
)
from src.rag.models import (
    Base,
    DomainModel,
    KnowledgeTableModel,
    KnowledgeColumnModel,
    TableDependencyModel,
    KnowledgeSegmentModel,
    SegmentVersionModel,
    EmbeddingJobModel,
    SegmentVectorModel,
)
from src.rag.database import (
    get_engine,
    get_session,
    get_db,
    is_db_connected,
    init_db,
    KnowledgeBaseRepository,
    JSONKnowledgeBaseMigrator,
    generate_stable_segment_uuid,
    KnowledgeChunk,
    build_domain_chunk,
    build_table_chunk,
    build_column_chunk,
    build_relationship_chunk,
)
from src.rag.service import (
    retrieve_erp_schema,
    generate_rag_llm,
    resolve_best_rag_model,
    get_rag_service,
    load_erp_registry,
    ERPEnterpriseRAGService,
    build_rag_augmented_prompt,
    get_embedding_model,
    get_chroma_engine,
    ChromaSchemaEngine,
    ChromaPostgresSynchronizer,
    get_embedding_synchronizer,
    sync_chroma_from_postgres,
    auto_sync_chroma,
)

__all__ = [
    # Core RAG Service
    "retrieve_erp_schema",
    "generate_rag_llm",
    "resolve_best_rag_model",
    "get_rag_service",
    "load_erp_registry",
    "ERPEnterpriseRAGService",
    # Prompts
    "RAG_SYSTEM_PROMPT",
    "RAG_CONTEXT_TEMPLATE",
    "build_rag_augmented_prompt",
    # ChromaDB Vector Engine & Embeddings
    "get_chroma_engine",
    "ChromaSchemaEngine",
    "get_embedding_model",
    "ChromaPostgresSynchronizer",
    "get_embedding_synchronizer",
    "sync_chroma_from_postgres",
    "auto_sync_chroma",
    # Database & DAL
    "get_engine",
    "get_session",
    "get_db",
    "is_db_connected",
    "init_db",
    "KnowledgeBaseRepository",
    "JSONKnowledgeBaseMigrator",
    # Chunking
    "generate_stable_segment_uuid",
    "KnowledgeChunk",
    "build_domain_chunk",
    "build_table_chunk",
    "build_column_chunk",
    "build_relationship_chunk",
    # SQLAlchemy Models
    "Base",
    "DomainModel",
    "KnowledgeTableModel",
    "KnowledgeColumnModel",
    "TableDependencyModel",
    "KnowledgeSegmentModel",
    "SegmentVersionModel",
    "EmbeddingJobModel",
    "SegmentVectorModel",
    # Pydantic Schemas
    "TableSchema",
    "RAGQueryRequest",
    "RAGSchemaResponse",
    # Configuration
    "DATABASE_URL",
    "USE_POSTGRES_KB",
    "DB_USER",
    "DB_PASSWORD",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "CHROMA_DATA_DIR",
    "CHROMA_COLLECTION_NAME",
    "DOMAINS_DATA_DIR",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIMENSION",
    "CHROMA_TOP_K",
    "CHROMA_SIMILARITY_THRESHOLD",
    "DEFAULT_RAG_LLM_MODEL",
    "RAG_LLM_PRIORITY_MODELS",
]
