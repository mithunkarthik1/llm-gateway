from src.rag.service import (
    retrieve_erp_schema,
    generate_rag_llm,
    resolve_best_rag_model,
    get_rag_service,
    load_erp_registry,
    ERPEnterpriseRAGService,
    RAG_SYSTEM_PROMPT,
    RAG_CONTEXT_TEMPLATE,
    build_rag_augmented_prompt,
)
from src.rag.faiss_service import (
    get_faiss_engine,
    FAISSSchemaEngine,
)
from src.rag.schema import (
    TableSchema,
    RAGQueryRequest,
    RAGSchemaResponse,
)
from src.rag.config import (
    FAISS_INDEX_DIR,
    DOMAINS_DATA_DIR,
    EMBEDDING_MODEL_NAME,
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
    # FAISS In-Memory Vector Engine
    "get_faiss_engine",
    "FAISSSchemaEngine",
    # Pydantic Schemas
    "TableSchema",
    "RAGQueryRequest",
    "RAGSchemaResponse",
    # Configuration
    "FAISS_INDEX_DIR",
    "DOMAINS_DATA_DIR",
    "EMBEDDING_MODEL_NAME",
]
