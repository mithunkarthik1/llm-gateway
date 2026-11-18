"""
PostgreSQL Knowledge Base & FAISS Integration package for Enterprise ERP RAG.
"""
from src.rag.db.database import get_db, init_db, get_engine
from src.rag.db.models import (
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

__all__ = [
    "get_db",
    "init_db",
    "get_engine",
    "Base",
    "DomainModel",
    "KnowledgeTableModel",
    "KnowledgeColumnModel",
    "TableDependencyModel",
    "KnowledgeSegmentModel",
    "SegmentVersionModel",
    "EmbeddingJobModel",
    "SegmentVectorModel",
]
