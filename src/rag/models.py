import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    BigInteger,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    JSON,
    func
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> uuid.UUID:
    return uuid.uuid4()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


UUIDType = PG_UUID(as_uuid=True).with_variant(String(36), "sqlite")


# ---------------------------------------------------------------------
# A. DomainModel: domains table
# ---------------------------------------------------------------------
class DomainModel(Base):
    __tablename__ = "domains"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    domain_id = Column(String(64), unique=True, nullable=False, index=True)
    system_name = Column(String(128), nullable=False, default="Enterprise ERP")
    domain_name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    keywords = Column(JSON, default=list)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    tables = relationship("KnowledgeTableModel", back_populates="domain", cascade="all, delete-orphan")
    segments = relationship("KnowledgeSegmentModel", back_populates="domain", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "domain_id": self.domain_id,
            "system_name": self.system_name,
            "domain_name": self.domain_name,
            "description": self.description,
            "keywords": self.keywords or [],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ---------------------------------------------------------------------
# B. KnowledgeTableModel: knowledge_tables table
# ---------------------------------------------------------------------
class KnowledgeTableModel(Base):
    __tablename__ = "knowledge_tables"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    domain_id = Column(UUIDType, ForeignKey("domains.id", ondelete="CASCADE"), nullable=False, index=True)
    table_name = Column(String(128), nullable=False, index=True)
    description = Column(Text, nullable=True)
    keywords = Column(JSON, default=list)
    sql_definition = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("domain_id", "table_name", name="uq_knowledge_tables_domain_table"),
    )

    # Relationships
    domain = relationship("DomainModel", back_populates="tables")
    columns = relationship("KnowledgeColumnModel", back_populates="table", cascade="all, delete-orphan")
    outgoing_dependencies = relationship(
        "TableDependencyModel",
        foreign_keys="TableDependencyModel.source_table_id",
        back_populates="source_table",
        cascade="all, delete-orphan"
    )
    incoming_dependencies = relationship(
        "TableDependencyModel",
        foreign_keys="TableDependencyModel.referenced_table_id",
        back_populates="referenced_table",
        cascade="all, delete-orphan"
    )
    segments = relationship("KnowledgeSegmentModel", back_populates="table")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "domain_id": str(self.domain_id),
            "table_name": self.table_name,
            "description": self.description,
            "keywords": self.keywords or [],
            "sql_definition": self.sql_definition,
            "content_hash": self.content_hash,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ---------------------------------------------------------------------
# C. KnowledgeColumnModel: knowledge_columns table
# ---------------------------------------------------------------------
class KnowledgeColumnModel(Base):
    __tablename__ = "knowledge_columns"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    table_id = Column(UUIDType, ForeignKey("knowledge_tables.id", ondelete="CASCADE"), nullable=False, index=True)
    column_name = Column(String(128), nullable=False, index=True)
    data_type = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    is_primary_key = Column(Boolean, nullable=False, default=False, index=True)
    is_foreign_key = Column(Boolean, nullable=False, default=False, index=True)
    references_table = Column(String(128), nullable=True)
    references_column = Column(String(128), nullable=True)
    keywords = Column(JSON, default=list)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("table_id", "column_name", name="uq_knowledge_columns_table_col"),
    )

    # Relationships
    table = relationship("KnowledgeTableModel", back_populates="columns")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "table_id": str(self.table_id),
            "column_name": self.column_name,
            "data_type": self.data_type,
            "description": self.description,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "references_table": self.references_table,
            "references_column": self.references_column,
            "keywords": self.keywords or [],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ---------------------------------------------------------------------
# D. TableDependencyModel: table_dependencies table
# ---------------------------------------------------------------------
class TableDependencyModel(Base):
    __tablename__ = "table_dependencies"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    source_table_id = Column(UUIDType, ForeignKey("knowledge_tables.id", ondelete="CASCADE"), nullable=False, index=True)
    referenced_table_id = Column(UUIDType, ForeignKey("knowledge_tables.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_type = Column(String(64), nullable=False, default="FOREIGN_KEY")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("source_table_id", "referenced_table_id", "relationship_type", name="uq_table_dependencies"),
    )

    # Relationships
    source_table = relationship("KnowledgeTableModel", foreign_keys=[source_table_id], back_populates="outgoing_dependencies")
    referenced_table = relationship("KnowledgeTableModel", foreign_keys=[referenced_table_id], back_populates="incoming_dependencies")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "source_table_id": str(self.source_table_id),
            "referenced_table_id": str(self.referenced_table_id),
            "relationship_type": self.relationship_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------
# E. KnowledgeSegmentModel: knowledge_segments table (Hybrid Chunks)
# ---------------------------------------------------------------------
class KnowledgeSegmentModel(Base):
    __tablename__ = "knowledge_segments"

    id = Column(UUIDType, primary_key=True)  # Stable UUID (retained across edits)
    domain_id = Column(UUIDType, ForeignKey("domains.id", ondelete="CASCADE"), nullable=False, index=True)
    table_id = Column(UUIDType, ForeignKey("knowledge_tables.id", ondelete="SET NULL"), nullable=True, index=True)
    section = Column(String(128), nullable=False, index=True)
    topic = Column(String(128), nullable=False, index=True)
    segment_type = Column(String(64), nullable=False, index=True)
    segment_name = Column(String(256), nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    domain = relationship("DomainModel", back_populates="segments")
    table = relationship("KnowledgeTableModel", back_populates="segments")
    versions = relationship("SegmentVersionModel", back_populates="segment", cascade="all, delete-orphan")
    embedding_jobs = relationship("EmbeddingJobModel", back_populates="segment", cascade="all, delete-orphan")
    vectors = relationship("SegmentVectorModel", back_populates="segment", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "domain_id": str(self.domain_id),
            "table_id": str(self.table_id) if self.table_id else None,
            "section": self.section,
            "topic": self.topic,
            "segment_type": self.segment_type,
            "segment_name": self.segment_name,
            "content": self.content,
            "content_hash": self.content_hash,
            "version": self.version,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ---------------------------------------------------------------------
# F. SegmentVersionModel: segment_versions table
# ---------------------------------------------------------------------
class SegmentVersionModel(Base):
    __tablename__ = "segment_versions"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    segment_id = Column(UUIDType, ForeignKey("knowledge_segments.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    changed_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("segment_id", "version", name="uq_segment_versions_seg_ver"),
    )

    # Relationships
    segment = relationship("KnowledgeSegmentModel", back_populates="versions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "segment_id": str(self.segment_id),
            "version": self.version,
            "content": self.content,
            "content_hash": self.content_hash,
            "changed_at": self.changed_at.isoformat() if self.changed_at else None,
        }


# ---------------------------------------------------------------------
# G. EmbeddingJobModel: embedding_jobs table
# ---------------------------------------------------------------------
class EmbeddingJobModel(Base):
    __tablename__ = "embedding_jobs"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    segment_id = Column(UUIDType, ForeignKey("knowledge_segments.id", ondelete="CASCADE"), nullable=False, index=True)
    content_hash = Column(String(64), nullable=False)
    operation = Column(String(32), nullable=False)  # INSERT, UPDATE, DELETE
    status = Column(String(32), nullable=False, default="PENDING", index=True)  # PENDING, PROCESSING, COMPLETED, FAILED
    retry_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    segment = relationship("KnowledgeSegmentModel", back_populates="embedding_jobs")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "segment_id": str(self.segment_id),
            "content_hash": self.content_hash,
            "operation": self.operation,
            "status": self.status,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


# ---------------------------------------------------------------------
# H. SegmentVectorModel: segment_vectors table
# ---------------------------------------------------------------------
class SegmentVectorModel(Base):
    __tablename__ = "segment_vectors"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    segment_id = Column(UUIDType, ForeignKey("knowledge_segments.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding_model = Column(String(128), nullable=False)
    model_version = Column(String(64), nullable=False, default="v1")
    faiss_vector_id = Column(BigInteger, nullable=False, index=True)
    content_hash = Column(String(64), nullable=False)
    indexed_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("embedding_model", "segment_id", name="uq_segment_vectors_model_seg"),
        UniqueConstraint("embedding_model", "faiss_vector_id", name="uq_segment_vectors_faiss_id"),
    )

    # Relationships
    segment = relationship("KnowledgeSegmentModel", back_populates="vectors")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "segment_id": str(self.segment_id),
            "embedding_model": self.embedding_model,
            "model_version": self.model_version,
            "faiss_vector_id": self.faiss_vector_id,
            "content_hash": self.content_hash,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
        }
