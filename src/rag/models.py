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

from src.config import get_logger

logger = get_logger(__name__)

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
        try:
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
        except Exception as e:
            logger.error(f"Error serializing DomainModel to dict: {e}")
            return {"id": str(self.id), "domain_id": getattr(self, "domain_id", "")}


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
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("domain_id", "table_name", name="uq_knowledge_tables_domain_table"),
    )

    # Relationships
    domain = relationship("DomainModel", back_populates="tables")
    columns = relationship("KnowledgeColumnModel", back_populates="table", cascade="all, delete-orphan")
    segments = relationship("KnowledgeSegmentModel", back_populates="table")

    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                "id": str(self.id),
                "domain_id": str(self.domain_id),
                "table_name": self.table_name,
                "description": self.description,
                "keywords": self.keywords or [],
                "sql_definition": self.sql_definition,
                "is_active": self.is_active,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            }
        except Exception as e:
            logger.error(f"Error serializing KnowledgeTableModel to dict: {e}")
            return {"id": str(self.id), "table_name": getattr(self, "table_name", "")}


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
        try:
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
        except Exception as e:
            logger.error(f"Error serializing KnowledgeColumnModel to dict: {e}")
            return {"id": str(self.id), "column_name": getattr(self, "column_name", "")}


# ---------------------------------------------------------------------
# D. KnowledgeSegmentModel: knowledge_segments table (Hybrid Chunks)
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
    segment_key = Column(String(256), nullable=True, index=True)  # Business key: domain_id:table_id:segment_name
    content = Column(Text, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    domain = relationship("DomainModel", back_populates="segments")
    table = relationship("KnowledgeTableModel", back_populates="segments")

    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                "id": str(self.id),
                "domain_id": str(self.domain_id),
                "table_id": str(self.table_id) if self.table_id else None,
                "section": self.section,
                "topic": self.topic,
                "segment_type": self.segment_type,
                "segment_name": self.segment_name,
                "segment_key": self.segment_key,
                "content": self.content,
                "version": self.version,
                "is_active": self.is_active,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            }
        except Exception as e:
            logger.error(f"Error serializing KnowledgeSegmentModel to dict: {e}")
            return {"id": str(self.id), "segment_name": getattr(self, "segment_name", "")}

