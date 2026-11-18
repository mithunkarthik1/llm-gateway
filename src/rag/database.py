import hashlib
import json
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from sqlalchemy import (
    create_engine,
    text,
    select,
    update,
    delete,
    and_,
    or_,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool, StaticPool

from src.config import get_logger
from src.rag.config import (
    DATABASE_URL,
    DB_POOL_SIZE,
    DB_MAX_OVERFLOW,
    DB_POOL_TIMEOUT,
    DB_POOL_RECYCLE,
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
    generate_uuid,
    utc_now,
)

logger = get_logger(__name__)

# =============================================================================
# Database Engine & Session Management
# =============================================================================

_ENGINE = None
_SESSION_FACTORY = None


def get_engine(db_url: Optional[str] = None):
    """
    Returns a configured SQLAlchemy engine singleton.
    Supports PostgreSQL with connection pooling and SQLite for test isolation.
    """
    global _ENGINE, _SESSION_FACTORY
    target_url = db_url or DATABASE_URL

    if _ENGINE is None or (db_url and str(_ENGINE.url) != db_url):
        if target_url.startswith("sqlite"):
            _ENGINE = create_engine(
                target_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            _ENGINE = create_engine(
                target_url,
                poolclass=QueuePool,
                pool_size=DB_POOL_SIZE,
                max_overflow=DB_MAX_OVERFLOW,
                pool_timeout=DB_POOL_TIMEOUT,
                pool_recycle=DB_POOL_RECYCLE,
                pool_pre_ping=True
            )
        _SESSION_FACTORY = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=_ENGINE)
        )
        logger.info(f"Initialized Database engine for: {_ENGINE.url.render_as_string(hide_password=True)}")

    return _ENGINE


def get_session(db_url: Optional[str] = None) -> Session:
    """Returns a new database session."""
    get_engine(db_url)
    return _SESSION_FACTORY()


@contextmanager
def get_db(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Context manager providing a transactional database session with
    automatic commit on success and rollback on exceptions.
    """
    session = get_session(db_url)
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rolled back due to error: {e}")
        raise
    finally:
        session.close()


def is_db_connected(db_url: Optional[str] = None) -> bool:
    """Verifies live database connectivity."""
    try:
        engine = get_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        return True
    except Exception as e:
        logger.warning(f"Database connectivity check failed: {e}")
        return False


def init_db(db_url: Optional[str] = None, drop_all: bool = False):
    """Initializes all database tables from models metadata."""
    engine = get_engine(db_url)
    if drop_all:
        logger.warning("Dropping all existing knowledge base tables...")
        Base.metadata.drop_all(bind=engine)
    logger.info("Creating knowledge base tables if not present...")
    Base.metadata.create_all(bind=engine)
    logger.info("Knowledge base tables initialized successfully.")


# =============================================================================
# Chunking & Hashing Utilities
# =============================================================================

SEGMENT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def compute_sha256(text: str) -> str:
    """Computes normalized SHA-256 hash of text content."""
    clean = text.strip().encode("utf-8")
    return hashlib.sha256(clean).hexdigest()


def generate_stable_segment_uuid(domain_id: str, topic: str, segment_name: str) -> uuid.UUID:
    """
    Generates a deterministic, collision-free UUIDv5 for a logical segment identity.
    Ensures that a segment retains its exact UUID across multiple content updates.
    """
    identity_key = f"{domain_id.lower().strip()}:{topic.lower().strip()}:{segment_name.lower().strip()}"
    return uuid.uuid5(SEGMENT_NAMESPACE, identity_key)


class KnowledgeChunk:
    """Represents a structured hybrid chunk ready for DB persistence & vectorization."""

    def __init__(
        self,
        segment_uuid: uuid.UUID,
        domain_id: str,
        section: str,
        topic: str,
        segment_type: str,
        segment_name: str,
        content: str,
        table_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.segment_uuid = segment_uuid
        self.domain_id = domain_id
        self.section = section
        self.topic = topic
        self.segment_type = segment_type
        self.segment_name = segment_name
        self.content = content.strip()
        self.content_hash = compute_sha256(self.content)
        self.table_name = table_name
        self.metadata = metadata or {}


def build_domain_chunk(domain_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a domain-level hybrid segment."""
    dom_id = domain_data.get("domain_id", "").strip()
    dom_name = domain_data.get("domain_name", dom_id).strip()
    system = domain_data.get("system", "Enterprise ERP").strip()
    desc = domain_data.get("description", "").strip()
    keywords = domain_data.get("keywords", [])
    tbls = domain_data.get("table_names", [])

    section = dom_name
    topic = f"{dom_name} Overview"
    seg_name = f"{dom_id}.overview"
    seg_uuid = generate_stable_segment_uuid(dom_id, topic, seg_name)

    content = (
        f"Section: {section}\n"
        f"Topic: {topic}\n"
        f"Segment: {seg_name}\n\n"
        f"System: {system}\n"
        f"Domain ID: {dom_id}\n"
        f"Domain Name: {dom_name}\n"
        f"Description: {desc}\n"
        f"Master Tables: {', '.join(tbls)}\n"
        f"Keywords: {', '.join(keywords)}"
    )

    return KnowledgeChunk(
        segment_uuid=seg_uuid,
        domain_id=dom_id,
        section=section,
        topic=topic,
        segment_type="domain",
        segment_name=seg_name,
        content=content,
        metadata={
            "domain_id": dom_id,
            "domain_name": dom_name,
            "segment_type": "domain",
            "identity_key": f"{dom_id}:domain:overview",
            "table_count": len(tbls)
        }
    )


def build_table_chunk(domain_data: Dict[str, Any], table_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a table-level hybrid segment."""
    dom_id = domain_data.get("domain_id", "").strip()
    dom_name = domain_data.get("domain_name", dom_id).strip()
    system = domain_data.get("system", "Enterprise ERP").strip()
    tname = table_data.get("table_name", "").strip()
    desc = table_data.get("description", "").strip()
    keywords = table_data.get("keywords", [])
    deps = table_data.get("dependencies", [])
    cols = table_data.get("columns", [])
    raw_sql = table_data.get("sql", "").strip()

    section = dom_name
    topic = tname.replace("_", " ").title()
    seg_name = f"{tname}.table_definition"
    seg_uuid = generate_stable_segment_uuid(dom_id, topic, seg_name)

    content = (
        f"Section: {section}\n"
        f"Topic: {topic}\n"
        f"Segment: {seg_name}\n\n"
        f"System: {system}\n"
        f"Domain: {dom_name} ({dom_id})\n"
        f"Table: {tname}\n"
        f"Description: {desc}\n"
        f"Foreign Key Dependencies: {', '.join(deps) if deps else 'None'}\n"
        f"Columns: {', '.join(cols)}\n"
        f"Keywords: {', '.join(keywords)}\n\n"
        f"SQL Definition:\n{raw_sql}"
    )

    return KnowledgeChunk(
        segment_uuid=seg_uuid,
        domain_id=dom_id,
        section=section,
        topic=topic,
        segment_type="table",
        segment_name=seg_name,
        content=content,
        table_name=tname,
        metadata={
            "domain_id": dom_id,
            "domain_name": dom_name,
            "table_name": tname,
            "segment_type": "table",
            "identity_key": f"{dom_id}:{tname}:definition",
            "dependencies": deps
        }
    )


def build_column_chunk(domain_data: Dict[str, Any], column_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a granular column-level hybrid segment."""
    dom_id = domain_data.get("domain_id", "").strip()
    dom_name = domain_data.get("domain_name", dom_id).strip()

    tname = column_data.get("table_name", "").strip()
    cname = column_data.get("column_name", "").strip()
    dtype = column_data.get("data_type", "").strip()
    cdesc = column_data.get("description", "").strip()
    ckw = column_data.get("keywords", [])
    is_pk = column_data.get("is_primary_key", False)
    is_fk = column_data.get("is_foreign_key", False)
    ref = column_data.get("references")

    section = dom_name
    topic = tname.replace("_", " ").title()
    seg_name = f"{tname}.{cname}"
    seg_uuid = generate_stable_segment_uuid(dom_id, topic, seg_name)

    pk_str = "Yes" if is_pk else "No"
    fk_str = f"Yes (References: {ref})" if is_fk else "No"

    content = (
        f"Section: {section}\n"
        f"Topic: {topic}\n"
        f"Segment: {seg_name}\n\n"
        f"Table: {tname}\n"
        f"Column: {cname}\n"
        f"Data Type: {dtype}\n"
        f"Primary Key: {pk_str}\n"
        f"Foreign Key: {fk_str}\n"
        f"Description: {cdesc}\n"
        f"Keywords: {', '.join(ckw) if ckw else cname}"
    )

    return KnowledgeChunk(
        segment_uuid=seg_uuid,
        domain_id=dom_id,
        section=section,
        topic=topic,
        segment_type="column",
        segment_name=seg_name,
        content=content,
        table_name=tname,
        metadata={
            "domain_id": dom_id,
            "domain_name": dom_name,
            "table_name": tname,
            "column_name": cname,
            "data_type": dtype,
            "is_primary_key": is_pk,
            "is_foreign_key": is_fk,
            "segment_type": "column",
            "identity_key": f"{dom_id}:{tname}:{cname}"
        }
    )


def build_relationship_chunk(
    domain_data: Dict[str, Any],
    source_table: str,
    target_table: str,
    fk_col: str,
    rel_type: str = "FOREIGN_KEY"
) -> KnowledgeChunk:
    """Builds a relational foreign-key dependency hybrid segment."""
    dom_id = domain_data.get("domain_id", "").strip()
    dom_name = domain_data.get("domain_name", dom_id).strip()

    section = dom_name
    topic = source_table.replace("_", " ").title()
    seg_name = f"{source_table}.ref_{target_table}"
    seg_uuid = generate_stable_segment_uuid(dom_id, topic, seg_name)

    content = (
        f"Section: {section}\n"
        f"Topic: {topic}\n"
        f"Segment: {seg_name}\n\n"
        f"Source Table: {source_table}\n"
        f"Foreign Key Column: {fk_col}\n"
        f"Referenced Parent Table: {target_table}\n"
        f"Relationship Type: {rel_type}\n"
        f"Relational Rule: Table '{source_table}' directly depends on parent table '{target_table}' via key '{fk_col}'."
    )

    return KnowledgeChunk(
        segment_uuid=seg_uuid,
        domain_id=dom_id,
        section=section,
        topic=topic,
        segment_type="relationship",
        segment_name=seg_name,
        content=content,
        table_name=source_table,
        metadata={
            "domain_id": dom_id,
            "domain_name": dom_name,
            "source_table": source_table,
            "referenced_table": target_table,
            "fk_column": fk_col,
            "segment_type": "relationship",
            "identity_key": f"{dom_id}:{source_table}:ref_{target_table}"
        }
    )


# =============================================================================
# Unified Repository Data Access Layer
# =============================================================================

class KnowledgeBaseRepository:
    """
    Unified Data Access Layer for PostgreSQL ERP Knowledge Base.
    Provides fast, transactional, zero-N+1 query operations.
    """

    def __init__(self, session: Session):
        self.session = session

    # 1. Domains
    def get_domain_by_id(self, domain_uuid: uuid.UUID) -> Optional[DomainModel]:
        return self.session.query(DomainModel).filter(DomainModel.id == domain_uuid).first()

    def get_domain_by_business_id(self, domain_id: str, active_only: bool = True) -> Optional[DomainModel]:
        q = self.session.query(DomainModel).filter(DomainModel.domain_id == domain_id.lower().strip())
        if active_only:
            q = q.filter(DomainModel.is_active.is_(True))
        return q.first()

    def list_domains(self, active_only: bool = True) -> List[DomainModel]:
        q = self.session.query(DomainModel)
        if active_only:
            q = q.filter(DomainModel.is_active.is_(True))
        return q.order_by(DomainModel.domain_id).all()

    def upsert_domain(
        self,
        domain_id: str,
        domain_name: str,
        system_name: str = "Enterprise ERP",
        description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        is_active: bool = True
    ) -> Tuple[DomainModel, bool]:
        dom_id_clean = domain_id.lower().strip()
        existing = self.session.query(DomainModel).filter(DomainModel.domain_id == dom_id_clean).first()
        if existing:
            existing.domain_name = domain_name
            existing.system_name = system_name
            existing.description = description
            existing.keywords = keywords or []
            existing.is_active = is_active
            existing.updated_at = utc_now()
            return existing, False

        new_domain = DomainModel(
            domain_id=dom_id_clean,
            domain_name=domain_name,
            system_name=system_name,
            description=description,
            keywords=keywords or [],
            is_active=is_active,
        )
        self.session.add(new_domain)
        self.session.flush()
        return new_domain, True

    # 2. Knowledge Tables
    def get_table_by_id(self, table_uuid: uuid.UUID) -> Optional[KnowledgeTableModel]:
        return self.session.query(KnowledgeTableModel).filter(KnowledgeTableModel.id == table_uuid).first()

    def get_table_by_name(self, table_name: str, domain_id: Optional[str] = None) -> Optional[KnowledgeTableModel]:
        tname = table_name.lower().strip()
        q = self.session.query(KnowledgeTableModel).filter(KnowledgeTableModel.table_name == tname)
        if domain_id:
            dom = self.get_domain_by_business_id(domain_id, active_only=False)
            if dom:
                q = q.filter(KnowledgeTableModel.domain_id == dom.id)
        return q.first()

    def list_tables(self, domain_id: Optional[str] = None, active_only: bool = True) -> List[KnowledgeTableModel]:
        q = self.session.query(KnowledgeTableModel)
        if active_only:
            q = q.filter(KnowledgeTableModel.is_active.is_(True))
        if domain_id:
            dom = self.get_domain_by_business_id(domain_id, active_only=False)
            if dom:
                q = q.filter(KnowledgeTableModel.domain_id == dom.id)
            else:
                return []
        return q.order_by(KnowledgeTableModel.table_name).all()

    def upsert_table(
        self,
        domain_uuid: uuid.UUID,
        table_name: str,
        sql_definition: str,
        content_hash: str,
        description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        is_active: bool = True
    ) -> Tuple[KnowledgeTableModel, bool]:
        tname_clean = table_name.lower().strip()
        existing = self.session.query(KnowledgeTableModel).filter(
            KnowledgeTableModel.domain_id == domain_uuid,
            KnowledgeTableModel.table_name == tname_clean
        ).first()

        if existing:
            existing.sql_definition = sql_definition
            existing.content_hash = content_hash
            existing.description = description
            existing.keywords = keywords or []
            existing.is_active = is_active
            existing.updated_at = utc_now()
            return existing, False

        new_tbl = KnowledgeTableModel(
            domain_id=domain_uuid,
            table_name=tname_clean,
            sql_definition=sql_definition,
            content_hash=content_hash,
            description=description,
            keywords=keywords or [],
            is_active=is_active
        )
        self.session.add(new_tbl)
        self.session.flush()
        return new_tbl, True

    # 3. Knowledge Columns
    def list_columns_by_table(self, table_uuid: uuid.UUID, active_only: bool = True) -> List[KnowledgeColumnModel]:
        q = self.session.query(KnowledgeColumnModel).filter(KnowledgeColumnModel.table_id == table_uuid)
        if active_only:
            q = q.filter(KnowledgeColumnModel.is_active.is_(True))
        return q.order_by(KnowledgeColumnModel.id).all()

    def upsert_column(
        self,
        table_uuid: uuid.UUID,
        column_name: str,
        data_type: str,
        description: Optional[str] = None,
        is_primary_key: bool = False,
        is_foreign_key: bool = False,
        references_table: Optional[str] = None,
        references_column: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        is_active: bool = True
    ) -> Tuple[KnowledgeColumnModel, bool]:
        cname_clean = column_name.lower().strip()
        existing = self.session.query(KnowledgeColumnModel).filter(
            KnowledgeColumnModel.table_id == table_uuid,
            KnowledgeColumnModel.column_name == cname_clean
        ).first()

        if existing:
            existing.data_type = data_type
            existing.description = description
            existing.is_primary_key = is_primary_key
            existing.is_foreign_key = is_foreign_key
            existing.references_table = references_table
            existing.references_column = references_column
            existing.keywords = keywords or []
            existing.is_active = is_active
            existing.updated_at = utc_now()
            return existing, False

        new_col = KnowledgeColumnModel(
            table_id=table_uuid,
            column_name=cname_clean,
            data_type=data_type,
            description=description,
            is_primary_key=is_primary_key,
            is_foreign_key=is_foreign_key,
            references_table=references_table,
            references_column=references_column,
            keywords=keywords or [],
            is_active=is_active
        )
        self.session.add(new_col)
        self.session.flush()
        return new_col, True

    # 4. Table Dependencies
    def add_table_dependency(
        self,
        source_table_id: uuid.UUID,
        referenced_table_id: uuid.UUID,
        relationship_type: str = "FOREIGN_KEY"
    ) -> Tuple[TableDependencyModel, bool]:
        existing = self.session.query(TableDependencyModel).filter(
            TableDependencyModel.source_table_id == source_table_id,
            TableDependencyModel.referenced_table_id == referenced_table_id,
            TableDependencyModel.relationship_type == relationship_type
        ).first()
        if existing:
            return existing, False

        dep = TableDependencyModel(
            source_table_id=source_table_id,
            referenced_table_id=referenced_table_id,
            relationship_type=relationship_type
        )
        self.session.add(dep)
        self.session.flush()
        return dep, True

    def list_dependencies_for_table(self, table_id: uuid.UUID) -> List[TableDependencyModel]:
        return self.session.query(TableDependencyModel).filter(
            TableDependencyModel.source_table_id == table_id
        ).all()

    # 5. Knowledge Segments (Hybrid Chunks)
    def get_segment_by_id(self, segment_uuid: uuid.UUID) -> Optional[KnowledgeSegmentModel]:
        return self.session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.id == segment_uuid).first()

    def list_active_segments(
        self,
        domain_id: Optional[uuid.UUID] = None,
        table_id: Optional[uuid.UUID] = None,
        segment_type: Optional[str] = None
    ) -> List[KnowledgeSegmentModel]:
        q = self.session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.is_active.is_(True))
        if domain_id:
            q = q.filter(KnowledgeSegmentModel.domain_id == domain_id)
        if table_id:
            q = q.filter(KnowledgeSegmentModel.table_id == table_id)
        if segment_type:
            q = q.filter(KnowledgeSegmentModel.segment_type == segment_type)
        return q.all()

    def upsert_segment(
        self,
        segment_uuid: uuid.UUID,
        domain_id: uuid.UUID,
        table_id: Optional[uuid.UUID],
        section: str,
        topic: str,
        segment_type: str,
        segment_name: str,
        content: str,
        content_hash: str,
        is_active: bool = True
    ) -> Tuple[KnowledgeSegmentModel, str]:
        existing = self.session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.id == segment_uuid).first()

        if existing:
            if existing.content_hash == content_hash and existing.is_active == is_active:
                return existing, "NONE"

            self.archive_segment_version(
                segment_id=existing.id,
                version=existing.version,
                content=existing.content,
                content_hash=existing.content_hash
            )

            existing.section = section
            existing.topic = topic
            existing.segment_type = segment_type
            existing.segment_name = segment_name
            existing.content = content
            existing.content_hash = content_hash
            existing.version += 1
            existing.is_active = is_active
            existing.updated_at = utc_now()
            return existing, "UPDATE"

        new_segment = KnowledgeSegmentModel(
            id=segment_uuid,
            domain_id=domain_id,
            table_id=table_id,
            section=section,
            topic=topic,
            segment_type=segment_type,
            segment_name=segment_name,
            content=content,
            content_hash=content_hash,
            version=1,
            is_active=is_active
        )
        self.session.add(new_segment)
        self.session.flush()
        return new_segment, "INSERT"

    def archive_segment_version(
        self,
        segment_id: uuid.UUID,
        version: int,
        content: str,
        content_hash: str
    ) -> SegmentVersionModel:
        existing = self.session.query(SegmentVersionModel).filter(
            SegmentVersionModel.segment_id == segment_id,
            SegmentVersionModel.version == version
        ).first()
        if existing:
            return existing

        version_record = SegmentVersionModel(
            segment_id=segment_id,
            version=version,
            content=content,
            content_hash=content_hash,
            changed_at=utc_now()
        )
        self.session.add(version_record)
        self.session.flush()
        return version_record

    def get_segment_history(self, segment_id: uuid.UUID) -> List[SegmentVersionModel]:
        return self.session.query(SegmentVersionModel).filter(
            SegmentVersionModel.segment_id == segment_id
        ).order_by(SegmentVersionModel.version.desc()).all()

    # 6. Embedding Jobs
    def create_embedding_job(
        self,
        segment_id: uuid.UUID,
        content_hash: str,
        operation: str
    ) -> EmbeddingJobModel:
        job = EmbeddingJobModel(
            segment_id=segment_id,
            content_hash=content_hash,
            operation=operation,
            status="PENDING",
            created_at=utc_now()
        )
        self.session.add(job)
        self.session.flush()
        return job

    def get_pending_embedding_jobs(self, limit: int = 100) -> List[EmbeddingJobModel]:
        return self.session.query(EmbeddingJobModel).filter(
            EmbeddingJobModel.status == "PENDING"
        ).order_by(EmbeddingJobModel.created_at.asc()).limit(limit).all()

    def update_job_status(
        self,
        job_id: uuid.UUID,
        status: str,
        error_message: Optional[str] = None
    ):
        job = self.session.query(EmbeddingJobModel).filter(EmbeddingJobModel.id == job_id).first()
        if job:
            job.status = status
            job.error_message = error_message
            if status in ("COMPLETED", "FAILED"):
                job.processed_at = utc_now()
            if status == "FAILED":
                job.retry_count += 1

    # 7. Segment Vectors
    def upsert_segment_vector(
        self,
        segment_id: uuid.UUID,
        embedding_model: str,
        faiss_vector_id: int,
        content_hash: str,
        model_version: str = "v1"
    ) -> SegmentVectorModel:
        existing = self.session.query(SegmentVectorModel).filter(
            SegmentVectorModel.embedding_model == embedding_model,
            SegmentVectorModel.segment_id == segment_id
        ).first()

        if existing:
            existing.faiss_vector_id = faiss_vector_id
            existing.content_hash = content_hash
            existing.model_version = model_version
            existing.indexed_at = utc_now()
            return existing

        vec = SegmentVectorModel(
            segment_id=segment_id,
            embedding_model=embedding_model,
            model_version=model_version,
            faiss_vector_id=faiss_vector_id,
            content_hash=content_hash,
            indexed_at=utc_now()
        )
        self.session.add(vec)
        self.session.flush()
        return vec

    def get_segment_vector_by_faiss_id(self, embedding_model: str, faiss_vector_id: int) -> Optional[SegmentVectorModel]:
        return self.session.query(SegmentVectorModel).filter(
            SegmentVectorModel.embedding_model == embedding_model,
            SegmentVectorModel.faiss_vector_id == faiss_vector_id
        ).first()

    def get_next_faiss_vector_id(self, embedding_model: str) -> int:
        max_id = self.session.query(func.max(SegmentVectorModel.faiss_vector_id)).filter(
            SegmentVectorModel.embedding_model == embedding_model
        ).scalar()
        return (max_id + 1) if max_id is not None else 0

    def delete_segment_vector(self, embedding_model: str, segment_id: uuid.UUID):
        self.session.query(SegmentVectorModel).filter(
            SegmentVectorModel.embedding_model == embedding_model,
            SegmentVectorModel.segment_id == segment_id
        ).delete()


# =============================================================================
# JSON Knowledge Base Migrator & Seeder
# =============================================================================

class JSONKnowledgeBaseMigrator:
    """
    Production-grade, idempotent migration engine from JSON knowledge-base
    files into PostgreSQL as the primary source of truth.
    """

    def __init__(self, data_dir: Optional[Path] = None, session: Optional[Session] = None):
        from src.rag.config import DOMAINS_DATA_DIR
        self.data_dir = data_dir or DOMAINS_DATA_DIR
        self._external_session = session

    def run_migration(self, force_rebuild_segments: bool = False) -> Dict[str, Any]:
        """Executes full transactional migration."""
        if self._external_session:
            return self._execute_migration(self._external_session, force_rebuild_segments)
        else:
            with get_db() as session:
                return self._execute_migration(session, force_rebuild_segments)

    def _execute_migration(self, session: Session, force_rebuild_segments: bool = False) -> Dict[str, Any]:
        repo = KnowledgeBaseRepository(session)
        logger.info(f"Starting Knowledge Base migration from: {self.data_dir}")

        stats = {
            "domains_created": 0,
            "domains_updated": 0,
            "tables_created": 0,
            "tables_updated": 0,
            "columns_created": 0,
            "columns_updated": 0,
            "dependencies_created": 0,
            "segments_created": 0,
            "segments_updated": 0,
            "segments_skipped": 0,
            "embedding_jobs_created": 0,
            "errors": []
        }

        if not self.data_dir.exists():
            err_msg = f"Data directory {self.data_dir} does not exist."
            logger.error(err_msg)
            stats["errors"].append(err_msg)
            return stats

        domain_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        logger.info(f"Discovered {len(domain_dirs)} domain directories.")

        table_obj_map: Dict[str, Any] = {}
        all_chunks: List[KnowledgeChunk] = []

        for dom_dir in sorted(domain_dirs):
            dom_id = dom_dir.name
            domain_file = dom_dir / "domain.json"
            tables_file = dom_dir / "tables.json"
            columns_file = dom_dir / "columns.json"

            domain_data: Dict[str, Any] = {}
            if domain_file.exists():
                try:
                    with open(domain_file, "r", encoding="utf-8") as df:
                        domain_data = json.load(df)
                except Exception as e:
                    stats["errors"].append(f"Failed to read {domain_file}: {e}")
                    continue
            else:
                domain_data = {
                    "domain_id": dom_id,
                    "domain_name": dom_id.replace("_", " ").title(),
                    "system": "Enterprise ERP"
                }

            dom_obj, created = repo.upsert_domain(
                domain_id=dom_id,
                domain_name=domain_data.get("domain_name", dom_id.title()),
                system_name=domain_data.get("system", "Enterprise ERP"),
                description=domain_data.get("description"),
                keywords=domain_data.get("keywords", [])
            )
            if created:
                stats["domains_created"] += 1
            else:
                stats["domains_updated"] += 1

            all_chunks.append(build_domain_chunk(domain_data))

            tables_data: List[Dict[str, Any]] = []
            if tables_file.exists():
                try:
                    with open(tables_file, "r", encoding="utf-8") as tf:
                        tables_data = json.load(tf)
                except Exception as e:
                    stats["errors"].append(f"Failed to read {tables_file}: {e}")

            for tdata in tables_data:
                tname = tdata.get("table_name", "").strip().lower()
                sql = tdata.get("sql", "").strip()
                thash = compute_sha256(sql)

                tbl_obj, t_created = repo.upsert_table(
                    domain_uuid=dom_obj.id,
                    table_name=tname,
                    sql_definition=sql,
                    content_hash=thash,
                    description=tdata.get("description"),
                    keywords=tdata.get("keywords", [])
                )
                if t_created:
                    stats["tables_created"] += 1
                else:
                    stats["tables_updated"] += 1

                table_obj_map[tname] = tbl_obj
                all_chunks.append(build_table_chunk(domain_data, tdata))

            columns_data: List[Dict[str, Any]] = []
            if columns_file.exists():
                try:
                    with open(columns_file, "r", encoding="utf-8") as cf:
                        columns_data = json.load(cf)
                except Exception as e:
                    stats["errors"].append(f"Failed to read {columns_file}: {e}")

            for cdata in columns_data:
                ctable = cdata.get("table_name", "").strip().lower()
                tbl_obj = table_obj_map.get(ctable)
                if not tbl_obj:
                    continue

                col_name = cdata.get("column_name", "").strip().lower()
                data_type = cdata.get("data_type", "VARCHAR(255)")
                ref_tbl = cdata.get("references_table")
                ref_col = cdata.get("references_column")

                col_obj, c_created = repo.upsert_column(
                    table_uuid=tbl_obj.id,
                    column_name=col_name,
                    data_type=data_type,
                    description=cdata.get("description"),
                    is_primary_key=cdata.get("is_primary_key", False),
                    is_foreign_key=cdata.get("is_foreign_key", False),
                    references_table=ref_tbl,
                    references_column=ref_col,
                    keywords=cdata.get("keywords", [])
                )
                if c_created:
                    stats["columns_created"] += 1
                else:
                    stats["columns_updated"] += 1

                all_chunks.append(build_column_chunk(domain_data, cdata))

                if ref_tbl and str(ref_tbl).strip().lower() in table_obj_map:
                    ref_tbl_clean = str(ref_tbl).strip().lower()
                    parent_tbl = table_obj_map[ref_tbl_clean]
                    _, dep_created = repo.add_table_dependency(
                        source_table_id=tbl_obj.id,
                        referenced_table_id=parent_tbl.id,
                        relationship_type="FOREIGN_KEY"
                    )
                    if dep_created:
                        stats["dependencies_created"] += 1

                    all_chunks.append(build_relationship_chunk(
                        domain_data=domain_data,
                        source_table=ctable,
                        target_table=ref_tbl_clean,
                        fk_col=col_name
                    ))

        # Stage 2: Ingest Hybrid Segments
        logger.info(f"Processing {len(all_chunks)} hybrid knowledge segments...")
        for chunk in all_chunks:
            dom = repo.get_domain_by_business_id(chunk.domain_id, active_only=False)
            if not dom:
                continue

            tbl_id = None
            if chunk.table_name and chunk.table_name in table_obj_map:
                tbl_id = table_obj_map[chunk.table_name].id

            seg_obj, op = repo.upsert_segment(
                segment_uuid=chunk.segment_uuid,
                domain_id=dom.id,
                table_id=tbl_id,
                section=chunk.section,
                topic=chunk.topic,
                segment_type=chunk.segment_type,
                segment_name=chunk.segment_name,
                content=chunk.content,
                content_hash=chunk.content_hash,
                is_active=True
            )

            if op == "INSERT":
                stats["segments_created"] += 1
                repo.create_embedding_job(seg_obj.id, chunk.content_hash, "INSERT")
                stats["embedding_jobs_created"] += 1
            elif op == "UPDATE":
                stats["segments_updated"] += 1
                repo.create_embedding_job(seg_obj.id, chunk.content_hash, "UPDATE")
                stats["embedding_jobs_created"] += 1
            else:
                stats["segments_skipped"] += 1

        logger.info(
            f"Migration completed successfully! "
            f"Domains: {stats['domains_created']} created, {stats['domains_updated']} updated | "
            f"Tables: {stats['tables_created']} created, {stats['tables_updated']} updated | "
            f"Columns: {stats['columns_created']} created, {stats['columns_updated']} updated | "
            f"Segments: {stats['segments_created']} created, {stats['segments_updated']} updated, {stats['segments_skipped']} skipped | "
            f"Jobs Queued: {stats['embedding_jobs_created']}"
        )
        return stats
