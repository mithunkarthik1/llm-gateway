"""
Production Database Engine, Session Management, Repository, and Migration Loader
for PostgreSQL Knowledge Base (Primary Source of Truth).
"""
import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    DOMAINS_DATA_DIR,
    EMBEDDING_MODEL_NAME,
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
    """Returns singleton SQLAlchemy Engine with robust connection pooling."""
    global _ENGINE, _SESSION_FACTORY
    target_url = db_url or DATABASE_URL

    if _ENGINE is None:
        if not target_url:
            logger.error("DATABASE_URL is not configured in environment or .env file.")
            raise ValueError("DATABASE_URL is not configured in environment or .env file.")

        try:
            masked_url = target_url
            if "@" in target_url and ":" in target_url.split("@")[0]:
                prefix = target_url.split("@")[0]
                masked_url = prefix.split(":")[0] + ":***@" + target_url.split("@")[1]

            logger.info(f"Initialized Database engine for: {masked_url}")

            if target_url.startswith("sqlite"):
                _ENGINE = create_engine(
                    target_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                )
            else:
                _ENGINE = create_engine(
                    target_url,
                    poolclass=QueuePool,
                    pool_size=DB_POOL_SIZE,
                    max_overflow=DB_MAX_OVERFLOW,
                    pool_timeout=DB_POOL_TIMEOUT,
                    pool_recycle=DB_POOL_RECYCLE,
                    pool_pre_ping=True,
                )

            _SESSION_FACTORY = scoped_session(
                sessionmaker(autocommit=False, autoflush=False, bind=_ENGINE)
            )
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise

    return _ENGINE


def get_session(db_url: Optional[str] = None) -> Session:
    """Creates a new SQLAlchemy session."""
    global _SESSION_FACTORY
    try:
        if _SESSION_FACTORY is None:
            get_engine(db_url)
        return _SESSION_FACTORY()
    except Exception as e:
        logger.error(f"Failed to create database session: {e}")
        raise


@contextmanager
def get_db(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """Context manager for transactional database sessions with automatic rollback."""
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
    """Verifies live PostgreSQL database connectivity."""
    try:
        engine = get_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"Database connectivity check failed: {e}")
        return False


def init_db(db_url: Optional[str] = None, drop_all: bool = False):
    """Initializes all database tables from models metadata."""
    try:
        engine = get_engine(db_url)
        if drop_all:
            logger.warning("Dropping all existing knowledge base tables...")
            Base.metadata.drop_all(bind=engine)
        logger.info("Creating knowledge base tables if not present...")
        Base.metadata.create_all(bind=engine)

        # Auto-migrate segment_key column and drop legacy content_hash columns
        try:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE knowledge_segments ADD COLUMN IF NOT EXISTS segment_key VARCHAR(256);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_knowledge_segments_segment_key ON knowledge_segments(segment_key);"))
                conn.execute(text("ALTER TABLE knowledge_tables DROP COLUMN IF EXISTS content_hash;"))
                conn.execute(text("ALTER TABLE knowledge_segments DROP COLUMN IF EXISTS content_hash;"))
                conn.execute(text("ALTER TABLE segment_versions DROP COLUMN IF EXISTS content_hash;"))
                conn.execute(text("ALTER TABLE embedding_jobs DROP COLUMN IF EXISTS content_hash;"))
                conn.execute(text("ALTER TABLE segment_vectors DROP COLUMN IF EXISTS content_hash;"))
        except Exception as e:
            logger.debug(f"Column migration check note: {e}")

        logger.info("Knowledge base tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise


# =============================================================================
# Chunking Data Structure & Semantic Builders (Key-Driven)
# =============================================================================

SEGMENT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def generate_stable_segment_uuid(domain_id: str, topic: str, segment_name: str) -> uuid.UUID:
    """Generates a deterministic, collision-free UUIDv5 for a logical segment identity."""
    identity_key = f"{domain_id.lower().strip()}:{topic.lower().strip()}:{segment_name.lower().strip()}"
    return uuid.uuid5(SEGMENT_NAMESPACE, identity_key)


@dataclass
class KnowledgeChunk:
    """Represents a structured hybrid chunk ready for DB persistence & vectorization."""
    segment_uuid: uuid.UUID
    domain_id: str
    section: str
    topic: str
    segment_type: str
    segment_name: str
    content: str
    table_name: Optional[str] = None
    segment_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.content = self.content.strip()
        if not self.segment_key:
            self.segment_key = (
                self.metadata.get("identity_key")
                or f"{self.domain_id}:{self.table_name or 'none'}:{self.segment_name}"
            )
        self.metadata["segment_key"] = self.segment_key
        self.metadata["identity_key"] = self.segment_key


def build_domain_chunk(domain_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a domain-level hybrid segment."""
    try:
        dom_id = domain_data.get("domain_id", "").strip()
        dom_name = domain_data.get("domain_name", dom_id).strip()
        system = domain_data.get("system", "Enterprise ERP").strip()
        desc = domain_data.get("description", "").strip()
        keywords = domain_data.get("keywords", [])
        tbls = domain_data.get("table_names", [])

        section = dom_name
        topic = f"{dom_name} Overview"
        seg_name = f"{dom_id}.overview"
        seg_key = f"{dom_id}:domain:overview"
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
            segment_key=seg_key,
            content=content,
            metadata={
                "domain_id": dom_id,
                "domain_name": dom_name,
                "segment_type": "domain",
                "segment_key": seg_key,
                "identity_key": seg_key,
                "table_count": len(tbls)
            }
        )
    except Exception as e:
        logger.error(f"Error building domain chunk: {e}")
        raise


def build_table_chunk(domain_data: Dict[str, Any], table_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a table-level hybrid segment."""
    try:
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
        seg_key = f"{dom_id}:{tname}:definition"
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
            segment_key=seg_key,
            content=content,
            table_name=tname,
            metadata={
                "domain_id": dom_id,
                "domain_name": dom_name,
                "table_name": tname,
                "segment_type": "table",
                "segment_key": seg_key,
                "identity_key": seg_key,
                "dependencies": deps
            }
        )
    except Exception as e:
        logger.error(f"Error building table chunk: {e}")
        raise


def build_column_chunk(domain_data: Dict[str, Any], column_data: Dict[str, Any]) -> KnowledgeChunk:
    """Builds a granular column-level hybrid segment."""
    try:
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
        seg_key = f"{dom_id}:{tname}:{cname}"
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
            segment_key=seg_key,
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
                "segment_key": seg_key,
                "identity_key": seg_key
            }
        )
    except Exception as e:
        logger.error(f"Error building column chunk: {e}")
        raise


def build_relationship_chunk(
    domain_data: Dict[str, Any],
    source_table: str,
    target_table: str,
    fk_col: str,
    rel_type: str = "FOREIGN_KEY"
) -> KnowledgeChunk:
    """Builds a relational foreign-key dependency hybrid segment."""
    try:
        dom_id = domain_data.get("domain_id", "").strip()
        dom_name = domain_data.get("domain_name", dom_id).strip()

        section = dom_name
        topic = source_table.replace("_", " ").title()
        seg_name = f"{source_table}.ref_{target_table}"
        seg_key = f"{dom_id}:{source_table}:ref_{target_table}"
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
            segment_key=seg_key,
            content=content,
            table_name=source_table,
            metadata={
                "domain_id": dom_id,
                "domain_name": dom_name,
                "source_table": source_table,
                "referenced_table": target_table,
                "fk_column": fk_col,
                "segment_type": "relationship",
                "segment_key": seg_key,
                "identity_key": seg_key
            }
        )
    except Exception as e:
        logger.error(f"Error building relationship chunk: {e}")
        raise


# =============================================================================
# Knowledge Base Repository (Clean 2-Tier CRUD without Hashing)
# =============================================================================

class KnowledgeBaseRepository:
    """
    Data Access Layer providing CRUD and 2-tier search/alteration operations.
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
        return q.order_by(DomainModel.domain_name).all()

    def upsert_domain(
        self,
        domain_id: str,
        domain_name: str,
        system_name: str = "Enterprise ERP",
        description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        is_active: bool = True
    ) -> Tuple[DomainModel, bool]:
        dom_clean = domain_id.lower().strip()
        existing = self.session.query(DomainModel).filter(DomainModel.domain_id == dom_clean).first()
        if existing:
            existing.domain_name = domain_name
            existing.system_name = system_name
            existing.description = description
            existing.keywords = keywords or []
            existing.is_active = is_active
            existing.updated_at = utc_now()
            return existing, False

        new_domain = DomainModel(
            domain_id=dom_clean,
            domain_name=domain_name,
            system_name=system_name,
            description=description,
            keywords=keywords or [],
            is_active=is_active
        )
        self.session.add(new_domain)
        self.session.flush()
        return new_domain, True

    # 2. Knowledge Tables
    def get_table_by_id(self, table_uuid: uuid.UUID) -> Optional[KnowledgeTableModel]:
        return self.session.query(KnowledgeTableModel).filter(KnowledgeTableModel.id == table_uuid).first()

    def get_table_by_name(self, table_name: str, domain_id: Optional[str] = None) -> Optional[KnowledgeTableModel]:
        q = self.session.query(KnowledgeTableModel).filter(
            KnowledgeTableModel.table_name == table_name.lower().strip(),
            KnowledgeTableModel.is_active.is_(True)
        )
        if domain_id:
            dom = self.get_domain_by_business_id(domain_id, active_only=False)
            if dom:
                q = q.filter(KnowledgeTableModel.domain_id == dom.id)
            else:
                return None
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
            existing.description = description
            existing.keywords = keywords or []
            existing.is_active = is_active
            existing.updated_at = utc_now()
            return existing, False

        new_tbl = KnowledgeTableModel(
            domain_id=domain_uuid,
            table_name=tname_clean,
            sql_definition=sql_definition,
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
        return q.order_by(KnowledgeColumnModel.column_name).all()

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
            existing.references_table = references_table.lower().strip() if references_table else None
            existing.references_column = references_column.lower().strip() if references_column else None
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
            references_table=references_table.lower().strip() if references_table else None,
            references_column=references_column.lower().strip() if references_column else None,
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
            TableDependencyModel.referenced_table_id == referenced_table_id
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

    # 5. Knowledge Segments (2-Tier: segment_id + segment_key)
    def get_segment_by_id(self, segment_uuid: uuid.UUID) -> Optional[KnowledgeSegmentModel]:
        """Tier 1: Direct point lookup by permanent UUID."""
        return self.session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.id == segment_uuid).first()

    def get_segment_by_key(self, segment_key: str, active_only: bool = True) -> Optional[KnowledgeSegmentModel]:
        """Tier 2: Business lookup by structured segment_key (domain:table:column)."""
        q = self.session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.segment_key == segment_key)
        if active_only:
            q = q.filter(KnowledgeSegmentModel.is_active.is_(True))
        return q.first()

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
        segment_key: Optional[str] = None,
        is_active: bool = True
    ) -> Tuple[KnowledgeSegmentModel, str]:
        """2-Tier Resolution: Search by segment_key first if provided, else by permanent UUID."""
        existing = None
        if segment_key:
            existing = self.session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.segment_key == segment_key).first()
        if not existing and segment_uuid:
            existing = self.session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.id == segment_uuid).first()

        if existing:
            # Change detection directly on content without hashing
            content_changed = (existing.content.strip() != content.strip())
            key_changed = (segment_key is not None and existing.segment_key != segment_key)
            status_changed = (existing.is_active != is_active)

            if not content_changed and not key_changed and not status_changed:
                return existing, "NONE"

            self.archive_segment_version(
                segment_id=existing.id,
                version=existing.version,
                content=existing.content
            )

            existing.section = section
            existing.topic = topic
            existing.segment_type = segment_type
            existing.segment_name = segment_name
            if segment_key:
                existing.segment_key = segment_key
            existing.content = content
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
            segment_key=segment_key,
            content=content,
            version=1,
            is_active=is_active
        )
        self.session.add(new_segment)
        self.session.flush()
        return new_segment, "INSERT"

    def alter_segment(
        self,
        segment_id: uuid.UUID,
        new_content: str,
        new_segment_key: Optional[str] = None
    ) -> Tuple[KnowledgeSegmentModel, str]:
        """
        Tier 1: Direct in-place mutation by permanent segment_id.
        Removes reliance on content hash for finding or updating segments.
        """
        seg = self.get_segment_by_id(segment_id)
        if not seg:
            raise ValueError(f"Knowledge segment with ID '{segment_id}' not found.")

        content_changed = (seg.content.strip() != new_content.strip())
        key_changed = (new_segment_key is not None and seg.segment_key != new_segment_key)

        if not content_changed and not key_changed:
            return seg, "NONE"

        self.archive_segment_version(
            segment_id=seg.id,
            version=seg.version,
            content=seg.content
        )

        seg.content = new_content.strip()
        seg.version += 1
        if new_segment_key:
            seg.segment_key = new_segment_key
        seg.updated_at = utc_now()

        # Enqueue ChromaDB synchronization job
        self.create_embedding_job(
            segment_id=seg.id,
            operation="UPDATE"
        )
        self.session.flush()
        return seg, "UPDATE"

    def archive_segment_version(
        self,
        segment_id: uuid.UUID,
        version: int,
        content: str
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
            changed_at=utc_now()
        )
        self.session.add(version_record)
        self.session.flush()
        return version_record

    def get_segment_history(self, segment_id: uuid.UUID) -> List[SegmentVersionModel]:
        return self.session.query(SegmentVersionModel).filter(
            SegmentVersionModel.segment_id == segment_id
        ).order_by(SegmentVersionModel.version.desc()).all()

    # 6. Embedding Jobs (Asynchronous Synchronization Queue)
    def create_embedding_job(
        self,
        segment_id: uuid.UUID,
        operation: str = "INSERT"
    ) -> EmbeddingJobModel:
        job = EmbeddingJobModel(
            segment_id=segment_id,
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


# =============================================================================
# JSON Knowledge Base Migrator
# =============================================================================

class JSONKnowledgeBaseMigrator:
    """
    Production-grade, idempotent migration engine from JSON knowledge-base
    files into PostgreSQL as the primary source of truth.
    """

    def __init__(self, data_dir: Optional[Path] = None, session: Optional[Session] = None):
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

            # Ingest Domain Chunk
            all_chunks.append(build_domain_chunk(domain_data))

            # Ingest Tables
            tables_data = []
            if tables_file.exists():
                try:
                    with open(tables_file, "r", encoding="utf-8") as tf:
                        tables_data = json.load(tf)
                except Exception as e:
                    stats["errors"].append(f"Failed to read {tables_file}: {e}")

            for tdata in tables_data:
                tname = tdata.get("table_name", "").strip().lower()
                sql = tdata.get("sql", "").strip()

                tbl_obj, t_created = repo.upsert_table(
                    domain_uuid=dom_obj.id,
                    table_name=tname,
                    sql_definition=sql,
                    description=tdata.get("description"),
                    keywords=tdata.get("keywords", [])
                )
                if t_created:
                    stats["tables_created"] += 1
                else:
                    stats["tables_updated"] += 1

                table_obj_map[tname] = tbl_obj
                all_chunks.append(build_table_chunk(domain_data, tdata))

            # Ingest Columns & Foreign Key Dependencies
            columns_data = []
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

        # Stage 2: Ingest Hybrid Segments (Key-Driven)
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
                segment_key=chunk.segment_key,
                is_active=True
            )

            if op == "INSERT":
                stats["segments_created"] += 1
                repo.create_embedding_job(seg_obj.id, operation="INSERT")
                stats["embedding_jobs_created"] += 1
            elif op == "UPDATE":
                stats["segments_updated"] += 1
                repo.create_embedding_job(seg_obj.id, operation="UPDATE")
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
