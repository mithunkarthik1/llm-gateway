import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete, and_, or_, func

from src.config import get_logger
from src.rag.db.models import (
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


class KnowledgeBaseRepository:
    """
    Unified Data Access Layer for PostgreSQL ERP Knowledge Base.
    Provides fast, transactional, zero-N+1 query operations.
    """

    def __init__(self, session: Session):
        self.session = session

    # =========================================================================
    # 1. Domains
    # =========================================================================

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
        """
        Idempotent domain upsert.
        Returns (DomainModel, created_flag).
        """
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

    # =========================================================================
    # 2. Knowledge Tables
    # =========================================================================

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
        """
        Idempotent table upsert based on domain_id + table_name.
        Returns (KnowledgeTableModel, created_flag).
        """
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

    # =========================================================================
    # 3. Knowledge Columns
    # =========================================================================

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
        """
        Idempotent column upsert based on table_id + column_name.
        Returns (KnowledgeColumnModel, created_flag).
        """
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

    # =========================================================================
    # 4. Table Dependencies
    # =========================================================================

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

    # =========================================================================
    # 5. Knowledge Segments (Hybrid Chunks)
    # =========================================================================

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
        """
        Idempotent segment upsert with version tracking:
        - If new: creates segment (version 1) and returns (segment, 'INSERT')
        - If modified (hash mismatch): saves previous version to segment_versions, increments version, returns (segment, 'UPDATE')
        - If unchanged (hash match): skips version increment, returns (segment, 'NONE')
        """
        existing = self.session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.id == segment_uuid).first()

        if existing:
            if existing.content_hash == content_hash and existing.is_active == is_active:
                # Content identical -> No changes needed
                return existing, "NONE"

            # Content modified -> Archive current version to segment_versions
            self.archive_segment_version(
                segment_id=existing.id,
                version=existing.version,
                content=existing.content,
                content_hash=existing.content_hash
            )

            # Update segment with new content and increment version
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

        # Brand new segment
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
        """Saves a historical snapshot before segment modification."""
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

    # =========================================================================
    # 6. Embedding Jobs Queue
    # =========================================================================

    def create_embedding_job(
        self,
        segment_id: uuid.UUID,
        content_hash: str,
        operation: str
    ) -> EmbeddingJobModel:
        """Creates an embedding synchronization job."""
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

    # =========================================================================
    # 7. Segment Vectors (UUID <-> FAISS ID Mapping)
    # =========================================================================

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
