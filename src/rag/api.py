import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from src.config import get_logger
from src.rag.db.database import get_db
from src.rag.db.repository import KnowledgeBaseRepository
from src.rag.db.chunking import compute_sha256, generate_stable_segment_uuid
from src.rag.db.migration_service import run_full_migration, JSONKnowledgeBaseMigrator
from src.rag.db.embedding_sync import get_faiss_synchronizer

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/kb", tags=["Knowledge Base Management"])


# =============================================================================
# Request Schemas
# =============================================================================

class DomainCreateRequest(BaseModel):
    domain_id: str
    domain_name: str
    system_name: str = "Enterprise ERP"
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    is_active: bool = True


class TableCreateRequest(BaseModel):
    domain_id: str
    table_name: str
    sql_definition: str
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    is_active: bool = True


class ColumnCreateRequest(BaseModel):
    column_name: str
    data_type: str
    description: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class TableDependencyRequest(BaseModel):
    source_table_name: str
    referenced_table_name: str
    relationship_type: str = "FOREIGN_KEY"


class SegmentUpdateRequest(BaseModel):
    content: str
    is_active: Optional[bool] = None


# =============================================================================
# Endpoints
# =============================================================================

# 1. Domains
@router.get("/domains")
def list_domains(active_only: bool = True):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        domains = repo.list_domains(active_only=active_only)
        return {"total": len(domains), "domains": [d.to_dict() for d in domains]}


@router.get("/domains/{domain_id}")
def get_domain(domain_id: str):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        dom = repo.get_domain_by_business_id(domain_id, active_only=False)
        if not dom:
            raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found.")
        return dom.to_dict()


@router.post("/domains")
def create_or_update_domain(payload: DomainCreateRequest):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        dom, created = repo.upsert_domain(
            domain_id=payload.domain_id,
            domain_name=payload.domain_name,
            system_name=payload.system_name,
            description=payload.description,
            keywords=payload.keywords,
            is_active=payload.is_active
        )
        return {"created": created, "domain": dom.to_dict()}


# 2. Knowledge Tables
@router.get("/tables")
def list_tables(domain_id: Optional[str] = None, active_only: bool = True):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        tables = repo.list_tables(domain_id=domain_id, active_only=active_only)
        return {"total": len(tables), "tables": [t.to_dict() for t in tables]}


@router.get("/tables/{table_name}")
def get_table(table_name: str, domain_id: Optional[str] = None):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        tbl = repo.get_table_by_name(table_name, domain_id=domain_id)
        if not tbl:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")
        cols = repo.list_columns_by_table(tbl.id)
        deps = repo.list_dependencies_for_table(tbl.id)
        data = tbl.to_dict()
        data["columns"] = [c.to_dict() for c in cols]
        data["dependencies"] = [d.to_dict() for d in deps]
        return data


@router.post("/tables")
def create_or_update_table(payload: TableCreateRequest):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        dom = repo.get_domain_by_business_id(payload.domain_id)
        if not dom:
            raise HTTPException(status_code=400, detail=f"Domain '{payload.domain_id}' does not exist.")
        thash = compute_sha256(f"{payload.sql_definition}:{payload.description or ''}")
        tbl, created = repo.upsert_table(
            domain_uuid=dom.id,
            table_name=payload.table_name,
            sql_definition=payload.sql_definition,
            content_hash=thash,
            description=payload.description,
            keywords=payload.keywords,
            is_active=payload.is_active
        )
        return {"created": created, "table": tbl.to_dict()}


# 3. Columns
@router.post("/tables/{table_name}/columns")
def create_or_update_column(table_name: str, payload: ColumnCreateRequest):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        tbl = repo.get_table_by_name(table_name)
        if not tbl:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")
        col, created = repo.upsert_column(
            table_uuid=tbl.id,
            column_name=payload.column_name,
            data_type=payload.data_type,
            description=payload.description,
            is_primary_key=payload.is_primary_key,
            is_foreign_key=payload.is_foreign_key,
            references_table=payload.references_table,
            references_column=payload.references_column,
            keywords=payload.keywords
        )
        return {"created": created, "column": col.to_dict()}


# 4. Dependencies
@router.post("/dependencies")
def create_dependency(payload: TableDependencyRequest):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        src = repo.get_table_by_name(payload.source_table_name)
        ref = repo.get_table_by_name(payload.referenced_table_name)
        if not src or not ref:
            raise HTTPException(status_code=400, detail="Source or referenced table not found.")
        dep, created = repo.add_table_dependency(
            source_table_id=src.id,
            referenced_table_id=ref.id,
            relationship_type=payload.relationship_type
        )
        return {"created": created, "dependency": dep.to_dict()}


# 5. Segments & Versions
@router.get("/segments")
def list_segments(domain_id: Optional[str] = None, segment_type: Optional[str] = None):
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        dom_uuid = None
        if domain_id:
            dom = repo.get_domain_by_business_id(domain_id)
            if dom:
                dom_uuid = dom.id
        segs = repo.list_active_segments(domain_id=dom_uuid, segment_type=segment_type)
        return {"total": len(segs), "segments": [s.to_dict() for s in segs]}


@router.put("/segments/{segment_id}")
def update_segment_content(segment_id: str, payload: SegmentUpdateRequest):
    try:
        seg_uuid = uuid.UUID(segment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid segment UUID format.")

    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        existing = repo.get_segment_by_id(seg_uuid)
        if not existing:
            raise HTTPException(status_code=404, detail="Segment not found.")

        new_hash = compute_sha256(payload.content)
        new_active = payload.is_active if payload.is_active is not None else existing.is_active

        seg, op = repo.upsert_segment(
            segment_uuid=seg_uuid,
            domain_id=existing.domain_id,
            table_id=existing.table_id,
            section=existing.section,
            topic=existing.topic,
            segment_type=existing.segment_type,
            segment_name=existing.segment_name,
            content=payload.content,
            content_hash=new_hash,
            is_active=new_active
        )
        if op in ("INSERT", "UPDATE"):
            repo.create_embedding_job(seg.id, new_hash, op)

        return {"operation": op, "segment": seg.to_dict()}


@router.get("/segments/{segment_id}/history")
def get_segment_history(segment_id: str):
    try:
        seg_uuid = uuid.UUID(segment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid segment UUID format.")

    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        history = repo.get_segment_history(seg_uuid)
        return {"segment_id": segment_id, "versions": [v.to_dict() for v in history]}


# 6. Synchronization & Embedding Jobs
@router.post("/sync/embeddings")
def trigger_embedding_sync(background_tasks: BackgroundTasks):
    syncer = get_faiss_synchronizer()
    background_tasks.add_task(syncer.sync_all)
    return {"message": "Embedding synchronization triggered in background."}


@router.get("/sync/status")
def check_sync_status():
    with get_db() as session:
        repo = KnowledgeBaseRepository(session)
        pending = repo.get_pending_embedding_jobs(limit=10)
        from src.rag.db.models import EmbeddingJobModel
        total_pending = session.query(EmbeddingJobModel).filter(EmbeddingJobModel.status == "PENDING").count()
        total_failed = session.query(EmbeddingJobModel).filter(EmbeddingJobModel.status == "FAILED").count()
        syncer = get_faiss_synchronizer()
        return {
            "faiss_total_vectors": syncer.index.ntotal if syncer.index else 0,
            "total_pending_jobs": total_pending,
            "total_failed_jobs": total_failed,
            "next_pending": [p.to_dict() for p in pending]
        }


@router.post("/sync/rebuild")
def rebuild_faiss_index(background_tasks: BackgroundTasks):
    def _rebuild():
        syncer = get_faiss_synchronizer()
        syncer._init_index()
        syncer.sync_all()
    background_tasks.add_task(_rebuild)
    return {"message": "FAISS index rebuild triggered in background."}


@router.post("/ingest")
def trigger_json_ingest(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_full_migration)
    return {"message": "Full JSON ingestion into PostgreSQL triggered in background."}
