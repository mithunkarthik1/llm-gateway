import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import faiss
from sqlalchemy.orm import Session

from src.config import get_logger
from src.rag.config import (
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    VECTOR_DB_BACKEND,
)
from src.rag.faiss_service import get_embedding_model
from src.rag.chroma_service import get_chroma_engine
from src.rag.db.database import get_db
from src.rag.db.repository import KnowledgeBaseRepository
from src.rag.db.models import KnowledgeSegmentModel, EmbeddingJobModel, SegmentVectorModel

logger = get_logger(__name__)


class FAISSPostgresSynchronizer:
    """
    Synchronizes PostgreSQL knowledge segments with FAISS vector index.
    Features:
    - ID-mapped vector indexing (IndexIDMap2 wrapping IndexFlatIP).
    - Incremental batch processing of embedding_jobs.
    - Safe removal of stale vectors upon segment updates or deletions.
    - Atomic index file persistence.
    - Disaster recovery / reconciliation from PostgreSQL source of truth.
    """

    def __init__(self, index_dir: Path = FAISS_INDEX_DIR, model_name: str = EMBEDDING_MODEL_NAME):
        self.index_dir = index_dir
        self.model_name = model_name
        self.index_path = self.index_dir / "index.faiss"
        self.index: Optional[faiss.IndexIDMap2] = None
        self._init_index()

    def _init_index(self):
        """Loads existing IndexIDMap2 or initializes a new flat inner product index."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            try:
                raw_idx = faiss.read_index(str(self.index_path))
                if isinstance(raw_idx, faiss.IndexIDMap2):
                    self.index = raw_idx
                elif isinstance(raw_idx, faiss.IndexIDMap):
                    self.index = faiss.IndexIDMap2(raw_idx.index)
                else:
                    self.index = faiss.IndexIDMap2(raw_idx)
                logger.info(f"Loaded existing FAISS IndexIDMap2 ({self.index.ntotal} vectors) from {self.index_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index from {self.index_path}: {e}. Creating new index.")

        base_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.index = faiss.IndexIDMap2(base_index)
        logger.info(f"Initialized new FAISS IndexIDMap2 with dimension {EMBEDDING_DIMENSION}.")

    def save_index(self):
        """Atomically saves the FAISS index to disk."""
        if self.index is None:
            return
        temp_path = self.index_dir / "index.faiss.tmp"
        try:
            faiss.write_index(self.index, str(temp_path))
            if temp_path.exists():
                shutil.move(str(temp_path), str(self.index_path))
                logger.info(f"Atomically saved FAISS index to {self.index_path} ({self.index.ntotal} vectors).")
        except Exception as e:
            logger.error(f"Error saving FAISS index to {self.index_path}: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    def process_pending_jobs(self, batch_size: int = 128, session: Optional[Session] = None) -> Dict[str, int]:
        """
        Pulls PENDING embedding_jobs from PostgreSQL, encodes segment text,
        updates FAISS index and segment_vectors mapping, and marks jobs COMPLETED.
        """
        stats = {"processed": 0, "inserted": 0, "updated": 0, "deleted": 0, "failed": 0}

        def _run(sess: Session):
            repo = KnowledgeBaseRepository(sess)
            jobs = repo.get_pending_embedding_jobs(limit=batch_size)
            if not jobs:
                return stats

            logger.info(f"Processing batch of {len(jobs)} pending embedding jobs...")
            model = get_embedding_model()

            to_encode_jobs: List[Tuple[EmbeddingJobModel, KnowledgeSegmentModel]] = []
            delete_jobs: List[EmbeddingJobModel] = []

            for job in jobs:
                seg = repo.get_segment_by_id(job.segment_id)
                if job.operation == "DELETE" or (seg and not seg.is_active):
                    delete_jobs.append(job)
                elif seg and seg.is_active:
                    to_encode_jobs.append((job, seg))
                else:
                    repo.update_job_status(job.id, "COMPLETED")
                    stats["processed"] += 1

            # Handle Deletions
            chroma_engine = get_chroma_engine()
            for job in delete_jobs:
                try:
                    existing_vec = repo.session.query(SegmentVectorModel).filter(
                        SegmentVectorModel.embedding_model == self.model_name,
                        SegmentVectorModel.segment_id == job.segment_id
                    ).first()
                    if existing_vec:
                        faiss_id = existing_vec.faiss_vector_id
                        try:
                            self.index.remove_ids(np.array([faiss_id], dtype=np.int64))
                        except Exception as fe:
                            logger.warning(f"Error removing FAISS vector ID {faiss_id}: {fe}")
                        repo.delete_segment_vector(self.model_name, job.segment_id)
                    # Delete from ChromaDB
                    try:
                        chroma_engine.delete_segment(str(job.segment_id))
                    except Exception as ce:
                        logger.warning(f"Error removing ChromaDB vector for segment {job.segment_id}: {ce}")

                    repo.update_job_status(job.id, "COMPLETED")
                    stats["deleted"] += 1
                    stats["processed"] += 1
                except Exception as e:
                    repo.update_job_status(job.id, "FAILED", error_message=str(e))
                    stats["failed"] += 1

            # Handle Insertions & Updates
            if to_encode_jobs:
                texts = [seg.content for _, seg in to_encode_jobs]
                try:
                    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                except Exception as enc_err:
                    logger.error(f"Failed to generate embeddings: {enc_err}")
                    for job, _ in to_encode_jobs:
                        repo.update_job_status(job.id, "FAILED", error_message=str(enc_err))
                        stats["failed"] += 1
                    return stats

                next_id = repo.get_next_faiss_vector_id(self.model_name)

                for i, (job, seg) in enumerate(to_encode_jobs):
                    try:
                        emb = embeddings[i : i + 1].astype(np.float32)

                        # Check if vector mapping already exists
                        existing_vec = repo.session.query(SegmentVectorModel).filter(
                            SegmentVectorModel.embedding_model == self.model_name,
                            SegmentVectorModel.segment_id == seg.id
                        ).first()

                        if existing_vec:
                            faiss_id = existing_vec.faiss_vector_id
                            # Remove old vector from FAISS
                            try:
                                self.index.remove_ids(np.array([faiss_id], dtype=np.int64))
                            except Exception:
                                pass
                            stats["updated"] += 1
                        else:
                            faiss_id = next_id
                            next_id += 1
                            stats["inserted"] += 1

                        # Add new vector with exact ID in FAISS
                        self.index.add_with_ids(emb, np.array([faiss_id], dtype=np.int64))

                        # Upsert in ChromaDB (direct in-place overwrite by UUID)
                        try:
                            meta = {
                                "domain_id": str(seg.domain_id),
                                "domain_name": seg.section,
                                "table_id": str(seg.table_id) if seg.table_id else "",
                                "topic": seg.topic,
                                "segment_type": seg.segment_type,
                                "segment_name": seg.segment_name,
                                "version": seg.version,
                                "identity_key": f"{seg.domain_id}:{seg.table_id or 'none'}:{seg.segment_name}"
                            }
                            chroma_engine.upsert_segment(
                                segment_id=str(seg.id),
                                document=seg.content,
                                metadata=meta,
                                embedding=emb[0]
                            )
                        except Exception as ce:
                            logger.warning(f"Error syncing segment {seg.id} to ChromaDB: {ce}")

                        # Upsert mapping record in PostgreSQL
                        repo.upsert_segment_vector(
                            segment_id=seg.id,
                            embedding_model=self.model_name,
                            faiss_vector_id=faiss_id,
                            content_hash=seg.content_hash
                        )

                        repo.update_job_status(job.id, "COMPLETED")
                        stats["processed"] += 1
                    except Exception as job_err:
                        logger.error(f"Failed processing job {job.id} for segment {seg.id}: {job_err}")
                        repo.update_job_status(job.id, "FAILED", error_message=str(job_err))
                        stats["failed"] += 1

            self.save_index()
            return stats

        if session:
            return _run(session)
        else:
            with get_db() as sess:
                return _run(sess)

    def sync_all(self, batch_size: int = 128) -> Dict[str, int]:
        """Runs batch processing until no PENDING jobs remain."""
        total_stats = {"processed": 0, "inserted": 0, "updated": 0, "deleted": 0, "failed": 0}
        while True:
            cur = self.process_pending_jobs(batch_size=batch_size)
            if cur["processed"] == 0 and cur["failed"] == 0:
                break
            for k in total_stats:
                total_stats[k] += cur.get(k, 0)
        logger.info(f"All pending jobs synchronized: {total_stats}")
        return total_stats

    def search_active_segments(
        self,
        query: str,
        top_k: int = 5,
        domain_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search against ChromaDB (with native metadata filtering)
        or FAISS, resolves returned vectors to PostgreSQL segment UUIDs,
        and fetches current active segments.
        """
        # 1. Primary Path: ChromaDB if enabled and populated
        if VECTOR_DB_BACKEND in ("chroma", "both"):
            try:
                chroma_engine = get_chroma_engine()
                if chroma_engine.count() > 0:
                    where_filter = None
                    if domain_id:
                        # Match by domain_id or domain_name
                        where_filter = {"domain_id": domain_id}

                    chroma_res = chroma_engine.search(query=query, top_k=top_k, where=where_filter)
                    if chroma_res:
                        results = []
                        with get_db() as sess:
                            repo = KnowledgeBaseRepository(sess)
                            for cr in chroma_res:
                                seg_uuid_str = cr["id"]
                                try:
                                    import uuid
                                    seg = repo.get_segment_by_id(uuid.UUID(seg_uuid_str))
                                except Exception:
                                    seg = None

                                if not seg or not seg.is_active:
                                    continue

                                vec_record = repo.session.query(SegmentVectorModel).filter(
                                    SegmentVectorModel.embedding_model == self.model_name,
                                    SegmentVectorModel.segment_id == seg.id
                                ).first()
                                fid = vec_record.faiss_vector_id if vec_record else -1

                                results.append({
                                    "segment_id": str(seg.id),
                                    "domain_id": str(seg.domain_id),
                                    "table_id": str(seg.table_id) if seg.table_id else None,
                                    "section": seg.section,
                                    "topic": seg.topic,
                                    "segment_type": seg.segment_type,
                                    "segment_name": seg.segment_name,
                                    "content": seg.content,
                                    "content_hash": seg.content_hash,
                                    "version": seg.version,
                                    "score": float(cr.get("similarity", 0.0)),
                                    "faiss_id": fid,
                                    "vector_engine": "chroma"
                                })
                        if results:
                            return results
            except Exception as ce:
                logger.warning(f"ChromaDB search encountered an issue; falling back to FAISS: {ce}")

        # 2. FAISS Path (Preserved 100% as robust fallback / parallel engine)
        if self.index is None or self.index.ntotal == 0:
            self._init_index()
            if self.index.ntotal == 0:
                return []

        model = get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        fetch_k = top_k * 4 if domain_id else top_k
        scores, ids = self.index.search(q_emb.astype(np.float32), fetch_k)

        results = []
        with get_db() as sess:
            repo = KnowledgeBaseRepository(sess)
            for rank in range(fetch_k):
                fid = int(ids[0][rank])
                if fid == -1:
                    continue
                score = float(scores[0][rank])

                vec_record = repo.get_segment_vector_by_faiss_id(self.model_name, fid)
                if not vec_record:
                    continue

                seg = repo.get_segment_by_id(vec_record.segment_id)
                if not seg or not seg.is_active:
                    continue

                # Filter by domain if requested
                if domain_id:
                    dom = repo.get_domain_by_id(seg.domain_id)
                    if not dom or dom.domain_id.lower() != domain_id.lower():
                        continue

                results.append({
                    "segment_id": str(seg.id),
                    "domain_id": str(seg.domain_id),
                    "table_id": str(seg.table_id) if seg.table_id else None,
                    "section": seg.section,
                    "topic": seg.topic,
                    "segment_type": seg.segment_type,
                    "segment_name": seg.segment_name,
                    "content": seg.content,
                    "content_hash": seg.content_hash,
                    "version": seg.version,
                    "score": score,
                    "faiss_id": fid,
                    "vector_engine": "faiss"
                })

                if len(results) >= top_k:
                    break

        return results


def sync_chroma_from_postgres(batch_size: int = 128) -> int:
    """
    Populates ChromaDB with all active knowledge segments from PostgreSQL.
    Ensures PostgreSQL and ChromaDB are in complete synchronization with
    structured domain_id:table_id:column_id identity keys and metadata.
    """
    chroma_engine = get_chroma_engine()
    model = get_embedding_model()

    with get_db() as sess:
        repo = KnowledgeBaseRepository(sess)
        from src.rag.db.models import DomainModel, KnowledgeTableModel
        dom_map = {d.id: d.domain_id for d in repo.session.query(DomainModel).all()}
        table_map = {t.id: t.table_name for t in repo.session.query(KnowledgeTableModel).all()}

        active_segs = repo.session.query(KnowledgeSegmentModel).filter(
            KnowledgeSegmentModel.is_active == True
        ).all()

        total = len(active_segs)
        logger.info(f"Synchronizing {total} active segments from PostgreSQL to ChromaDB...")

        for i in range(0, total, batch_size):
            batch = active_segs[i : i + batch_size]
            ids = [str(s.id) for s in batch]
            docs = [s.content for s in batch]
            metas = []
            for s in batch:
                d_code = dom_map.get(s.domain_id, "")
                t_name = table_map.get(s.table_id, "") if s.table_id else (
                    s.segment_name.split('.')[0] if '.' in s.segment_name and s.segment_type in ('table', 'column', 'relationship') else ""
                )
                ident_key = f"{d_code or str(s.domain_id)}:{t_name or 'none'}:{s.segment_name}"
                metas.append({
                    "domain_id": str(s.domain_id),
                    "domain_code": d_code,
                    "domain_name": s.section,
                    "table_id": str(s.table_id) if s.table_id else "",
                    "table_name": t_name,
                    "topic": s.topic,
                    "segment_type": s.segment_type,
                    "segment_name": s.segment_name,
                    "version": s.version,
                    "identity_key": ident_key
                })
            embs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            chroma_engine.upsert_batch(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

        logger.info(f"ChromaDB synchronized successfully! Total vectors in ChromaDB: {chroma_engine.count()}")
        return total


_FAISS_SYNC_INSTANCE: Optional[FAISSPostgresSynchronizer] = None


def get_faiss_synchronizer() -> FAISSPostgresSynchronizer:
    global _FAISS_SYNC_INSTANCE
    if _FAISS_SYNC_INSTANCE is None:
        _FAISS_SYNC_INSTANCE = FAISSPostgresSynchronizer()
    return _FAISS_SYNC_INSTANCE
