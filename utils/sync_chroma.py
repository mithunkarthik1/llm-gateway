"""
ChromaDB Vector Engine, Local Embeddings & PostgreSQL Synchronizer:
Consolidated utility for vector storage, similarity search, and automated 2-tier synchronization.
Scans PostgreSQL primary knowledge base against ChromaDB vectors, automatically identifies
any modified, new, or deleted segments using the 2-tier key architecture,
and re-indexes only the changed segments. Whenever ChromaDB is updated,
it automatically restarts the Docker service (nexus_gateway_app).

Usage:
    poetry run python utils/sync_chroma.py
"""
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import get_logger

logger = get_logger("utils.sync_chroma")

# Default Vector & Embedding Configuration Constants
CHROMA_DATA_DIR = Path("src/rag/data/chroma_db")
CHROMA_COLLECTION_NAME = "nexus_erp_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CHROMA_TOP_K = 10
CHROMA_SIMILARITY_THRESHOLD = 0.35

# =============================================================================
# Local Embedding Model (SentenceTransformer)
# =============================================================================

_EMBEDDING_MODEL: Any = None


def get_embedding_model():
    """
    Lazy loads SentenceTransformer locally on CPU with zero remote network delays.
    Generates normalized 384-dimensional dense vectors for semantic search.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: '{EMBEDDING_MODEL_NAME}'...")
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


# Load any overrides from config if available
try:
    from src.rag.config import (
        CHROMA_COLLECTION_NAME as _CFG_COLL_NAME,
        CHROMA_DATA_DIR as _CFG_DATA_DIR,
        CHROMA_SIMILARITY_THRESHOLD as _CFG_SIM_THRESH,
        CHROMA_TOP_K as _CFG_TOP_K,
        EMBEDDING_DIMENSION as _CFG_DIM,
        EMBEDDING_MODEL_NAME as _CFG_MODEL_NAME,
    )
    CHROMA_COLLECTION_NAME = _CFG_COLL_NAME
    CHROMA_DATA_DIR = _CFG_DATA_DIR
    CHROMA_SIMILARITY_THRESHOLD = _CFG_SIM_THRESH
    CHROMA_TOP_K = _CFG_TOP_K
    EMBEDDING_DIMENSION = _CFG_DIM
    EMBEDDING_MODEL_NAME = _CFG_MODEL_NAME
except Exception:
    pass


# =============================================================================
# ChromaDB Persistent Client & Collection Singletons
# =============================================================================

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.ERROR)

_CHROMA_CLIENT: Optional[Any] = None
_CHROMA_COLLECTION: Optional[Any] = None


def get_chroma_client():
    """Returns a singleton ChromaDB PersistentClient."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        import chromadb
        from chromadb.config import Settings
        CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing ChromaDB PersistentClient at: {CHROMA_DATA_DIR}")
        _CHROMA_CLIENT = chromadb.PersistentClient(
            path=str(CHROMA_DATA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
    return _CHROMA_CLIENT


def get_chroma_collection(collection_name: str = CHROMA_COLLECTION_NAME):
    """Returns the ChromaDB collection with cosine distance metric."""
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None or _CHROMA_COLLECTION.name != collection_name:
        client = get_chroma_client()
        logger.info(f"Getting or creating ChromaDB collection: '{collection_name}' (cosine metric)")
        _CHROMA_COLLECTION = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    return _CHROMA_COLLECTION


def reset_chroma_client() -> None:
    """Resets cached ChromaDB client, collection, engine, and synchronizer singletons."""
    global _CHROMA_CLIENT, _CHROMA_COLLECTION, _CHROMA_ENGINE, _SYNC_INSTANCE
    _CHROMA_CLIENT = None
    _CHROMA_COLLECTION = None
    _CHROMA_ENGINE = None
    _SYNC_INSTANCE = None


# =============================================================================
# ChromaSchemaEngine
# =============================================================================

class ChromaSchemaEngine:
    """
    Production ChromaDB Vector Engine for ERP Schema Knowledge Base.
    Features:
    - Persistent SQLite + HNSW storage in CHROMA_DATA_DIR.
    - Native metadata filtering by domain, table, segment_type, or identity_key.
    - Zero-loop O(1) single-chunk upserts and deletions by permanent UUID.
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        data_dir: Path = CHROMA_DATA_DIR,
        model_name: str = EMBEDDING_MODEL_NAME,
        similarity_threshold: float = CHROMA_SIMILARITY_THRESHOLD
    ):
        self.collection_name = collection_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._collection = get_chroma_collection(collection_name=self.collection_name)

    @property
    def collection(self):
        return get_chroma_collection(collection_name=self.collection_name)

    def count(self) -> int:
        """Returns total vector count in ChromaDB collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get ChromaDB collection count: {e}")
            return 0

    def upsert_segment(
        self,
        segment_id: str,
        document: str,
        metadata: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> None:
        """O(1) upsert for a single segment by permanent UUID."""
        if embedding is None:
            model = get_embedding_model()
            emb = model.encode([document], convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            emb = embedding

        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        clean_meta = {}
        for k, v in metadata.items():
            if v is not None:
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)

        self.collection.upsert(
            ids=[segment_id],
            documents=[document],
            metadatas=[clean_meta],
            embeddings=[emb_list]
        )

    def upsert_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """Batch upsert into ChromaDB."""
        if not ids:
            return

        if embeddings is None:
            model = get_embedding_model()
            embs = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        else:
            embs = embeddings

        emb_lists = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]
        clean_metas = []
        for m in metadatas:
            cm = {}
            for k, v in m.items():
                if v is not None:
                    if isinstance(v, (str, int, float, bool)):
                        cm[k] = v
                    else:
                        cm[k] = str(v)
            clean_metas.append(cm)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=clean_metas,
            embeddings=emb_lists
        )

    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a segment by its ID from ChromaDB."""
        try:
            res = self.collection.get(ids=[segment_id])
            if res and res.get("ids") and len(res["ids"]) > 0:
                doc = res["documents"][0] if res.get("documents") and len(res["documents"]) > 0 else ""
                meta = res["metadatas"][0] if res.get("metadatas") and len(res["metadatas"]) > 0 else {}
                return {"id": res["ids"][0], "document": doc, "metadata": meta}
        except Exception as e:
            logger.warning(f"Error retrieving segment '{segment_id}' from ChromaDB: {e}")
        return None

    def delete_segment(self, segment_id: str) -> None:
        """Deletes vector by permanent UUID from ChromaDB."""
        try:
            self.collection.delete(ids=[segment_id])
            logger.info(f"Deleted segment '{segment_id}' from ChromaDB.")
        except Exception as e:
            logger.warning(f"Failed deleting segment '{segment_id}' from ChromaDB: {e}")

    def reset_collection(self) -> None:
        """Flushes collection completely."""
        client = get_chroma_client()
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"Deleted ChromaDB collection '{self.collection_name}'.")
        except Exception:
            pass
        global _CHROMA_COLLECTION
        _CHROMA_COLLECTION = None
        self._collection = get_chroma_collection(self.collection_name)
        logger.info(f"Re-initialized empty ChromaDB collection '{self.collection_name}'.")

    def search(
        self,
        query: str,
        top_k: int = CHROMA_TOP_K,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic cosine distance search against ChromaDB.
        Converts cosine distance to similarity score: similarity = 1.0 - distance.
        Filters results by similarity_threshold.
        """
        if self.count() == 0:
            return []

        model = get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)

        query_kwargs = {
            "query_embeddings": [q_emb_list],
            "n_results": min(top_k, self.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            query_kwargs["where"] = where

        try:
            raw_res = self.collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"ChromaDB query execution error: {e}")
            return []

        results = []
        if not raw_res or not raw_res.get("ids") or not raw_res["ids"][0]:
            return results

        ids = raw_res["ids"][0]
        docs = raw_res["documents"][0] if raw_res.get("documents") else [""] * len(ids)
        metas = raw_res["metadatas"][0] if raw_res.get("metadatas") else [{}] * len(ids)
        dists = raw_res["distances"][0] if raw_res.get("distances") else [0.0] * len(ids)

        for seg_id, doc, meta, dist in zip(ids, docs, metas, dists):
            similarity = 1.0 - float(dist)
            if similarity >= self.similarity_threshold:
                results.append({
                    "id": seg_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity": similarity
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


_CHROMA_ENGINE: Optional[ChromaSchemaEngine] = None


def get_chroma_engine() -> ChromaSchemaEngine:
    """Returns singleton ChromaSchemaEngine."""
    global _CHROMA_ENGINE
    if _CHROMA_ENGINE is None:
        _CHROMA_ENGINE = ChromaSchemaEngine()
    return _CHROMA_ENGINE


# =============================================================================
# Docker Service Management
# =============================================================================

def restart_docker_service(container_name: str = "nexus_gateway_app") -> bool:
    """
    Restarts the Docker container running the FastAPI service so updated ChromaDB
    vectors and cached models are cleanly re-loaded.
    """
    if os.environ.get("IS_DOCKER", "false").lower() in ("true", "1") or os.environ.get("PYTEST_CURRENT_TEST"):
        return False

    docker_bin = shutil.which("docker")
    if not docker_bin:
        logger.warning("[DOCKER RESTART] Docker executable not found in PATH; skipping container restart.")
        return False

    try:
        check_proc = subprocess.run(
            [docker_bin, "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        running_names = [line.strip() for line in check_proc.stdout.splitlines() if line.strip()]
        if container_name not in running_names:
            logger.info(f"[DOCKER RESTART] Container '{container_name}' is not currently running. Skipping restart.")
            return False

        logger.info(f"[DOCKER RESTART] ChromaDB updated. Restarting Docker container '{container_name}'...")
        restart_proc = subprocess.run(
            [docker_bin, "restart", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if restart_proc.returncode == 0:
            logger.info(f"[DOCKER RESTART] Successfully restarted Docker container '{container_name}'.")
            return True
        else:
            logger.warning(f"[DOCKER RESTART] Docker restart exited with error: {restart_proc.stderr.strip()}")
            return False
    except Exception as exc:
        logger.warning(f"[DOCKER RESTART] Failed to restart Docker service '{container_name}': {exc}")
        return False


# =============================================================================
# ChromaPostgresSynchronizer (2-Tier Direct Synchronization)
# =============================================================================

class ChromaPostgresSynchronizer:
    """
    Direct, 2-Tier Knowledge Base Synchronizer:
    - Tier 1: Logical Business Segment Key (domain:table:column / domain:table / domain)
    - Tier 2: Permanent UUID segment_id
    Connects PostgreSQL ground truth directly with ChromaDB vectors with zero ghost chunks.
    """

    def __init__(self, chroma_engine: Optional[ChromaSchemaEngine] = None):
        self.chroma_engine = chroma_engine or get_chroma_engine()

    def process_pending_jobs(self, batch_size: int = 10, session=None) -> Dict[str, Any]:
        """Backward-compatible entry point for on-demand sync."""
        return self.auto_sync_changes(session=session)

    def reindex_segment(
        self,
        segment_id: uuid.UUID,
        session=None
    ) -> Dict[str, Any]:
        """
        Directly embeds and upserts a single segment into ChromaDB by permanent UUID.
        Guarantees zero ghost chunks by updating in place.
        """
        from src.rag.database import get_db, KnowledgeBaseRepository
        from src.rag.models import KnowledgeSegmentModel, DomainModel, KnowledgeTableModel

        def _do_reindex(sess):
            repo = KnowledgeBaseRepository(sess)
            seg = repo.get_segment_by_id(segment_id)
            if not seg:
                raise ValueError(f"Segment with id='{segment_id}' does not exist in PostgreSQL.")

            if not seg.is_active:
                self.chroma_engine.delete_segment(str(seg.id))
                return {
                    "segment_id": str(seg.id),
                    "segment_key": seg.segment_key,
                    "status": "deleted"
                }

            dom = sess.query(DomainModel).filter(DomainModel.id == seg.domain_id).first()
            dom_code = dom.domain_id if dom else ""
            table_name = ""
            if seg.table_id:
                t = sess.query(KnowledgeTableModel).filter(KnowledgeTableModel.id == seg.table_id).first()
                if t:
                    table_name = t.table_name

            ident_key = seg.segment_key or f"{dom_code}:{table_name or 'none'}:{seg.segment_name}"

            meta = {
                "domain_id": str(seg.domain_id),
                "domain_code": dom_code,
                "domain_name": seg.section,
                "table_id": str(seg.table_id) if seg.table_id else "",
                "table_name": table_name,
                "topic": seg.topic,
                "segment_type": seg.segment_type,
                "segment_name": seg.segment_name,
                "version": seg.version,
                "identity_key": ident_key,
                "content_hash": getattr(seg, "content_hash", ""),
            }

            self.chroma_engine.upsert_segment(
                segment_id=str(seg.id),
                document=seg.content,
                metadata=meta
            )
            return {
                "segment_id": str(seg.id),
                "segment_key": ident_key,
                "version": seg.version,
                "status": "upserted"
            }

        if session:
            return _do_reindex(session)
        else:
            with get_db() as sess:
                return _do_reindex(sess)

    def auto_sync_changes(self, session=None) -> Dict[str, Any]:
        """
        Full 2-Tier Change Detection & Synchronization:
        1. Compares active segments in PostgreSQL with existing ChromaDB vectors.
        2. Identifies:
           - Modified segments: Version higher or content changed
           - New segments: In PostgreSQL, missing in ChromaDB
           - Stale segments: In ChromaDB, inactive or deleted in PostgreSQL
        3. Re-indexes only modified/new segments and purges stale vectors.
        4. Triggers automatic Docker container restart if changes were applied.
        """
        from src.rag.database import KnowledgeBaseRepository
        from src.rag.models import KnowledgeSegmentModel, DomainModel, KnowledgeTableModel

        t_start = time.perf_counter()

        def _run(sess):
            logger.info("=" * 76)
            logger.info("  CHROMA-POSTGRES 2-TIER KNOWLEDGE SYNCHRONIZER")
            logger.info("=" * 76)
            logger.info("SYSTEM CONFIGURATION:")
            logger.info(f"  • PostgreSQL DB     : db_nexus_rag (Host: localhost:5432)")
            logger.info(f"  • ChromaDB Path     : {CHROMA_DATA_DIR}")
            logger.info(f"  • Target Collection : {CHROMA_COLLECTION_NAME}")
            logger.info(f"  • Embedding Model   : {EMBEDDING_MODEL_NAME}")
            logger.info(f"  • Architecture      : 2-Tier (Tier 1: Business segment_key | Tier 2: UUID segment_id)")
            logger.info(f"  • Strategy          : Zero-hash direct content & version verification")
            logger.info("─" * 76)

            # Phase 1: Scan
            logger.info("[PHASE 1: 2-TIER KNOWLEDGE SCAN & CHANGE DETECTION]")
            logger.info("  • Fetching live active segments from PostgreSQL...")
            pg_segments = sess.query(KnowledgeSegmentModel).filter(
                KnowledgeSegmentModel.is_active == True
            ).all()
            pg_count = len(pg_segments)

            logger.info(f"  • Fetching existing vector records from ChromaDB collection '{CHROMA_COLLECTION_NAME}'...")
            chroma_data = self.chroma_engine.collection.get(
                include=["metadatas", "documents"]
            )
            chroma_ids = set(chroma_data.get("ids", []))
            chroma_count = len(chroma_ids)

            logger.info(f"  - PostgreSQL Active Segments : {pg_count}")
            logger.info(f"  - ChromaDB Indexed Vectors   : {chroma_count}")
            logger.info("─" * 76)

            chroma_lookup = {}
            for i, cid in enumerate(chroma_data.get("ids", [])):
                meta = chroma_data["metadatas"][i] if chroma_data.get("metadatas") else {}
                doc = chroma_data["documents"][i] if chroma_data.get("documents") else ""
                chroma_lookup[cid] = {
                    "metadata": meta or {},
                    "document": doc or "",
                    "version": int(meta.get("version", 1)) if meta and "version" in meta else 1,
                    "content_hash": meta.get("content_hash", "") if meta else "",
                    "identity_key": meta.get("identity_key", "") if meta else ""
                }

            # Phase 2: Audit
            modified_segments = []
            new_segments = []
            pg_ids = set()

            for seg in pg_segments:
                sid_str = str(seg.id)
                pg_ids.add(sid_str)

                if sid_str not in chroma_lookup:
                    new_segments.append((seg, "New segment not found in ChromaDB"))
                else:
                    c_info = chroma_lookup[sid_str]
                    c_ver = c_info["version"]
                    c_doc = c_info["document"]
                    c_hash = c_info["content_hash"]

                    is_version_bumped = seg.version > c_ver
                    seg_hash = getattr(seg, "content_hash", "")
                    is_hash_mismatch = bool(seg_hash and c_hash and seg_hash != c_hash)
                    is_content_diff = (seg.content.strip() != c_doc.strip())

                    if is_version_bumped:
                        modified_segments.append((seg, f"Version bumped: v{c_ver} -> v{seg.version}"))
                    elif is_hash_mismatch:
                        modified_segments.append((seg, f"Content hash changed: {c_hash[:8]}... -> {seg.content_hash[:8]}..."))
                    elif is_content_diff:
                        modified_segments.append((seg, "Direct text content altered"))

            deleted_ids = list(chroma_ids - pg_ids)
            total_changes = len(modified_segments) + len(new_segments) + len(deleted_ids)

            logger.info("[PHASE 2: AUDIT & CHANGE EVALUATION]")
            logger.info(f"  - Total Changes Detected     : {total_changes}")
            logger.info(f"    * Modified Segments        : {len(modified_segments)}")
            logger.info(f"    * New Segments             : {len(new_segments)}")
            logger.info(f"    * Deleted/Stale Segments   : {len(deleted_ids)}")
            logger.info("─" * 76)

            if total_changes == 0:
                logger.info("  ✓ RESULT: PostgreSQL ground truth and ChromaDB vectors are in 100% PERFECT SYNC.")
                logger.info("  ✓ No re-indexing or embedding computations required.")
                logger.info("=" * 76)
                return {
                    "total_changes": 0,
                    "modified": 0,
                    "new": 0,
                    "deleted": 0,
                    "final_count": chroma_count,
                    "status": "UP_TO_DATE"
                }

            # Display Queued Changes
            logger.info(f"[CHANGES QUEUED FOR RE-INDEXING]:")
            idx = 1
            for seg, reason in modified_segments:
                logger.info(f"  [{idx}] MODIFIED: {seg.segment_key}")
                logger.info(f"       • Reason : {reason}")
                logger.info(f"       • UUID   : {seg.id}")
                logger.info(f"       • Entity : {seg.section} -> {seg.topic}")
                idx += 1
            for seg, reason in new_segments:
                logger.info(f"  [{idx}] NEW: {seg.segment_key}")
                logger.info(f"       • Reason : {reason}")
                logger.info(f"       • UUID   : {seg.id}")
                logger.info(f"       • Entity : {seg.section} -> {seg.topic}")
                idx += 1
            logger.info("─" * 76)

            # Purge Stale Vectors
            if deleted_ids:
                logger.info(f"[PURGING {len(deleted_ids)} STALE VECTOR(S)]")
                for did in deleted_ids:
                    logger.info(f"  • Removing stale segment ID '{did}' from ChromaDB...")
                    self.chroma_engine.delete_segment(did)
                logger.info("─" * 76)

            # Re-index
            to_reindex = modified_segments + new_segments
            logger.info(f"[PHASE 3: EXECUTING RE-INDEXING PIPELINE FOR {len(to_reindex)} SEGMENT(S)]")
            reindexed = []

            for i, (seg, reason) in enumerate(to_reindex, start=1):
                logger.info(f"\n>>> [ITEM {i}/{len(to_reindex)}] Re-indexing modified segment: '{seg.segment_key}' <<<")
                res = self.reindex_segment(seg.id, session=sess)
                reindexed.append(res)

            final_count = self.chroma_engine.count()
            elapsed = time.perf_counter() - t_start

            # Reload in-memory schema registry if present
            try:
                from src.rag.service import load_erp_registry
                load_erp_registry(force_reload=True)
            except Exception as e:
                logger.debug(f"Schema registry reload note: {e}")

            # Automatically restart Docker container whenever ChromaDB changes
            if total_changes > 0:
                try:
                    restart_docker_service()
                except Exception as e:
                    logger.warning(f"Could not trigger docker service restart: {e}")

            logger.info("\n" + "=" * 76)
            logger.info("  AUTO-SYNC COMPLETED SUCCESSFULLY")
            logger.info("=" * 76)
            logger.info(f"  • Overall Status      : SYNCHRONIZED")
            logger.info(f"  • PostgreSQL Records  : {pg_count} active segments")
            logger.info(f"  • Modified Re-indexed : {len(modified_segments)}")
            logger.info(f"  • New Vectors Added   : {len(new_segments)}")
            logger.info(f"  • Stale Vectors Purged: {len(deleted_ids)}")
            logger.info(f"  • Final ChromaDB Total: {final_count} vectors")
            logger.info(f"  • Execution Time      : {elapsed:.3f} seconds")
            logger.info("=" * 76)

            return {
                "total_changes": total_changes,
                "modified": len(modified_segments),
                "new": len(new_segments),
                "deleted": len(deleted_ids),
                "reindexed_segments": [r["segment_key"] for r in reindexed],
                "final_count": final_count,
                "elapsed_seconds": round(elapsed, 3),
                "status": "SYNCHRONIZED"
            }

        if session:
            return _run(session)
        else:
            from src.rag.database import get_db
            with get_db() as sess:
                return _run(sess)

    def search_active_segments(
        self,
        query: str,
        top_k: int = CHROMA_TOP_K,
        domain_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Executes semantic search against ChromaDB resolved to PostgreSQL segments."""
        where_filter = None
        if domain_id:
            if len(domain_id) == 36 and "-" in domain_id:
                where_filter = {"domain_id": domain_id}
            else:
                where_filter = {"domain_code": domain_id.lower()}

        chroma_res = self.chroma_engine.search(query=query, top_k=top_k, where=where_filter)
        results = []
        if not chroma_res:
            return results

        from src.rag.database import get_db, KnowledgeBaseRepository

        with get_db() as sess:
            repo = KnowledgeBaseRepository(sess)
            for cr in chroma_res:
                seg_uuid_str = cr["id"]
                try:
                    seg = repo.get_segment_by_id(uuid.UUID(seg_uuid_str))
                except Exception:
                    seg = None

                if not seg or not seg.is_active:
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
                    "version": seg.version,
                    "score": float(cr.get("similarity", 0.0)),
                    "identity_key": cr.get("metadata", {}).get("identity_key", ""),
                    "vector_engine": "chroma",
                })

        return results


def sync_chroma_from_postgres(batch_size: int = 128) -> int:
    """Populates ChromaDB with all active knowledge segments from PostgreSQL."""
    from src.rag.database import get_db, KnowledgeBaseRepository
    from src.rag.models import KnowledgeSegmentModel, DomainModel, KnowledgeTableModel

    chroma_engine = get_chroma_engine()
    model = get_embedding_model()

    with get_db() as sess:
        repo = KnowledgeBaseRepository(sess)
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
                    s.segment_name.split(".")[0]
                    if "." in s.segment_name and s.segment_type in ("table", "column", "relationship")
                    else ""
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
                    "identity_key": ident_key,
                })
            embs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            chroma_engine.upsert_batch(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

        logger.info(f"ChromaDB synchronized successfully! Total vectors in ChromaDB: {chroma_engine.count()}")
        return total


def auto_sync_chroma() -> Dict[str, Any]:
    """
    Zero-argument change-detection synchronizer:
    Automatically identifies any altered, added, or deleted segments in PostgreSQL
    and synchronizes ChromaDB.
    """
    return get_embedding_synchronizer().auto_sync_changes()


_SYNC_INSTANCE: Optional[ChromaPostgresSynchronizer] = None


def get_embedding_synchronizer() -> ChromaPostgresSynchronizer:
    global _SYNC_INSTANCE
    if _SYNC_INSTANCE is None:
        _SYNC_INSTANCE = ChromaPostgresSynchronizer()
    return _SYNC_INSTANCE


def run_sync(container_name: str = "nexus_gateway_app") -> int:
    """Executes full automated 2-tier synchronization from PostgreSQL truth to ChromaDB."""
    try:
        result = auto_sync_chroma()
        total_changes = result.get("total_changes", 0) if isinstance(result, dict) else 0
        if total_changes == 0:
            logger.info("[SYNC] ChromaDB is fully synchronized with PostgreSQL. No container restart required.")
        return 0
    except Exception as e:
        logger.error(f"[AUTO-SYNC FATAL ERROR] Failed to complete sync: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synchronize PostgreSQL Knowledge Base to ChromaDB Vector Engine.")
    parser.add_argument("--container", type=str, default="nexus_gateway_app", help="Docker container name to restart on update.")
    args = parser.parse_args()
    sys.exit(run_sync(container_name=args.container))
