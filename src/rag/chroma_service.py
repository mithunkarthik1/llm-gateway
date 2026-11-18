import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Silence chromadb posthog telemetry mismatch
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.ERROR)

import chromadb
from chromadb.config import Settings

from src.config import get_logger
from src.rag.config import (
    CHROMA_DATA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from src.rag.faiss_service import get_embedding_model

logger = get_logger(__name__)

_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None
_CHROMA_COLLECTION: Optional[Any] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a singleton ChromaDB PersistentClient."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
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


class ChromaSchemaEngine:
    """
    Production ChromaDB Vector Engine for ERP Schema Knowledge Base.
    Features:
    - Persistent SQLite + HNSW storage in CHROMA_DATA_DIR.
    - Native metadata filtering by domain, table, segment_type, or identity_key.
    - Zero-loop O(1) single-chunk upserts and deletions by permanent UUID.
    - Normalized cosine vector similarity using local sentence-transformers.
    """

    def __init__(self, data_dir: Path = CHROMA_DATA_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self.data_dir = data_dir
        self.collection_name = collection_name
        self.client = get_chroma_client()
        self.collection = get_chroma_collection(collection_name)

    def count(self) -> int:
        """Returns the total number of vectors in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.warning(f"Error getting ChromaDB count: {e}")
            return 0

    def upsert_segment(
        self,
        segment_id: str,
        document: str,
        metadata: Dict[str, Any],
        embedding: Optional[Union[List[float], np.ndarray]] = None
    ):
        """
        Upserts a single segment in-place by ID.
        If the ID exists, updates document, metadata, and vector in place (Zero ghost chunks).
        """
        # Ensure metadata values are primitive strings, ints, floats, or bools
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = ", ".join(str(item) for item in v)
            elif v is not None:
                clean_meta[k] = str(v)

        if embedding is None:
            model = get_embedding_model()
            emb = model.encode(document, convert_to_numpy=True, normalize_embeddings=True)
            emb_list = emb.tolist()
        elif isinstance(embedding, np.ndarray):
            emb_list = embedding.tolist()
        else:
            emb_list = embedding

        self.collection.upsert(
            ids=[str(segment_id)],
            documents=[document],
            metadatas=[clean_meta],
            embeddings=[emb_list]
        )

    def upsert_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[Union[List[List[float]], np.ndarray]] = None
    ):
        """Batch upsert multiple segments with optional pre-computed embeddings."""
        if not ids:
            return

        clean_metadatas = []
        for meta in metadatas:
            clean_m = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                elif isinstance(v, list):
                    clean_m[k] = ", ".join(str(item) for item in v)
                elif v is not None:
                    clean_m[k] = str(v)
            clean_metadatas.append(clean_m)

        if embeddings is None:
            model = get_embedding_model()
            emb_arr = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
            emb_list = emb_arr.tolist()
        elif isinstance(embeddings, np.ndarray):
            emb_list = embeddings.tolist()
        else:
            emb_list = embeddings

        self.collection.upsert(
            ids=[str(i) for i in ids],
            documents=documents,
            metadatas=clean_metadatas,
            embeddings=emb_list
        )

    def delete_segment(self, segment_id: str):
        """Deletes a segment by ID from ChromaDB."""
        try:
            self.collection.delete(ids=[str(segment_id)])
            logger.info(f"Deleted segment '{segment_id}' from ChromaDB.")
        except Exception as e:
            logger.warning(f"Error deleting segment '{segment_id}' from ChromaDB: {e}")

    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a specific segment by its permanent ID."""
        try:
            res = self.collection.get(ids=[str(segment_id)], include=["documents", "metadatas"])
            if res and res.get("ids") and len(res["ids"]) > 0:
                return {
                    "id": res["ids"][0],
                    "document": res["documents"][0] if res.get("documents") else "",
                    "metadata": res["metadatas"][0] if res.get("metadatas") else {}
                }
        except Exception as e:
            logger.warning(f"Error fetching segment '{segment_id}' from ChromaDB: {e}")
        return None

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search with native metadata filtering.
        Example where clauses:
        - where={"domain_id": "erp_finance"}
        - where={"table_name": "customer_invoices"}
        - where={"$and": [{"domain_id": "erp_hrms"}, {"segment_type": "column"}]}
        """
        model = get_embedding_model()
        query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        emb_list = query_emb.tolist()

        kwargs: Dict[str, Any] = {
            "query_embeddings": [emb_list],
            "n_results": min(top_k, max(self.count(), 1)),
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        try:
            raw_res = self.collection.query(**kwargs)
        except Exception as e:
            logger.error(f"ChromaDB search query failed: {e}")
            return []

        results: List[Dict[str, Any]] = []
        if not raw_res or not raw_res.get("ids") or len(raw_res["ids"]) == 0:
            return results

        ids = raw_res["ids"][0]
        docs = raw_res.get("documents", [[]])[0]
        metas = raw_res.get("metadatas", [[]])[0]
        dists = raw_res.get("distances", [[]])[0]

        for i, seg_id in enumerate(ids):
            # For cosine distance: similarity = 1 - distance
            dist = dists[i] if i < len(dists) else 1.0
            similarity = max(0.0, 1.0 - dist)
            results.append({
                "id": seg_id,
                "similarity": similarity,
                "distance": dist,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {}
            })

        return results


_CHROMA_ENGINE: Optional[ChromaSchemaEngine] = None


def get_chroma_engine() -> ChromaSchemaEngine:
    """Returns singleton ChromaSchemaEngine."""
    global _CHROMA_ENGINE
    if _CHROMA_ENGINE is None:
        _CHROMA_ENGINE = ChromaSchemaEngine()
    return _CHROMA_ENGINE
