import os
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.config import get_logger
from src.rag.config import (
    FAISS_INDEX_DIR,
    DOMAINS_DATA_DIR,
    EMBEDDING_MODEL_NAME,
    FAISS_TOP_K,
    FAISS_SIMILARITY_THRESHOLD,
)

logger = get_logger(__name__)

# Lazy global model and index holders
_EMBEDDING_MODEL: Any = None
_FAISS_INDEX: Any = None
_FAISS_METADATA: Optional[List[Dict[str, Any]]] = None


def get_embedding_model():
    """Lazy loads SentenceTransformer locally without remote network delays."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: '{EMBEDDING_MODEL_NAME}'...")
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


class FAISSSchemaEngine:
    """
    High-performance, in-memory FAISS vector engine for granular table-wise
    and column-wise semantic search and relational dependency merging.
    """

    def __init__(self, index_dir: Path = FAISS_INDEX_DIR):
        self.index_dir = index_dir
        self.index_file = self.index_dir / "index.faiss"
        self.meta_file = self.index_dir / "metadata.pkl"
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._load_or_build()

    def _load_or_build(self):
        """Loads existing FAISS index or builds from granular JSONs if missing."""
        if self.index_file.exists() and self.meta_file.exists():
            try:
                import faiss
                self.index = faiss.read_index(str(self.index_file))
                with open(self.meta_file, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index ({self.index.ntotal} vectors) from {self.index_dir}.")
                return
            except Exception as e:
                logger.warning(f"Could not load FAISS index ({e}). Rebuilding...")

        self.build_index()

    def build_index(self) -> int:
        """
        Parses all granular table JSONs, builds table-level & column-level
        documents, encodes with SentenceTransformer, and creates FAISS IndexFlatIP.
        """
        import faiss

        logger.info("Building FAISS granular table & column vector index...")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        documents: List[str] = []
        meta_list: List[Dict[str, Any]] = []

        # 1. Discover all domain, table, and column JSON files
        domain_dirs = [d for d in DOMAINS_DATA_DIR.iterdir() if d.is_dir()] if DOMAINS_DATA_DIR.exists() else []
        logger.info(f"Indexing schema architecture from {len(domain_dirs)} domain directories in {DOMAINS_DATA_DIR}...")

        total_tables = 0
        total_columns = 0

        for dom_dir in sorted(domain_dirs):
            dom_id = dom_dir.name
            domain_meta_file = dom_dir / "domain.json"
            domain_info = {}
            if domain_meta_file.exists():
                try:
                    with open(domain_meta_file, "r", encoding="utf-8") as df:
                        domain_info = json.load(df)
                except Exception as e:
                    logger.warning(f"Error reading domain meta {domain_meta_file}: {e}")

            system = domain_info.get("system", "Enterprise ERP")
            domain_name = domain_info.get("domain_name", dom_id)

            # --- A. Table & Column Files ---
            tables_v2_file = dom_dir / "tables.json"
            columns_v2_file = dom_dir / "columns.json"
            tables_dir = dom_dir / "tables"
            columns_dir = dom_dir / "columns"

            if tables_v2_file.exists() and columns_v2_file.exists():
                try:
                    with open(tables_v2_file, "r", encoding="utf-8") as tf:
                        raw_tables = json.load(tf)
                    for tbl in raw_tables:
                        total_tables += 1
                        tname = tbl.get("table_name", "")
                        desc = tbl.get("description", "")
                        keywords = tbl.get("keywords", [])
                        deps = tbl.get("dependencies", [])
                        raw_sql = tbl.get("sql", "")
                        cols = tbl.get("columns", [])

                        tbl_base_meta = {
                            "type": "table",
                            "table_name": tname,
                            "domain": dom_id,
                            "domain_name": domain_name,
                            "system": system,
                            "description": desc,
                            "dependencies": deps,
                            "sql": raw_sql,
                            "columns": cols,
                            "keywords": keywords
                        }

                        # Perspective 1: Full Composite & Technical (Architecture, DDL, and Dependencies)
                        p1_text = (
                            f"System: {system} | Domain: {domain_name} ({dom_id}) | "
                            f"Table: {tname} | Dependencies: {', '.join(deps) if deps else 'None'} | "
                            f"Columns: {', '.join(cols)} | DDL: {raw_sql}"
                        )
                        documents.append(p1_text)
                        meta_list.append({**tbl_base_meta, "perspective": "technical"})

                        # Perspective 2: Business Semantics & Domain Narrative
                        p2_text = (
                            f"System: {system} | Domain: {domain_name} ({dom_id}) | "
                            f"Entity: {tname.replace('_', ' ')} | Description: {desc} | "
                            f"Keywords: {', '.join(keywords)}"
                        )
                        documents.append(p2_text)
                        meta_list.append({**tbl_base_meta, "perspective": "business"})

                        # Perspective 3: Natural Language User Query Intent & Phrasing
                        kw_synonyms = [k for k in keywords if k.lower() != tname.lower()]
                        p3_text = (
                            f"Query: create {tname} table | setup {tname.replace('_', ' ')} schema | "
                            f"show {tname} columns | Domain: {domain_name} | "
                            f"Synonyms: {', '.join(kw_synonyms)} | Purpose: {desc}"
                        )
                        documents.append(p3_text)
                        meta_list.append({**tbl_base_meta, "perspective": "query_intent"})

                    with open(columns_v2_file, "r", encoding="utf-8") as cf:
                        raw_cols = json.load(cf)
                    for col_data in raw_cols:
                        total_columns += 1
                        tname = col_data.get("table_name", "")
                        cname = col_data.get("column_name", "")
                        dtype = col_data.get("data_type", "")
                        cdesc = col_data.get("description", "")
                        ckw = col_data.get("keywords", [])

                        col_base_meta = {
                            "type": "column",
                            "table_name": tname,
                            "column_name": cname,
                            "domain": dom_id,
                            "domain_name": domain_name,
                            "data_type": dtype,
                            "description": cdesc,
                            "is_primary_key": col_data.get("is_primary_key", False),
                            "is_foreign_key": col_data.get("is_foreign_key", False),
                            "references": col_data.get("references"),
                            "keywords": ckw
                        }

                        # Perspective 1: Technical Schema Signature & Referential Wire
                        col_p1_text = (
                            f"Table: {tname} | Column: {cname} | Data Type: {dtype} | Domain: {domain_name} | "
                            f"References: {col_data.get('references') or 'None'}"
                        )
                        documents.append(col_p1_text)
                        meta_list.append({**col_base_meta, "perspective": "technical"})

                        # Perspective 2: Business Meaning, Synonyms & Natural Language Phrasing
                        col_p2_text = (
                            f"Table: {tname} | Column: {cname} ({cname.replace('_', ' ')}) | "
                            f"Description: {cdesc} | Synonyms: {', '.join(ckw)}"
                        )
                        documents.append(col_p2_text)
                        meta_list.append({**col_base_meta, "perspective": "business"})
                except Exception as e:
                    logger.error(f"Error indexing v2 files in {dom_dir}: {e}")

            elif tables_dir.exists():
                for tbl_file in sorted(tables_dir.glob("*.json")):
                    try:
                        with open(tbl_file, "r", encoding="utf-8") as tf:
                            tbl = json.load(tf)
                            total_tables += 1
                            tname = tbl.get("table_name", "")
                            desc = tbl.get("description", "")
                            keywords = tbl.get("keywords", [])
                            deps = tbl.get("dependencies", [])
                            raw_sql = tbl.get("sql", "")
                            cols = tbl.get("columns", [])

                            table_text = (
                                f"System: {system} | Domain: {domain_name} ({dom_id}) | "
                                f"Table: {tname} | Description: {desc} | "
                                f"Keywords: {', '.join(keywords)} | Columns: {', '.join(cols)}"
                            )
                            documents.append(table_text)
                            meta_list.append({
                                "type": "table",
                                "table_name": tname,
                                "domain": dom_id,
                                "domain_name": domain_name,
                                "system": system,
                                "description": desc,
                                "dependencies": deps,
                                "sql": raw_sql,
                                "columns": cols,
                                "keywords": keywords
                            })
                    except Exception as e:
                        logger.error(f"Error indexing table file {tbl_file}: {e}")

                if columns_dir.exists():
                    for col_file in sorted(columns_dir.rglob("*.json")):
                        try:
                            with open(col_file, "r", encoding="utf-8") as cf:
                                col_data = json.load(cf)
                                total_columns += 1
                                tname = col_data.get("table_name", "")
                                cname = col_data.get("column_name", "")
                                dtype = col_data.get("data_type", "")
                                cdesc = col_data.get("description", "")
                                ckw = col_data.get("keywords", [])

                                col_text = (
                                    f"System: {system} | Domain: {domain_name} ({dom_id}) | "
                                    f"Table: {tname} | Column: {cname} | Type: {dtype} | "
                                    f"Description: {cdesc} | Keywords: {', '.join(ckw)}"
                                )
                                documents.append(col_text)
                                meta_list.append({
                                    "type": "column",
                                    "table_name": tname,
                                    "column_name": cname,
                                    "domain": dom_id,
                                    "domain_name": domain_name,
                                    "data_type": dtype,
                                    "description": cdesc,
                                    "is_primary_key": col_data.get("is_primary_key", False),
                                    "is_foreign_key": col_data.get("is_foreign_key", False),
                                    "references": col_data.get("references"),
                                    "keywords": ckw
                                })
                        except Exception as e:
                            logger.error(f"Error indexing column file {col_file}: {e}")

        logger.info(f"Encoding {len(documents)} vectors ({total_tables} tables + {total_columns} columns across {len(domain_dirs)} domains)...")
        model = get_embedding_model()
        embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Cosine similarity for normalized vectors
        self.index.add(embeddings.astype(np.float32))
        self.metadata = meta_list

        # Save to disk
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Successfully saved FAISS index to {self.index_file} ({self.index.ntotal} vectors).")
        return self.index.ntotal

    def search(self, query: str, top_k: int = 5, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes semantic search against the FAISS index.
        Returns list of matched chunks with cosine similarity score.
        Supports optional domain filtering.
        """
        try:
            if self.index is None or not self.metadata:
                self._load_or_build()

            fetch_k = top_k * 3 if domain else top_k
            model = get_embedding_model()
            q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores, indices = self.index.search(q_emb.astype(np.float32), fetch_k)

            results = []
            for rank in range(fetch_k):
                idx = indices[0][rank]
                if idx == -1 or idx >= len(self.metadata):
                    continue
                item = dict(self.metadata[idx])
                item["score"] = float(scores[0][rank])
                if domain and item.get("domain") != domain:
                    continue
                results.append(item)
                if len(results) >= top_k:
                    break

            return results
        except Exception as e:
            logger.error(f"Error during FAISS semantic search for query '{query}': {e}", exc_info=True)
            return []

    def search_top1_table(self, query: str, top_k: int = 16) -> Optional[Dict[str, Any]]:
        """
        Finds the single best-matching primary table (Top-1) across multi-perspective
        table vectors and collects any specific matching columns from the top results.
        """
        try:
            results = self.search(query=query, top_k=top_k)
            if not results:
                return None

            top_table = None
            matched_columns: List[str] = []

            for r in results:
                if r.get("type") == "table" and top_table is None:
                    top_table = r
                elif r.get("type") == "column":
                    col_name = r.get("column_name")
                    if col_name and col_name not in matched_columns:
                        matched_columns.append(col_name)

            if top_table:
                top_table["matched_columns"] = matched_columns
                return top_table

            # Fallback to table corresponding to top column
            top_col = results[0]
            parent_tname = top_col.get("table_name")
            for m in self.metadata:
                if m.get("type") == "table" and m.get("table_name") == parent_tname:
                    res = dict(m)
                    res["score"] = top_col.get("score", 0.0)
                    res["matched_columns"] = [top_col.get("column_name")]
                    return res

            return None
        except Exception as e:
            logger.error(f"Error finding top-1 table for query '{query}': {e}", exc_info=True)
            return None


# Global singleton instance
_FAISS_ENGINE_INSTANCE: Optional[FAISSSchemaEngine] = None


def get_faiss_engine() -> FAISSSchemaEngine:
    global _FAISS_ENGINE_INSTANCE
    if _FAISS_ENGINE_INSTANCE is None:
        _FAISS_ENGINE_INSTANCE = FAISSSchemaEngine()
    return _FAISS_ENGINE_INSTANCE


if __name__ == "__main__":
    logger.info("Running manual FAISS schema indexing via CLI...")
    engine = get_faiss_engine()
    total = engine.build_index()
    print(f"FAISS indexing complete! Indexed {total} granular vectors.")

