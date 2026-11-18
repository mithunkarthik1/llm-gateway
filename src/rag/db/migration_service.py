import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy.orm import Session

from src.config import get_logger
from src.rag.config import DOMAINS_DATA_DIR
from src.rag.db.database import get_db, init_db
from src.rag.db.repository import KnowledgeBaseRepository
from src.rag.db.chunking import (
    compute_sha256,
    generate_stable_segment_uuid,
    build_domain_chunk,
    build_table_chunk,
    build_column_chunk,
    build_relationship_chunk,
    KnowledgeChunk,
)

logger = get_logger(__name__)


class JSONKnowledgeBaseMigrator:
    """
    Production-grade, idempotent migration engine from JSON knowledge-base
    files into PostgreSQL as the primary source of truth.
    """

    def __init__(self, data_dir: Path = DOMAINS_DATA_DIR, session: Optional[Session] = None):
        self.data_dir = data_dir
        self._external_session = session

    def run_migration(self, force_rebuild_segments: bool = False) -> Dict[str, Any]:
        """
        Executes full, transactional migration of all domains, tables, columns,
        dependencies, and hybrid segments.
        """
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

        # ---------------------------------------------------------------------
        # Stage 1: Ingest Domains, Tables, and Columns
        # ---------------------------------------------------------------------
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

            # 1. Upsert Domain
            dom_model, d_created = repo.upsert_domain(
                domain_id=domain_data.get("domain_id", dom_id),
                domain_name=domain_data.get("domain_name", dom_id),
                system_name=domain_data.get("system", "Enterprise ERP"),
                description=domain_data.get("description", ""),
                keywords=domain_data.get("keywords", []),
                is_active=True
            )
            if d_created:
                stats["domains_created"] += 1
            else:
                stats["domains_updated"] += 1

            # Domain-level chunk
            all_chunks.append(build_domain_chunk(domain_data))

            # 2. Upsert Tables
            if tables_file.exists():
                try:
                    with open(tables_file, "r", encoding="utf-8") as tf:
                        tables_list = json.load(tf)
                except Exception as e:
                    stats["errors"].append(f"Failed to read {tables_file}: {e}")
                    tables_list = []

                for tbl in tables_list:
                    tname = tbl.get("table_name", "").lower().strip()
                    sql_def = tbl.get("sql", "").strip()
                    desc = tbl.get("description", "")
                    keywords = tbl.get("keywords", [])
                    thash = compute_sha256(f"{sql_def}:{desc}")

                    tbl_model, t_created = repo.upsert_table(
                        domain_uuid=dom_model.id,
                        table_name=tname,
                        sql_definition=sql_def,
                        content_hash=thash,
                        description=desc,
                        keywords=keywords,
                        is_active=True
                    )
                    table_obj_map[tname] = tbl_model
                    if t_created:
                        stats["tables_created"] += 1
                    else:
                        stats["tables_updated"] += 1

                    # Table-level chunk
                    all_chunks.append(build_table_chunk(domain_data, tbl))

            # 3. Upsert Columns
            if columns_file.exists():
                try:
                    with open(columns_file, "r", encoding="utf-8") as cf:
                        columns_list = json.load(cf)
                except Exception as e:
                    stats["errors"].append(f"Failed to read {columns_file}: {e}")
                    columns_list = []

                for col in columns_list:
                    tname = col.get("table_name", "").lower().strip()
                    tbl_model = table_obj_map.get(tname)
                    if not tbl_model:
                        continue

                    cname = col.get("column_name", "").lower().strip()
                    dtype = col.get("data_type", "").strip()
                    desc = col.get("description", "")
                    is_pk = bool(col.get("is_primary_key", False))
                    is_fk = bool(col.get("is_foreign_key", False))
                    ref_str = col.get("references")
                    ref_tbl = None
                    ref_col = None
                    if ref_str and "(" in ref_str:
                        parts = ref_str.split("(")
                        ref_tbl = parts[0].strip()
                        ref_col = parts[1].replace(")", "").strip()

                    col_model, c_created = repo.upsert_column(
                        table_uuid=tbl_model.id,
                        column_name=cname,
                        data_type=dtype,
                        description=desc,
                        is_primary_key=is_pk,
                        is_foreign_key=is_fk,
                        references_table=ref_tbl,
                        references_column=ref_col,
                        keywords=col.get("keywords", []),
                        is_active=True
                    )
                    if c_created:
                        stats["columns_created"] += 1
                    else:
                        stats["columns_updated"] += 1

                    # Column-level chunk
                    all_chunks.append(build_column_chunk(domain_data, col))

                    # Relationship chunk if FK
                    if is_fk and ref_tbl:
                        all_chunks.append(build_relationship_chunk(
                            domain_data=domain_data,
                            source_table=tname,
                            target_table=ref_tbl,
                            fk_col=cname
                        ))

        # ---------------------------------------------------------------------
        # Stage 2: Resolve Table Dependencies
        # ---------------------------------------------------------------------
        for dom_dir in sorted(domain_dirs):
            tables_file = dom_dir / "tables.json"
            if not tables_file.exists():
                continue
            with open(tables_file, "r", encoding="utf-8") as tf:
                tables_list = json.load(tf)
            for tbl in tables_list:
                src_tname = tbl.get("table_name", "").lower().strip()
                src_model = table_obj_map.get(src_tname)
                if not src_model:
                    continue
                for dep_name in tbl.get("dependencies", []):
                    target_tname = dep_name.lower().strip()
                    target_model = table_obj_map.get(target_tname)
                    if target_model:
                        _, dep_created = repo.add_table_dependency(
                            source_table_id=src_model.id,
                            referenced_table_id=target_model.id,
                            relationship_type="FOREIGN_KEY"
                        )
                        if dep_created:
                            stats["dependencies_created"] += 1

        # ---------------------------------------------------------------------
        # Stage 3: Ingest Hybrid Segments & Change Detection
        # ---------------------------------------------------------------------
        logger.info(f"Processing {len(all_chunks)} hybrid knowledge segments...")
        for chunk in all_chunks:
            dom = repo.get_domain_by_business_id(chunk.domain_id, active_only=False)
            if not dom:
                continue

            tbl_model = table_obj_map.get(chunk.table_name.lower()) if chunk.table_name else None

            seg, op = repo.upsert_segment(
                segment_uuid=chunk.segment_uuid,
                domain_id=dom.id,
                table_id=tbl_model.id if tbl_model else None,
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
                repo.create_embedding_job(
                    segment_id=seg.id,
                    content_hash=seg.content_hash,
                    operation="INSERT"
                )
                stats["embedding_jobs_created"] += 1
            elif op == "UPDATE":
                stats["segments_updated"] += 1
                repo.create_embedding_job(
                    segment_id=seg.id,
                    content_hash=seg.content_hash,
                    operation="UPDATE"
                )
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


def run_full_migration(db_url: Optional[str] = None) -> Dict[str, Any]:
    """CLI / Service helper to initialize tables and run migration."""
    init_db(db_url)
    migrator = JSONKnowledgeBaseMigrator()
    return migrator.run_migration()
