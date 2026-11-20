"""
Executable Knowledge Base Loader for PostgreSQL:
Scans JSON domain documents from the domain data directory and stores/updates them
into PostgreSQL using the 4 minimal, user-friendly tables:
  1. domains             (Domain metadata & keywords)
  2. knowledge_tables    (Tables & SQL DDL definitions)
  3. knowledge_columns   (Columns, data types, PKs & foreign-key references)
  4. knowledge_segments  (Semantic hybrid chunks for search & vector embedding)

Usage:
    poetry run python utils/load_domains_to_postgres.py
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import get_logger
from src.rag.config import DOMAINS_DATA_DIR
from src.rag.database import (
    get_db,
    init_db,
    is_db_connected,
    JSONKnowledgeBaseMigrator,
    KnowledgeBaseRepository,
)
from src.rag.models import (
    DomainModel,
    KnowledgeTableModel,
    KnowledgeColumnModel,
    KnowledgeSegmentModel,
)

logger = get_logger("utils.load_domains_to_postgres")


def load_domains_to_postgres(data_dir: Optional[Path] = None) -> int:
    """
    Ingests all JSON domain specifications into PostgreSQL.
    """
    target_dir = Path(data_dir) if data_dir else DOMAINS_DATA_DIR
    logger.info("=" * 76)
    logger.info("  POSTGRESQL KNOWLEDGE BASE INGESTION (Ground Truth)")
    logger.info("=" * 76)
    logger.info(f"  • Source Directory : {target_dir}")
    logger.info("  • Target Database  : PostgreSQL (db_nexus_rag)")
    logger.info("  • Core Tables      : domains, knowledge_tables, knowledge_columns, knowledge_segments")
    logger.info("─" * 76)

    if not is_db_connected():
        logger.error("[FATAL] Cannot connect to PostgreSQL database. Check connection settings.")
        return 1

    # Ensure schema initialized
    init_db()

    t0 = time.perf_counter()
    migrator = JSONKnowledgeBaseMigrator(data_dir=target_dir)
    stats = migrator.run_migration()

    elapsed = time.perf_counter() - t0

    # Query current counts from database
    with get_db() as sess:
        dom_count = sess.query(DomainModel).filter(DomainModel.is_active == True).count()
        tbl_count = sess.query(KnowledgeTableModel).filter(KnowledgeTableModel.is_active == True).count()
        col_count = sess.query(KnowledgeColumnModel).filter(KnowledgeColumnModel.is_active == True).count()
        seg_count = sess.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.is_active == True).count()

    logger.info("─" * 76)
    logger.info("INGESTION SUMMARY:")
    logger.info(f"  • Domains in PostgreSQL  : {dom_count} active")
    logger.info(f"  • Tables in PostgreSQL   : {tbl_count} active")
    logger.info(f"  • Columns in PostgreSQL  : {col_count} mapped")
    logger.info(f"  • Segments in PostgreSQL : {seg_count} persisted chunks")
    logger.info(f"  • Changes Detected       : {stats['segments_created']} created, {stats['segments_updated']} updated, {stats['segments_skipped']} unchanged")
    logger.info(f"  • Ingestion Duration     : {elapsed:.3f} seconds")
    logger.info("=" * 76)
    logger.info("Ground truth is up to date in PostgreSQL!")
    logger.info("Next step: Run 'poetry run python utils/sync_chroma.py' to synchronize ChromaDB vectors.")
    logger.info("=" * 76)
    return 0


def alter_column_in_postgres(
    table_name: str,
    column_name: str,
    data_type: Optional[str] = None,
    is_primary_key: Optional[bool] = None,
    sync: bool = False
) -> int:
    """
    Alters a column in PostgreSQL ground truth.
    The database trigger automatically cascades changes to:
      1. knowledge_tables.sql_definition (regenerated DDL)
      2. knowledge_segments (column & table chunks + version bump)
    Optionally synchronizes ChromaDB and restarts Docker if sync=True.
    """
    from sqlalchemy import text
    from src.rag.database import get_db, is_db_connected
    if not is_db_connected():
        logger.error("[FATAL] Cannot connect to PostgreSQL database.")
        return 1

    with get_db() as sess:
        res = sess.execute(
            text("SELECT alter_column(:t, :c, :dt, :pk)"),
            {"t": table_name, "c": column_name, "dt": data_type, "pk": is_primary_key}
        ).scalar()
        sess.commit()
        logger.info(f"[ALTER SUCCESS] {res}")

    if sync:
        from utils.sync_chroma import run_sync
        return run_sync()
    else:
        logger.info("Run 'poetry run python utils/sync_chroma.py' to synchronize ChromaDB vectors.")
        return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest JSON domains or alter PostgreSQL Knowledge Base.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to domains directory (defaults to src/rag/data/domains).")
    parser.add_argument("--alter", type=str, default=None, help="Alter column format: table_name.column_name (e.g. doctors.email)")
    parser.add_argument("--type", type=str, default=None, help="New data type (e.g. VARCHAR(120), NUMERIC(12,2))")
    parser.add_argument("--pk", type=str, default=None, help="Set primary key flag: true or false")
    parser.add_argument("--sync", action="store_true", help="Automatically synchronize ChromaDB and restart Docker service after altering.")
    args = parser.parse_args()

    if args.alter:
        if "." not in args.alter:
            logger.error("Invalid --alter format. Use table_name.column_name (e.g. doctors.email)")
            sys.exit(1)
        tbl, col = args.alter.split(".", 1)
        pk_val = None
        if args.pk is not None:
            pk_val = args.pk.strip().lower() in ("true", "1", "yes")
        sys.exit(alter_column_in_postgres(tbl.strip(), col.strip(), data_type=args.type, is_primary_key=pk_val, sync=args.sync))
    else:
        data_dir = Path(args.data_dir) if args.data_dir else None
        sys.exit(load_domains_to_postgres(data_dir=data_dir))
