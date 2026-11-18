import hashlib
import json
import os
import unittest
import uuid
from pathlib import Path

from src.rag.config import DATABASE_URL
from src.rag.database import (
    get_db,
    init_db,
    KnowledgeBaseRepository,
    JSONKnowledgeBaseMigrator,
    compute_sha256,
    generate_stable_segment_uuid,
    build_domain_chunk,
    build_table_chunk,
    build_column_chunk,
    build_relationship_chunk,
)
from src.rag.models import (
    DomainModel,
    KnowledgeTableModel,
    KnowledgeColumnModel,
    TableDependencyModel,
    KnowledgeSegmentModel,
    SegmentVersionModel,
    EmbeddingJobModel,
    SegmentVectorModel,
)
from src.rag.service import (
    ChromaPostgresSynchronizer,
    get_embedding_synchronizer,
    retrieve_erp_schema,
    load_erp_registry,
)


class TestPostgreSQLKnowledgeBaseIntegration(unittest.TestCase):
    """
    Comprehensive test suite for PostgreSQL Primary Knowledge Base,
    hybrid chunking, incremental hashing, FAISS synchronization, and RAG retrieval.
    """

    @classmethod
    def setUpClass(cls):
        init_db()

    # =========================================================================
    # 1. Knowledge Ingestion & Idempotency
    # =========================================================================

    def test_01_all_domains_ingested(self):
        """Verify all 5 enterprise domains are stored with proper metadata."""
        with get_db() as session:
            repo = KnowledgeBaseRepository(session)
            domains = repo.list_domains()
            domain_ids = [d.domain_id for d in domains]
            expected = ["erp_ecommerce", "erp_finance", "erp_healthcare", "erp_hrms", "erp_supply_chain"]
            self.assertEqual(len(domains), 5)
            for exp in expected:
                self.assertIn(exp, domain_ids)

    def test_02_tables_and_columns_counts(self):
        """Verify tables and columns are persisted with strict foreign keys."""
        with get_db() as session:
            repo = KnowledgeBaseRepository(session)
            tables = repo.list_tables()
            self.assertGreaterEqual(len(tables), 103)
            col_count = session.query(KnowledgeColumnModel).count()
            self.assertGreaterEqual(col_count, 630)

    def test_03_table_relationships_persisted(self):
        """Verify table dependencies are stored and foreign-key links resolved."""
        with get_db() as session:
            repo = KnowledgeBaseRepository(session)
            inv_tbl = repo.get_table_by_name("customer_invoices")
            self.assertIsNotNone(inv_tbl)
            deps = repo.list_dependencies_for_table(inv_tbl.id)
            ref_ids = [d.referenced_table_id for d in deps]
            cust_tbl = repo.get_table_by_name("finance_customers")
            self.assertIn(cust_tbl.id, ref_ids)

    def test_04_idempotent_ingestion_no_duplicates(self):
        """Running migration on unchanged data must skip segments and queue 0 jobs."""
        migrator = JSONKnowledgeBaseMigrator()
        stats = migrator.run_migration()
        self.assertEqual(stats["domains_created"], 0)
        self.assertEqual(stats["tables_created"], 0)
        self.assertEqual(stats["columns_created"], 0)
        self.assertEqual(stats["segments_created"], 0)
        self.assertEqual(stats["embedding_jobs_created"], 0)
        self.assertGreaterEqual(stats["segments_skipped"], 700)

    # =========================================================================
    # 2. Incremental Updates & Version History
    # =========================================================================

    def test_05_segment_content_change_increments_version_and_preserves_uuid(self):
        """Modifying segment content must preserve UUID, increment version, and log history."""
        with get_db() as session:
            repo = KnowledgeBaseRepository(session)
            dom = repo.get_domain_by_business_id("erp_finance")
            self.assertIsNotNone(dom)

            test_uuid = generate_stable_segment_uuid("erp_finance", "Test Topic", "test_segment")
            initial_content = "Initial test content for version tracking"
            initial_hash = compute_sha256(initial_content)

            # 1. Create version 1
            seg1, op1 = repo.upsert_segment(
                segment_uuid=test_uuid,
                domain_id=dom.id,
                table_id=None,
                section="Finance Management",
                topic="Test Topic",
                segment_type="table",
                segment_name="test_segment",
                content=initial_content,
                content_hash=initial_hash
            )
            self.assertEqual(op1, "INSERT")
            self.assertEqual(seg1.version, 1)
            self.assertEqual(seg1.id, test_uuid)

            # 2. Update with modified content
            modified_content = "Modified test content with extra financial notes"
            modified_hash = compute_sha256(modified_content)
            seg2, op2 = repo.upsert_segment(
                segment_uuid=test_uuid,
                domain_id=dom.id,
                table_id=None,
                section="Finance Management",
                topic="Test Topic",
                segment_type="table",
                segment_name="test_segment",
                content=modified_content,
                content_hash=modified_hash
            )
            self.assertEqual(op2, "UPDATE")
            self.assertEqual(seg2.id, test_uuid)  # UUID strictly preserved!
            self.assertEqual(seg2.version, 2)     # Version incremented!

            # 3. Verify version history in segment_versions
            history = repo.get_segment_history(test_uuid)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].version, 1)
            self.assertEqual(history[0].content, initial_content)

            # Cleanup test segment
            session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.id == test_uuid).delete()

    def test_06_inactive_segment_excluded_from_retrieval(self):
        """Deactivated segments must be excluded from active retrieval."""
        with get_db() as session:
            repo = KnowledgeBaseRepository(session)
            dom = repo.get_domain_by_business_id("erp_finance")
            inactive_uuid = generate_stable_segment_uuid("erp_finance", "Inactive", "inactive_seg")
            seg, _ = repo.upsert_segment(
                segment_uuid=inactive_uuid,
                domain_id=dom.id,
                table_id=None,
                section="Finance",
                topic="Inactive",
                segment_type="table",
                segment_name="inactive_seg",
                content="Inactive confidential segment",
                content_hash=compute_sha256("Inactive confidential segment"),
                is_active=False
            )
            active_segs = repo.list_active_segments(domain_id=dom.id)
            active_ids = [s.id for s in active_segs]
            self.assertNotIn(inactive_uuid, active_ids)

            # Cleanup
            session.query(KnowledgeSegmentModel).filter(KnowledgeSegmentModel.id == inactive_uuid).delete()

    # =========================================================================
    # 3. Vector Database Synchronization & Persistence (ChromaDB)
    # =========================================================================

    def test_07_chroma_vector_persistence_and_retrieval(self):
        """Verify ChromaDB vector search resolves to active PostgreSQL segment UUID."""
        from src.rag.service import get_embedding_synchronizer
        syncer = get_embedding_synchronizer()
        results = syncer.search_active_segments("customer invoices billing due date", top_k=1)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("customer_invoices", top["segment_name"])
        self.assertIsNotNone(top["segment_id"])

        # Check DB model
        with get_db() as session:
            repo = KnowledgeBaseRepository(session)
            seg = repo.get_segment_by_id(uuid.UUID(top["segment_id"]))
            self.assertIsNotNone(seg)
            self.assertEqual(str(seg.id), top["segment_id"])

    # =========================================================================
    # 4. End-to-End RAG Retrieval with PostgreSQL Primary
    # =========================================================================

    def test_08_e2e_rag_retrieval_customer_invoices(self):
        """End-to-end schema retrieval grounded in PostgreSQL knowledge base."""
        res = retrieve_erp_schema("create customer_invoices table with invoice_number and total_amount")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["customer_invoices"])
        col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(col_names, ["id", "invoice_number", "total_amount"])
        self.assertIn("id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY", res.sql)
        self.assertIn("invoice_number", res.sql)
        self.assertIn("total_amount", res.sql)

    def test_09_e2e_rag_retrieval_2_tables_finance(self):
        """End-to-end domain table count prioritization grounded in PostgreSQL."""
        res = retrieve_erp_schema("create 2 tables for finance")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["chart_of_accounts", "finance_customers"])

    def test_10_e2e_anti_hallucination_boundary(self):
        """Nonexistent entities in PostgreSQL KB are rejected with 0 SQL."""
        res = retrieve_erp_schema("Search for employee_benefits table")
        self.assertEqual(res.status, "rejected")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.sql, "")
        self.assertIn("ENTITY_NOT_FOUND: employee_benefits", res.warnings)


if __name__ == "__main__":
    unittest.main()
