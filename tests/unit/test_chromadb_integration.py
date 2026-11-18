import unittest
import uuid
import numpy as np

from src.rag.service import get_chroma_engine, ChromaSchemaEngine, get_embedding_synchronizer, sync_chroma_from_postgres
from src.rag.config import CHROMA_DATA_DIR, CHROMA_COLLECTION_NAME


class TestChromaDBIntegration(unittest.TestCase):
    """
    Integration test suite verifying ChromaDB vector database functionality,
    structured identity keys (domain_id:table_id:column_id), native metadata
    filtering, and zero-ghost-chunk in-place updates.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = get_chroma_engine()
        # Ensure ChromaDB is synchronized
        if cls.engine.count() < 700:
            sync_chroma_from_postgres()

    def test_01_chroma_collection_populated(self):
        """Verify ChromaDB has all active segments indexed."""
        count = self.engine.count()
        self.assertGreaterEqual(count, 700, f"Expected at least 700 segments in ChromaDB, found {count}")

    def test_02_semantic_search_with_metadata(self):
        """Verify semantic query returns documents and rich metadata."""
        results = self.engine.search("doctor patient appointment", top_k=3)
        self.assertGreater(len(results), 0)
        first = results[0]
        self.assertIn("similarity", first)
        self.assertGreater(first["similarity"], 0.3)
        self.assertIn("identity_key", first["metadata"])
        self.assertIn("domain_code", first["metadata"])

    def test_03_native_metadata_filtering_by_domain(self):
        """Verify native WHERE filtering by domain_code."""
        results = self.engine.search(
            "invoices and payments",
            top_k=5,
            where={"domain_code": "erp_finance"}
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["metadata"].get("domain_code"), "erp_finance")

    def test_04_native_metadata_filtering_by_table(self):
        """Verify native WHERE filtering by table_name."""
        results = self.engine.search(
            "employee salary and net pay",
            top_k=5,
            where={"table_name": "hrms_employees"}
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["metadata"].get("table_name"), "hrms_employees")

    def test_05_in_place_update_zero_ghost_chunks(self):
        """
        Verify that upserting an existing ID updates the vector and document in place
        without creating duplicate ghost chunks.
        """
        test_id = str(uuid.uuid4())
        initial_doc = "Table: test_table | Column: original_name | Type: VARCHAR(50)"
        initial_meta = {
            "domain_code": "erp_test",
            "table_name": "test_table",
            "column_name": "original_name",
            "identity_key": "erp_test:test_table:original_name"
        }

        # Step 1: Add new chunk
        initial_count = self.engine.count()
        self.engine.upsert_segment(
            segment_id=test_id,
            document=initial_doc,
            metadata=initial_meta
        )
        self.assertEqual(self.engine.count(), initial_count + 1)

        # Step 2: In-place rename simulation (same ID, new content & metadata)
        renamed_doc = "Table: test_table | Column: renamed_user_name | Type: VARCHAR(50)"
        renamed_meta = {
            "domain_code": "erp_test",
            "table_name": "test_table",
            "column_name": "renamed_user_name",
            "identity_key": "erp_test:test_table:renamed_user_name"
        }
        self.engine.upsert_segment(
            segment_id=test_id,
            document=renamed_doc,
            metadata=renamed_meta
        )

        # Count MUST NOT increase (Zero ghost chunks!)
        self.assertEqual(self.engine.count(), initial_count + 1)

        # Verify the fetched chunk reflects the updated content
        fetched = self.engine.get_segment(test_id)
        self.assertIsNotNone(fetched)
        self.assertIn("renamed_user_name", fetched["document"])
        self.assertEqual(fetched["metadata"]["identity_key"], "erp_test:test_table:renamed_user_name")

        # Clean up test segment
        self.engine.delete_segment(test_id)
        self.assertEqual(self.engine.count(), initial_count)

    def test_06_hybrid_search_active_segments_resolver(self):
        """
        Verify search_active_segments in embedding_sync resolves vectors
        from ChromaDB with full PostgreSQL segment integrity.
        """
        syncer = get_embedding_synchronizer()
        results = syncer.search_active_segments(
            query="patient appointments and prescriptions",
            top_k=3,
            domain_id="erp_healthcare"
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["section"], "Healthcare Management")
            self.assertIn("content", r)
            self.assertIn("segment_id", r)


if __name__ == "__main__":
    unittest.main()
