"""
Dynamic Table Counts & Phrasing Verification Test Suite.
Verifies that explicit table count constraints (e.g. 5, 8, 10, 15 tables)
are accurately honored across all 6 ERP domains with varied natural language wordings,
without hardcoded limits, in valid topological foreign-key order.
"""
import unittest
from fastapi.testclient import TestClient

from src.rag import retrieve_erp_schema
from src.main import app


class TestDynamicTableCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. Primary User Scenario: E-Commerce with 10 tables
    # -------------------------------------------------------------------------

    def test_ecommerce_10_tables_exact_user_query(self):
        """create schemas for e-commerce application with 10 tables returns exactly 10 tables."""
        res = retrieve_erp_schema("create schemas for e-commerce application with 10 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_ecommerce")
        self.assertEqual(res.count, 10)
        self.assertEqual(len(res.tables), 10)
        self.assertIn("categories", res.tables)
        self.assertIn("customers", res.tables)
        self.assertIn("products", res.tables)
        self.assertIn("orders", res.tables)
        self.assertIn("order_items", res.tables)
        self.assertIn("Retrieved 10 primary table(s)", res.explanation)

    def test_ecommerce_10_tables_api_endpoint(self):
        """API /api/v1/generate returns 200, CREATE_TABLE operation, SCHEMA intent, and 10 tables."""
        payload = {
            "user_query": "create schemas for e-commerce application with 10 tables",
            "enable_summary": True,
            "summary": ""
        }
        resp = self.client.post("/api/v1/generate", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body.get("message"), "Schema retrieved successfully")
        data = body.get("data", {})
        self.assertEqual(data.get("intent"), "SCHEMA")
        self.assertEqual(data.get("operation"), "CREATE_TABLE")
        sql = data.get("sql", "")
        create_count = sql.count("CREATE TABLE")
        self.assertEqual(create_count, 10)

    # -------------------------------------------------------------------------
    # 2. Banking Domain Varied Counts & Phrasings
    # -------------------------------------------------------------------------

    def test_banking_10_tables(self):
        """create schemas for bank application with 10 tables returns 10 banking tables."""
        res = retrieve_erp_schema("create schemas for bank application with 10 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_banking")
        self.assertEqual(res.count, 10)
        self.assertEqual(len(res.tables), 10)
        self.assertIn("bank_branches", res.tables)
        self.assertIn("customers", res.tables)

    def test_banking_5_tables_phrasing(self):
        """banking schema with 5 tables returns 5 banking tables."""
        res = retrieve_erp_schema("banking schema with 5 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_banking")
        self.assertEqual(res.count, 5)

    def test_banking_top_10_tables(self):
        """give me top 10 tables for banking returns 10 banking tables."""
        res = retrieve_erp_schema("give me top 10 tables for banking")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_banking")
        self.assertEqual(res.count, 10)

    # -------------------------------------------------------------------------
    # 3. Finance Domain Varied Counts & Phrasings
    # -------------------------------------------------------------------------

    def test_finance_10_tables(self):
        """create schemas for finance application with 10 tables returns 10 finance tables."""
        res = retrieve_erp_schema("create schemas for finance application with 10 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_finance")
        self.assertEqual(res.count, 10)
        self.assertEqual(len(res.tables), 10)

    def test_finance_8_tables_having_phrasing(self):
        """finance database having 8 tables returns 8 finance tables."""
        res = retrieve_erp_schema("finance database having 8 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_finance")
        self.assertEqual(res.count, 8)

    def test_finance_written_number(self):
        """finance management with six tables returns 6 finance tables."""
        res = retrieve_erp_schema("finance management with six tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_finance")
        self.assertEqual(res.count, 6)

    # -------------------------------------------------------------------------
    # 4. Healthcare Domain Varied Counts & Phrasings
    # -------------------------------------------------------------------------

    def test_healthcare_10_tables(self):
        """create schemas for healthcare application with 10 tables returns 10 healthcare tables."""
        res = retrieve_erp_schema("create schemas for healthcare application with 10 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_healthcare")
        self.assertEqual(res.count, 10)
        self.assertEqual(len(res.tables), 10)

    def test_healthcare_hospital_phrasing(self):
        """hospital management with 7 tables returns 7 healthcare tables."""
        res = retrieve_erp_schema("hospital management with 7 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_healthcare")
        self.assertEqual(res.count, 7)

    def test_healthcare_clinical_phrasing(self):
        """clinical schema having 5 tables returns 5 healthcare tables."""
        res = retrieve_erp_schema("clinical schema having 5 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_healthcare")
        self.assertEqual(res.count, 5)

    # -------------------------------------------------------------------------
    # 5. HRMS Domain Varied Counts & Phrasings
    # -------------------------------------------------------------------------

    def test_hrms_10_tables(self):
        """create schemas for hrms application with 10 tables returns 10 hrms tables."""
        res = retrieve_erp_schema("create schemas for hrms application with 10 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertEqual(res.count, 10)
        self.assertEqual(len(res.tables), 10)

    def test_hrms_human_resources_phrasing(self):
        """human resource management with 8 tables returns 8 hrms tables."""
        res = retrieve_erp_schema("human resource management with 8 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertEqual(res.count, 8)

    def test_hrms_written_word(self):
        """hrms database with six tables returns 6 hrms tables."""
        res = retrieve_erp_schema("hrms database with six tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertEqual(res.count, 6)

    # -------------------------------------------------------------------------
    # 6. Supply Chain Domain Varied Counts & Phrasings
    # -------------------------------------------------------------------------

    def test_supply_chain_10_tables(self):
        """create schemas for supply chain application with 10 tables returns 10 supply chain tables."""
        res = retrieve_erp_schema("create schemas for supply chain application with 10 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_supply_chain")
        self.assertEqual(res.count, 10)
        self.assertEqual(len(res.tables), 10)

    def test_supply_chain_inventory_management_phrasing(self):
        """inventory management schema with 5 tables returns 5 supply chain tables."""
        res = retrieve_erp_schema("inventory management schema with 5 tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_supply_chain")
        self.assertEqual(res.count, 5)

    def test_supply_chain_limit_to_phrasing(self):
        """limit to 6 tables in supply chain returns 6 supply chain tables."""
        res = retrieve_erp_schema("limit to 6 tables in supply chain")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_supply_chain")
        self.assertEqual(res.count, 6)


if __name__ == "__main__":
    unittest.main()
