import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["RAG_ONLY_MODE"] = "true"

from src.rag.service import retrieve_erp_schema, load_erp_registry
from src.main import app
from fastapi.testclient import TestClient


class TestCreationMatrix(unittest.TestCase):
    """
    Comprehensive test suite focused on schema CREATION intents across:
    1. Full domain creation (Finance, HRMS, Supply Chain)
    2. Multi-table creation with strict foreign-key topological order
    3. Isolated table creation (explicit zero dependency scope)
    4. Selective column creation (requesting specific columns only)
    5. Multi-turn summary continuity (ledger cascading across turns)
    6. Mixed entities (e.g., 2 tables present in KB + 2 not present)
    7. Strict anti-hallucination rejections (ungrounded entities or queries)
    8. Creative phrasing and synonym variations (build, setup, architect, construct)
    9. End-to-end API response contract validation (/api/v1/generate)
    """

    @classmethod
    def setUpClass(cls):
        cls.registry = load_erp_registry()
        cls.client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. Full Domain Creation Tests
    # -------------------------------------------------------------------------

    def test_01_full_domain_creation_finance(self):
        """Create entire ERP Finance database schema (21 tables >= 20 tables)."""
        res = retrieve_erp_schema("create database schema for ERP FINANCE")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertEqual(res.domain, "erp_finance")
        self.assertGreaterEqual(len(res.tables), 20)
        self.assertEqual(len(res.tables), 21)
        # Check core finance tables
        for tbl in ["chart_of_accounts", "finance_customers", "customer_invoices", "fiscal_periods", "journal_entries", "journal_lines", "vendors", "vendor_bills", "bank_accounts", "payments", "general_ledger"]:
            self.assertIn(tbl, res.tables)
            self.assertIn(f"CREATE TABLE {tbl} (", res.sql)

        # Validate topological ordering: parent before child
        coa_idx = res.sql.find("CREATE TABLE chart_of_accounts")
        fc_idx = res.sql.find("CREATE TABLE finance_customers")
        ci_idx = res.sql.find("CREATE TABLE customer_invoices")
        self.assertTrue(coa_idx < fc_idx < ci_idx)

        # Validate Explanation & Summary
        self.assertIn("Retrieved all 21 tables for domain 'Finance Management'", res.explanation)
        self.assertIn("Available Tables:", res.summary)
        self.assertIn("Status: ACTIVE", res.summary)

    def test_02_full_domain_creation_hrms(self):
        """Build full schema for HRMS domain (21 tables >= 20 tables)."""
        res = retrieve_erp_schema("build complete schema for HRMS human resources domain")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertGreaterEqual(len(res.tables), 20)
        self.assertEqual(len(res.tables), 21)
        for tbl in ["hrms_departments", "hrms_designations", "hrms_employees", "hrms_attendance", "hrms_payrolls", "hrms_leave_requests", "hrms_salary_structures", "hrms_job_openings", "hrms_candidates"]:
            self.assertIn(tbl, res.tables)
            self.assertIn(f"CREATE TABLE {tbl} (", res.sql)

    def test_03_full_domain_creation_supply_chain(self):
        """Generate complete supply chain database schema (21 tables >= 20 tables)."""
        res = retrieve_erp_schema("generate complete supply chain database schema")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_supply_chain")
        self.assertGreaterEqual(len(res.tables), 20)
        self.assertEqual(len(res.tables), 21)
        for tbl in ["product_items", "warehouses", "inventory_stocks", "suppliers", "purchase_orders", "purchase_order_items", "goods_receipts", "shipments", "bill_of_materials"]:
            self.assertIn(tbl, res.tables)
            self.assertIn(f"CREATE TABLE {tbl} (", res.sql)

    def test_03b_full_domain_creation_ecommerce(self):
        """Create complete E-Commerce database schema (21 tables >= 20 tables)."""
        res = retrieve_erp_schema("create complete database schema for ecommerce")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_ecommerce")
        self.assertGreaterEqual(len(res.tables), 20)
        self.assertEqual(len(res.tables), 21)
        # Check core e-commerce tables
        for tbl in ["categories", "customers", "addresses", "products", "orders", "order_items", "payments", "shipments", "product_reviews", "wishlists", "order_refunds", "discount_promotions", "product_tags"]:
            self.assertIn(tbl, res.tables)
            self.assertIn(f"CREATE TABLE {tbl} (", res.sql)
        # Verify topological ordering: customers precedes orders, orders precedes order_items
        c_idx = res.sql.find("CREATE TABLE customers")
        o_idx = res.sql.find("CREATE TABLE orders")
        oi_idx = res.sql.find("CREATE TABLE order_items")
        self.assertTrue(c_idx < o_idx < oi_idx)

    def test_03c_full_domain_creation_healthcare(self):
        """Build complete Healthcare database schema (21 tables >= 20 tables)."""
        res = retrieve_erp_schema("build complete database schema for healthcare domain")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.domain, "erp_healthcare")
        self.assertGreaterEqual(len(res.tables), 20)
        self.assertEqual(len(res.tables), 21)
        # Check core healthcare tables
        for tbl in ["departments", "doctors", "patients", "appointments", "medical_records", "prescriptions", "prescription_items", "lab_tests", "lab_orders", "insurance_claims", "hospital_rooms", "patient_admissions", "vital_signs", "patient_allergies", "patient_bills", "medical_surgeries", "operating_theaters"]:
            self.assertIn(tbl, res.tables)
            self.assertIn(f"CREATE TABLE {tbl} (", res.sql)
        # Verify topological ordering: departments precedes doctors, doctors precedes appointments
        d_idx = res.sql.find("CREATE TABLE departments")
        doc_idx = res.sql.find("CREATE TABLE doctors")
        apt_idx = res.sql.find("CREATE TABLE appointments")
        self.assertTrue(d_idx < doc_idx < apt_idx)

    # -------------------------------------------------------------------------
    # 2. Multi-Table Creation & Topological Sorting
    # -------------------------------------------------------------------------

    def test_04_multi_table_creation_topological(self):
        """Construct customer_invoices and finance_customers schema."""
        res = retrieve_erp_schema("construct customer_invoices and finance_customers schema")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["finance_customers", "customer_invoices"])

        # finance_customers MUST precede customer_invoices in SQL
        fc_pos = res.sql.index("CREATE TABLE finance_customers")
        ci_pos = res.sql.index("CREATE TABLE customer_invoices")
        self.assertLess(fc_pos, ci_pos)

        # Foreign key validation
        self.assertIn("REFERENCES finance_customers(id)", res.sql)
        self.assertIn("Available References:", res.summary)
        self.assertIn("FOREIGN KEY (customer_id) REFERENCES finance_customers(id)", res.summary)

    def test_04b_ecommerce_multi_table_topological(self):
        """Construct orders, products, and order_items from ecommerce."""
        res = retrieve_erp_schema("construct orders, products, and order_items from ecommerce")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(len(res.tables), 3)
        self.assertIn("products", res.tables)
        self.assertIn("orders", res.tables)
        self.assertIn("order_items", res.tables)
        p_pos = res.sql.index("CREATE TABLE products")
        o_pos = res.sql.index("CREATE TABLE orders")
        oi_pos = res.sql.index("CREATE TABLE order_items")
        self.assertTrue(p_pos < oi_pos and o_pos < oi_pos)

    def test_04c_healthcare_multi_table_topological(self):
        """Construct patients, doctors, and appointments from healthcare."""
        res = retrieve_erp_schema("construct patients, doctors, and appointments from healthcare")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(len(res.tables), 3)
        self.assertIn("patients", res.tables)
        self.assertIn("doctors", res.tables)
        self.assertIn("appointments", res.tables)
        pat_pos = res.sql.index("CREATE TABLE patients")
        doc_pos = res.sql.index("CREATE TABLE doctors")
        apt_pos = res.sql.index("CREATE TABLE appointments")
        self.assertTrue(pat_pos < apt_pos and doc_pos < apt_pos)

    # -------------------------------------------------------------------------
    # 3. Isolated & Selective Scope Creation
    # -------------------------------------------------------------------------

    def test_05_isolated_table_creation(self):
        """Create only customer_invoices table without dependencies."""
        res = retrieve_erp_schema("create only customer_invoices table without dependencies")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual(len(res.tables), 1)
        self.assertIn("CREATE TABLE customer_invoices (", res.sql)
        self.assertNotIn("CREATE TABLE finance_customers", res.sql)
        # Warning for omitted parent table
        self.assertTrue(any("finance_customers" in w for w in res.warnings))

    def test_06_selective_column_creation(self):
        """Create chart_of_accounts table with id and account_name columns."""
        res = retrieve_erp_schema("create chart_of_accounts table with id and account_name columns")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["chart_of_accounts"])
        self.assertIn("CREATE TABLE chart_of_accounts (", res.sql)
        self.assertIn("id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,", res.sql)
        self.assertIn("account_name VARCHAR(150) NOT NULL", res.sql)
        self.assertNotIn("account_code", res.sql)
        self.assertNotIn("account_type", res.sql)
        self.assertNotIn("is_active", res.sql)
        self.assertEqual(len(res.columns_metadata), 2)

    # -------------------------------------------------------------------------
    # 4. Multi-Turn Summary Continuity (Ledger Cascading)
    # -------------------------------------------------------------------------

    def test_07_multi_turn_summary_cascade(self):
        """
        Verify multi-turn summary ledger continuity across 3 sequential creation turns:
        Turn 1: Create chart_of_accounts
        Turn 2: Create finance_customers (references active chart_of_accounts) -> 0 warnings
        Turn 3: Create customer_invoices (references active finance_customers) -> 0 warnings
        """
        # Turn 1
        t1 = retrieve_erp_schema("create chart_of_accounts table")
        self.assertEqual(t1.status, "accepted")
        self.assertIn("chart_of_accounts", t1.tables)
        self.assertIn("chart_of_accounts: id, account_code, account_name, account_type, is_active; Status: ACTIVE", t1.summary)

        # Turn 2: Pass Turn 1 summary
        t2 = retrieve_erp_schema("create finance_customers table", summary=t1.summary)
        self.assertEqual(t2.status, "accepted")
        # No warning about missing chart_of_accounts because it is already ACTIVE in summary
        self.assertListEqual(t2.warnings, [])
        self.assertIn("chart_of_accounts", t2.summary)
        self.assertIn("finance_customers", t2.summary)
        self.assertIn("finance_customers: FOREIGN KEY (gl_account_id) REFERENCES chart_of_accounts(id)", t2.summary)
        self.assertIn("Foreign key relationships wire directly to active table(s) in summary ledger: chart_of_accounts.", t2.explanation)

        # Turn 3: Pass Turn 2 summary
        t3 = retrieve_erp_schema("create customer_invoices table", summary=t2.summary)
        self.assertEqual(t3.status, "accepted")
        self.assertListEqual(t3.warnings, [])
        self.assertIn("chart_of_accounts", t3.summary)
        self.assertIn("finance_customers", t3.summary)
        self.assertIn("customer_invoices", t3.summary)
        self.assertIn("customer_invoices: FOREIGN KEY (customer_id) REFERENCES finance_customers(id)", t3.summary)

    # -------------------------------------------------------------------------
    # 5. Mixed Entities (2 in KB + 2 Not in KB)
    # -------------------------------------------------------------------------

    def test_08_mixed_tables_partial_creation(self):
        """
        Query with 2 valid tables and 2 ungrounded tables:
        'create customer_invoices, finance_customers, user_tokens, and analytics_events'
        Must return status 'partial', generate DDL for valid tables, and flag ungrounded tables.
        """
        res = retrieve_erp_schema("create customer_invoices, finance_customers, user_tokens, and analytics_events")
        self.assertEqual(res.status, "partial")
        self.assertListEqual(res.tables, ["finance_customers", "customer_invoices"])

        # Generated DDL has valid tables only
        self.assertIn("CREATE TABLE finance_customers (", res.sql)
        self.assertIn("CREATE TABLE customer_invoices (", res.sql)
        self.assertNotIn("user_tokens", res.sql)
        self.assertNotIn("analytics_events", res.sql)

        # Explicit warnings and explanation note
        self.assertTrue(any("TABLE_NOT_FOUND: user_tokens" in w for w in res.warnings))
        self.assertTrue(any("TABLE_NOT_FOUND: analytics_events" in w for w in res.warnings))
        self.assertIn("user_tokens, analytics_events", res.explanation)

        # Summary ledger contains only valid active tables
        self.assertIn("finance_customers:", res.summary)
        self.assertIn("customer_invoices:", res.summary)
        self.assertNotIn("user_tokens:", res.summary)

    # -------------------------------------------------------------------------
    # 6. Strict Anti-Hallucination Rejection
    # -------------------------------------------------------------------------

    def test_09_invalid_table_rejection_zero_sql(self):
        """Create a account schema for ERP FINANCE (invalid candidate rejected with 0 SQL)."""
        res = retrieve_erp_schema("create a account schema for ERP FINANCE")
        self.assertEqual(res.status, "rejected")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.sql, "")
        self.assertIn("does not exist in the enterprise schema knowledge base", res.explanation)

    def test_10_ungrounded_external_domain_rejection(self):
        """Create schema for cryptocurrency web3 blockchain smart contracts."""
        res = retrieve_erp_schema("architect cryptocurrency web3 blockchain smart contracts")
        self.assertIn(res.status, ["rejected", "not_found"])
        self.assertEqual(res.tables, [])
        self.assertEqual(res.sql, "")

    # -------------------------------------------------------------------------
    # 7. Phrasing Variations & Verbs
    # -------------------------------------------------------------------------

    def test_11_creation_verb_variations(self):
        """Test informal creation phrasing: setup, architect, construct."""
        res_setup = retrieve_erp_schema("setup hrms_attendance with hrms_employees")
        self.assertEqual(res_setup.status, "accepted")
        self.assertEqual(res_setup.tables, ["hrms_employees", "hrms_attendance"])

        res_arch = retrieve_erp_schema("architect inventory_stock and inventory_items schema")
        self.assertEqual(res_arch.status, "accepted")
        self.assertEqual(res_arch.tables, ["product_items", "inventory_stocks"])

    # -------------------------------------------------------------------------
    # 8. End-to-End API Response Contract (/api/v1/generate)
    # -------------------------------------------------------------------------

    def test_12_end_to_end_api_generation_5_keys(self):
        """
        Verify the FastAPI /api/v1/generate endpoint returns the exact 5 clean keys
        in data: intent, operation, sql, updated_summary, explanation.
        """
        payload = {
            "prompt": "create database schema for ERP FINANCE",
            "enable_summary": True
        }
        response = self.client.post("/api/v1/generate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["message"], "Schema retrieved successfully")
        self.assertEqual(data["status_code"], 200)

        data_obj = data["data"]
        expected_keys = {"intent", "operation", "sql", "updated_summary", "explanation"}
        self.assertEqual(set(data_obj.keys()), expected_keys)
        self.assertEqual(data_obj["intent"], "SCHEMA")
        self.assertEqual(data_obj["operation"], "CREATE_TABLE")
        self.assertIn("CREATE TABLE chart_of_accounts", data_obj["sql"])
        self.assertIn("Available Tables:", data_obj["updated_summary"])
        self.assertIn("Retrieved all 21 tables for domain 'Finance Management'", data_obj["explanation"])


if __name__ == "__main__":
    unittest.main()
