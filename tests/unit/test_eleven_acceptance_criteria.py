import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.rag.service import retrieve_erp_schema, load_erp_registry


class TestElevenAcceptanceCriteria(unittest.TestCase):
    """
    Dedicated test suite validating all 11 Acceptance Criteria tests
    and 4 additional edge test scenarios specified in the task prompt.
    """

    @classmethod
    def setUpClass(cls):
        cls.registry = load_erp_registry()

    # -------------------------------------------------------------------------
    # 11 Acceptance Criteria Tests
    # -------------------------------------------------------------------------

    def test_01_single_column_isolated_scope(self):
        """Test 1: 'Retrieve only customer_id from customer_invoices'"""
        res = retrieve_erp_schema("Retrieve only customer_id from customer_invoices")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual(res.count, 1)

        # Assert sql contains valid CREATE TABLE with only requested column
        self.assertIn("CREATE TABLE customer_invoices (", res.sql)
        self.assertIn("customer_id UUID NOT NULL REFERENCES finance_customers(id) ON DELETE RESTRICT", res.sql)
        self.assertNotIn("id UUID PRIMARY KEY", res.sql)
        self.assertNotIn("invoice_number", res.sql)
        self.assertNotIn("due_date", res.sql)
        self.assertNotIn("total_amount", res.sql)

        # Structured column metadata
        self.assertEqual(len(res.columns_metadata), 1)
        col_meta = res.columns_metadata[0]
        self.assertEqual(col_meta["name"], "customer_id")
        self.assertEqual(col_meta["type"], "UUID")
        self.assertFalse(col_meta["nullable"])
        self.assertIsNotNone(col_meta["foreign_key"])
        self.assertEqual(col_meta["foreign_key"]["table"], "finance_customers")
        self.assertEqual(col_meta["foreign_key"]["column"], "id")
        self.assertEqual(col_meta["foreign_key"]["on_delete"], "RESTRICT")

        # Scope warning for referenced table omitted due to requested scope
        self.assertTrue(any("finance_customers" in w for w in res.warnings))
        self.assertIn("debug", dir(res))
        self.assertIsNotNone(res.debug)
        self.assertEqual(res.debug.get("query_plan", {}).get("expand_dependencies"), "none")

    def test_02_direct_dependency_expansion(self):
        """Test 2: 'Show hrms_attendance schema'"""
        res = retrieve_erp_schema("Show hrms_attendance schema")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        # Topological order: parent hrms_employees precedes child hrms_attendance
        self.assertEqual(res.tables, ["hrms_employees", "hrms_attendance"])
        self.assertEqual(res.count, 2)
        self.assertIn("REFERENCES hrms_employees(id) ON DELETE CASCADE", res.sql)
        self.assertNotIn("hrms_departments", res.tables)
        self.assertNotIn("hrms_designations", res.tables)

    def test_03_nonexistent_table_rejection(self):
        """Test 3: 'Search for employee_benefits'"""
        res = retrieve_erp_schema("Search for employee_benefits")
        self.assertEqual(res.status, "rejected")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertIn("ENTITY_NOT_FOUND: employee_benefits", res.warnings)
        self.assertIn("employee_benefits", res.explanation)

    def test_04_multiple_columns_no_id(self):
        """Test 4: 'Retrieve employee_id, work_date, check_in, and check_out from hrms_attendance'"""
        res = retrieve_erp_schema("Retrieve employee_id, work_date, check_in, and check_out from hrms_attendance")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, ["hrms_attendance"])
        self.assertEqual(res.count, 1)

        # Assert all 4 columns present in SQL and primary key 'id' column is omitted
        self.assertIn("employee_id UUID NOT NULL REFERENCES hrms_employees(id) ON DELETE CASCADE", res.sql)
        self.assertIn("work_date DATE NOT NULL", res.sql)
        self.assertIn("check_in TIME NOT NULL", res.sql)
        self.assertIn("check_out TIME", res.sql)
        self.assertNotIn("-- Column: id\n", res.sql)
        self.assertNotIn("    id UUID PRIMARY KEY", res.sql)

        # Structured column metadata contains exactly the 4 requested columns
        retrieved_col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(retrieved_col_names, ["employee_id", "work_date", "check_in", "check_out"])

    def test_05_multi_parent_direct_dependencies(self):
        """Test 5: 'Show inventory_stocks schema'"""
        res = retrieve_erp_schema("Show inventory_stocks schema")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, ["product_items", "warehouses", "inventory_stocks"])
        self.assertEqual(res.count, 3)
        self.assertIn("REFERENCES product_items(id) ON DELETE CASCADE", res.sql)
        self.assertIn("REFERENCES warehouses(id) ON DELETE CASCADE", res.sql)
        self.assertNotIn("suppliers", res.tables)
        self.assertNotIn("purchase_orders", res.tables)

    def test_06_hrms_employees_direct_dependencies(self):
        """Test 6: 'Retrieve hrms_employees and its actual direct dependencies'"""
        res = retrieve_erp_schema("Retrieve hrms_employees and its actual direct dependencies")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, ["hrms_departments", "hrms_designations", "hrms_employees"])
        self.assertEqual(res.count, 3)
        self.assertIn("REFERENCES hrms_departments(id) ON DELETE RESTRICT", res.sql)
        self.assertIn("REFERENCES hrms_designations(id) ON DELETE RESTRICT", res.sql)
        self.assertNotIn("hrms_attendance", res.tables)
        self.assertNotIn("hrms_payrolls", res.tables)

    def test_07_mixed_valid_and_nonexistent_columns(self):
        """Test 7: 'Retrieve employee_code, full_name, salary_grade, department_id from hrms_employees'"""
        res = retrieve_erp_schema("Retrieve employee_code, full_name, salary_grade, department_id from hrms_employees")
        self.assertEqual(res.status, "partial")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, ["hrms_employees"])
        self.assertEqual(res.count, 1)

        # Valid columns present in SQL, missing column NOT in SQL
        self.assertIn("employee_code", res.sql)
        self.assertIn("full_name", res.sql)
        self.assertIn("department_id", res.sql)
        self.assertNotIn("salary_grade", res.sql)

        # Warning contains exact formatted missing column
        self.assertIn("COLUMN_NOT_FOUND: hrms_employees.salary_grade", res.warnings)
        self.assertIn("salary_grade", res.explanation)

        # Metadata contains only the 3 valid columns
        col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(col_names, ["employee_code", "full_name", "department_id"])

    def test_08_multi_table_no_dependencies(self):
        """Test 8: 'Retrieve customer_invoices and hrms_attendance without dependencies'"""
        res = retrieve_erp_schema("Retrieve customer_invoices and hrms_attendance without dependencies")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, ["customer_invoices", "hrms_attendance"])
        self.assertEqual(res.count, 2)
        self.assertIn("Zero dependencies expanded", res.explanation)
        self.assertNotIn("finance_customers", res.tables)
        self.assertNotIn("hrms_employees", res.tables)

    def test_09_ambiguous_table_clarification(self):
        """Test 9: 'employees table. If no exact match, do NOT select hrms_employees; ask me to clarify.'"""
        res = retrieve_erp_schema("employees table. If no exact match, do NOT select hrms_employees; ask me to clarify.")
        self.assertEqual(res.status, "needs_clarification")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertTrue(res.needs_clarification)
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertEqual(res.suggestions, ["hrms_employees"])
        self.assertIn("hrms_employees", res.explanation)

    def test_10_exact_table_count_and_exclusion(self):
        """Test 10: 'Retrieve exactly two tables: finance_customers and chart_of_accounts. Do not retrieve customer_invoices.'"""
        res = retrieve_erp_schema("Retrieve exactly two tables: finance_customers and chart_of_accounts. Do not retrieve customer_invoices.")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, ["chart_of_accounts", "finance_customers"])
        self.assertEqual(res.count, 2)
        self.assertNotIn("customer_invoices", res.tables)
        self.assertIn("EXCLUDED_TABLE: customer_invoices", res.warnings)

    def test_11_natural_language_column_descriptions(self):
        """Test 11: 'Retrieve the employee reference, attendance date, check-in time, and check-out time from hrms_attendance'"""
        res = retrieve_erp_schema("Retrieve the employee reference, attendance date, check-in time, and check-out time from hrms_attendance")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.intent, "SCHEMA")
        self.assertFalse(res.needs_clarification)
        self.assertEqual(res.tables, ["hrms_attendance"])
        self.assertEqual(res.count, 1)

        # Semantic mapping mapped all 4 natural language phrases to exact columns
        retrieved_cols = [c["name"] for c in res.columns_metadata]
        self.assertEqual(retrieved_cols, ["employee_id", "work_date", "check_in", "check_out"])
        self.assertNotIn("-- Column: id\n", res.sql)
        self.assertNotIn("    id UUID PRIMARY KEY", res.sql)
        self.assertNotIn("hrms_employees", res.tables)

    # -------------------------------------------------------------------------
    # 4 Additional Edge Tests
    # -------------------------------------------------------------------------

    def test_12_forward_dependency_expansion_direct_referenced(self):
        """Extra 1: 'Retrieve hrms_employees with its directly referenced tables'"""
        res = retrieve_erp_schema("Retrieve hrms_employees with its directly referenced tables")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["hrms_departments", "hrms_designations", "hrms_employees"])
        self.assertNotIn("hrms_attendance", res.tables)
        self.assertNotIn("hrms_payrolls", res.tables)

    def test_13_reverse_dependent_exclusion(self):
        """Extra 2: 'Retrieve finance_customers' (child customer_invoices must NOT be included)"""
        res = retrieve_erp_schema("Retrieve finance_customers")
        self.assertEqual(res.status, "accepted")
        self.assertNotIn("customer_invoices", res.tables)
        for t in res.tables:
            self.assertIn(t, ["chart_of_accounts", "finance_customers"])

    def test_14_shared_column_name_specific_table(self):
        """Extra 3: 'Retrieve is_active from chart_of_accounts' (is_active exists in multiple tables)"""
        res = retrieve_erp_schema("Retrieve is_active from chart_of_accounts")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["chart_of_accounts"])
        self.assertEqual(res.count, 1)
        self.assertEqual([c["name"] for c in res.columns_metadata], ["is_active"])
        self.assertIn("is_active BOOLEAN NOT NULL DEFAULT TRUE", res.sql)

    def test_15_case_variant_and_singular_near_miss(self):
        """Extra 4: Case variant 'Retrieve HRMS_Employees' vs singular 'Retrieve employee'"""
        # Case variant matches exact table
        res_case = retrieve_erp_schema("Retrieve HRMS_Employees")
        self.assertEqual(res_case.status, "accepted")
        self.assertIn("hrms_employees", res_case.tables)

        # Singular near-miss triggers clarification
        res_near = retrieve_erp_schema("Retrieve employee")
        self.assertEqual(res_near.status, "needs_clarification")
        self.assertTrue(res_near.needs_clarification)
        self.assertEqual(res_near.suggestions, ["hrms_employees"])
        self.assertEqual(res_near.tables, [])
        self.assertEqual(res_near.sql, "")

    def test_16_leading_from_clause(self):
        """Leading 'From customer_invoices, retrieve only customer_id'"""
        res = retrieve_erp_schema("From customer_invoices, retrieve only customer_id")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual([c["name"] for c in res.columns_metadata], ["customer_id"])
        self.assertNotIn("id UUID PRIMARY KEY", res.sql)
        self.assertIn("customer_id UUID NOT NULL", res.sql)
        self.assertIn("query_plan", res.debug)
        self.assertEqual(res.debug["query_plan"]["roots"], ["customer_invoices"])
        self.assertEqual(res.debug["query_plan"]["columns"], ["customer_id"])
        self.assertFalse(res.debug["query_plan"]["include_dependencies"])

    def test_17_leading_natural_language_column_descriptions(self):
        """Leading 'From hrms_attendance, retrieve only the employee reference... Do not include the primary key.'"""
        q = "From hrms_attendance, retrieve only the employee reference, attendance date, check-in time and check-out time. Do not include the primary key."
        res = retrieve_erp_schema(q)
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["hrms_attendance"])
        self.assertEqual([c["name"] for c in res.columns_metadata], ["employee_id", "work_date", "check_in", "check_out"])
        self.assertNotIn("-- Column: id\n", res.sql)
        self.assertNotIn("    id UUID PRIMARY KEY", res.sql)

    def test_18_dot_notation_column_retrieval(self):
        """Dot-notation 'Retrieve customer_invoices.customer_id'"""
        res = retrieve_erp_schema("Retrieve customer_invoices.customer_id")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual([c["name"] for c in res.columns_metadata], ["customer_id"])

    def test_19_nonexistent_column_rejection(self):
        """Strict rejection when requested columns do not exist in table: 'Retrieve salary_grade from hrms_employees'"""
        res = retrieve_erp_schema("Retrieve salary_grade from hrms_employees")
        self.assertEqual(res.status, "rejected")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.sql, "")
        self.assertIn("COLUMN_NOT_FOUND: hrms_employees.salary_grade", res.warnings)
        self.assertIn("salary_grade", res.explanation)

    def test_20_employees_table_clarification(self):
        """Test 9 exact phrasing: 'Retrieve the employees table.'"""
        res = retrieve_erp_schema("Retrieve the employees table.")
        self.assertEqual(res.status, "needs_clarification")
        self.assertTrue(res.needs_clarification)
        self.assertEqual(res.suggestions, ["hrms_employees"])
        self.assertEqual(res.tables, [])
        self.assertEqual(res.sql, "")
        self.assertIn("No table named 'employees' exists. Did you mean 'hrms_employees'?", res.explanation)

    def test_21_api_response_structure_consistency(self):
        """Assert RAG response structure and status consistency across all execution paths"""
        test_queries = [
            ("Retrieve only customer_id from customer_invoices", "accepted"),
            ("Retrieve employee_code, full_name, salary_grade, department_id from hrms_employees", "partial"),
            ("Search for employee_benefits", "rejected"),
            ("Retrieve salary_grade from hrms_employees", "rejected"),
            ("Retrieve the employees table.", "needs_clarification"),
        ]

        expected_attributes = {
            "intent", "status", "sql", "summary", "explanation"
        }

        for query, expected_status in test_queries:
            res = retrieve_erp_schema(query)
            self.assertEqual(res.status, expected_status, f"Mismatch in status for '{query}'")
            res_dict = res.to_dict()
            for key in expected_attributes:
                self.assertIn(key, res_dict, f"Missing key '{key}' in response for '{query}'")


if __name__ == "__main__":
    unittest.main()
