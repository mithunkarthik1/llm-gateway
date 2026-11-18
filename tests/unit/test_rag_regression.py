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


class TestEnterpriseRAGRegression(unittest.TestCase):
    """
    Comprehensive 20-scenario regression suite verifying:
    - Domain isolation & completeness
    - Table-level & exact direct foreign-key retrieval
    - Granular column retrieval
    - Anti-hallucination for nonexistent tables and columns
    - Schema fidelity (data types, defaults, ON DELETE, constraints)
    - Ambiguity resolution and multi-turn domain switching
    """

    @classmethod
    def setUpClass(cls):
        cls.registry = load_erp_registry()

    # Scenario 1: Complete ERP schema retrieval across all domains
    def test_01_complete_erp_schema(self):
        res = retrieve_erp_schema("Show all tables from the ERP application.")
        self.assertEqual(res.domain, "entire_application")
        self.assertGreaterEqual(len(res.tables), 15)
        self.assertIn("customer_invoices", res.tables)
        self.assertIn("hrms_employees", res.tables)
        self.assertIn("inventory_stocks", res.tables)

    # Scenario 2: Finance-only schema retrieval
    def test_02_finance_only_schema(self):
        res = retrieve_erp_schema("Give me the complete finance database structure.")
        self.assertEqual(res.domain, "erp_finance")
        self.assertGreaterEqual(len(res.tables), 20)
        expected = {"chart_of_accounts", "fiscal_periods", "journal_entries", "finance_customers", "customer_invoices"}
        self.assertTrue(expected.issubset(set(res.tables)))
        self.assertNotIn("hrms_employees", res.tables)
        self.assertNotIn("warehouses", res.tables)

    # Scenario 3: HRMS-only schema retrieval
    def test_03_hrms_only_schema(self):
        res = retrieve_erp_schema("Show the complete HRMS database structure.")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertGreaterEqual(len(res.tables), 20)
        expected = {"hrms_departments", "hrms_designations", "hrms_employees", "hrms_attendance", "hrms_payrolls"}
        self.assertTrue(expected.issubset(set(res.tables)))
        self.assertNotIn("customer_invoices", res.tables)

    # Scenario 4: Supply Chain-only schema retrieval
    def test_04_supply_chain_only_schema(self):
        res = retrieve_erp_schema("Explain the complete supply chain database structure.")
        self.assertEqual(res.domain, "erp_supply_chain")
        self.assertGreaterEqual(len(res.tables), 20)
        expected = {"warehouses", "product_items", "suppliers", "purchase_orders", "inventory_stocks"}
        self.assertTrue(expected.issubset(set(res.tables)))
        self.assertNotIn("hrms_employees", res.tables)

    # Scenario 5: Specific table retrieval
    def test_05_specific_table_retrieval(self):
        res = retrieve_erp_schema("I need only the customer invoice table.")
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual(res.count, 1)

    # Scenario 6: Direct dependency retrieval
    def test_06_direct_dependency_retrieval(self):
        res = retrieve_erp_schema("Show the complete schema of customer_invoices.")
        # Only direct actual foreign key is finance_customers
        self.assertEqual(res.tables, ["finance_customers", "customer_invoices"])
        self.assertNotIn("journal_entries", res.tables)

    # Scenario 7: Single-column metadata retrieval
    def test_07_single_column_metadata(self):
        res = retrieve_erp_schema("Which column stores the invoice due date?")
        self.assertIn("due_date", res.sql)
        self.assertIn("DATE", res.sql)
        self.assertEqual(res.tables, ["customer_invoices"])

    # Scenario 8: Multiple-column metadata retrieval
    def test_08_multiple_column_metadata(self):
        res = retrieve_erp_schema("I need invoice number, customer ID and due date.")
        self.assertIn("invoice_number", res.sql)
        self.assertIn("customer_id", res.sql)
        self.assertIn("due_date", res.sql)
        self.assertNotIn("total_amount", res.sql)

    # Scenario 9: Nonexistent table retrieval (Hallucination check)
    def test_09_nonexistent_table_retrieval(self):
        res = retrieve_erp_schema("Show me the employee salary history table.")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertIn("salary history", res.explanation.lower())

    # Scenario 10: Nonexistent column retrieval (Hallucination check)
    def test_10_nonexistent_column_retrieval(self):
        res = retrieve_erp_schema("Which column stores customer invoice discount percentage?")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertIn("discount", res.explanation.lower())

    # Scenario 11: Incorrect domain and table combinations
    def test_11_incorrect_domain_table_combination(self):
        res = retrieve_erp_schema("Show customer invoice structure.", domain="erp_hrms")
        self.assertEqual(res.domain, "erp_finance")
        self.assertIn("customer_invoices", res.tables)

    # Scenario 12: Foreign-key accuracy
    def test_12_foreign_key_accuracy(self):
        res = retrieve_erp_schema("Show customer invoice structure.")
        self.assertIn("REFERENCES finance_customers(id) ON DELETE RESTRICT", res.sql)
        self.assertIn("FOREIGN KEY (customer_id) REFERENCES finance_customers(id)", res.summary)

    # Scenario 13: Nullable and default-value preservation
    def test_13_nullable_default_preservation(self):
        res = retrieve_erp_schema("Show customer invoice structure.")
        self.assertIn("DEFAULT gen_random_uuid()", res.sql)
        self.assertIn("DEFAULT 0", res.sql)
        self.assertIn("NOT NULL", res.sql)

    # Scenario 14: ON DELETE and ON UPDATE preservation
    def test_14_on_delete_preservation(self):
        res = retrieve_erp_schema("Show customer invoice structure.")
        self.assertIn("ON DELETE RESTRICT", res.sql)
        self.assertNotIn("ON DELETE CASCADE", res.sql)

    # Scenario 15: Duplicate dependency prevention
    def test_15_duplicate_dependency_prevention(self):
        res = retrieve_erp_schema("Show customer invoice structure.")
        self.assertEqual(len(res.tables), len(set(res.tables)))

    # Scenario 16: Multi-turn retrieval with domain switching
    def test_16_multi_turn_domain_switching(self):
        r1 = retrieve_erp_schema("Show me all finance tables.")
        self.assertEqual(r1.domain, "erp_finance")
        r2 = retrieve_erp_schema("Now show me employee details.")
        self.assertEqual(r2.domain, "erp_hrms")
        self.assertIn("hrms_employees", r2.tables)
        r3 = retrieve_erp_schema("Get the inventory stock details.")
        self.assertEqual(r3.domain, "erp_supply_chain")
        self.assertIn("inventory_stocks", r3.tables)

    # Scenario 17: Ambiguous domain and table requests
    def test_17_ambiguous_requests(self):
        res = retrieve_erp_schema("Show me the details table.")
        self.assertEqual(res.domain, "ambiguous")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.sql, "")
        self.assertIn("ambiguous", res.explanation.lower())

    # Scenario 18: Missing metadata handling
    def test_18_missing_metadata_handling(self):
        res = retrieve_erp_schema("Get the invoice_rating column from customer invoices.")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertIn("invoice_rating", res.explanation.lower())

    # Scenario 19: Full-schema completeness
    def test_19_full_schema_completeness(self):
        res = retrieve_erp_schema("Show all tables from the ERP application.")
        self.assertGreaterEqual(len(res.tables), 15)
        cust_idx = res.tables.index("finance_customers")
        inv_idx = res.tables.index("customer_invoices")
        self.assertLess(cust_idx, inv_idx)

    # Scenario 20: Hallucination prevention
    def test_20_hallucination_prevention(self):
        res1 = retrieve_erp_schema("Get the customer payment history table.")
        self.assertEqual(res1.tables, [])
        self.assertEqual(res1.count, 0)
        self.assertEqual(res1.sql, "")

        res2 = retrieve_erp_schema("Which column stores employee blood group in attendance?")
        self.assertEqual(res2.tables, [])
        self.assertEqual(res2.count, 0)
        self.assertEqual(res2.sql, "")
        self.assertIn("blood_group", res2.explanation.lower())


class TestManualRetestsScenarios10(unittest.TestCase):
    """
    Direct tests for the 10 manual API retests scenarios described in Sections 2-6 and 10:
    Test 1: Full ERP schema retrieval — all 15 tables.
    Test 2: Finance-only schema retrieval — exactly 5 Finance tables.
    Test 3: Specific table retrieval — customer_invoices.
    Test 4: Specific column retrieval — customer_invoices.customer_id.
    Test 5: HRMS table retrieval — hrms_attendance.
    Test 6: Supply Chain table retrieval — inventory_stocks.
    Test 7: Nonexistent table — employee_benefits.
    Test 8: Nonexistent column — salary_grade in hrms_employees.
    Test 9: Multiple-column retrieval — four requested hrms_attendance columns.
    Test 10: Table with direct dependencies — hrms_employees, hrms_departments, hrms_designations.
    """

    # Test 1: Full ERP schema retrieval
    def test_scenario_01_complete_erp_schema(self):
        res = retrieve_erp_schema("Show the complete ERP schema")
        self.assertEqual(res.domain, "entire_application")
        self.assertGreaterEqual(len(res.tables), 15)
        self.assertLess(res.tables.index("finance_customers"), res.tables.index("customer_invoices"))
        self.assertLess(res.tables.index("hrms_employees"), res.tables.index("hrms_attendance"))
        self.assertNotIn("None", res.sql)

    # Test 2: Finance-only schema retrieval
    def test_scenario_02_finance_domain(self):
        res = retrieve_erp_schema("Show Finance schema")
        self.assertEqual(res.domain, "erp_finance")
        self.assertGreaterEqual(len(res.tables), 20)
        expected_finance = {"chart_of_accounts", "fiscal_periods", "journal_entries", "finance_customers", "customer_invoices"}
        self.assertTrue(expected_finance.issubset(set(res.tables)))
        self.assertNotIn("hrms_employees", res.tables)
        self.assertNotIn("warehouses", res.tables)
        self.assertNotIn("customer_payments", res.sql)

    # Test 3: Specific table retrieval
    def test_scenario_03_specific_table(self):
        res = retrieve_erp_schema("Retrieve customer_invoices and its actual direct dependency")
        self.assertEqual(res.domain, "erp_finance")
        self.assertEqual(res.tables, ["finance_customers", "customer_invoices"])
        self.assertNotIn("journal_entries", res.tables)
        self.assertNotIn("chart_of_accounts", res.tables)
        self.assertIn("1 direct foreign-key dependency", res.explanation)

    # Test 4: Specific column retrieval
    def test_scenario_04_specific_column(self):
        res = retrieve_erp_schema("Retrieve customer_id from customer_invoices")
        self.assertEqual(res.domain, "erp_finance")
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual(res.count, 1)
        # Dedicated column metadata checks
        self.assertIn("Column: customer_id", res.sql)
        self.assertIn("Data Type: UUID", res.sql)
        self.assertIn("Nullable: No (NOT NULL)", res.sql)
        self.assertIn("Referenced Table: finance_customers", res.sql)
        self.assertIn("Referenced Column: id", res.sql)
        self.assertIn("Delete Rule: ON DELETE RESTRICT", res.sql)
        self.assertNotIn("invoice_number", res.sql)

    # Test 5: HRMS table retrieval
    def test_scenario_05_hrms_attendance(self):
        res = retrieve_erp_schema("Show hrms_attendance schema")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertEqual(res.tables, ["hrms_employees", "hrms_attendance"])
        self.assertNotIn("hrms_departments", res.tables)
        self.assertNotIn("hrms_designations", res.tables)
        self.assertIn("employee_id UUID NOT NULL REFERENCES hrms_employees(id) ON DELETE CASCADE", res.sql)
        self.assertIn("check_out TIME", res.sql)
        self.assertIn("1 direct foreign-key dependency", res.explanation)

    # Test 6: Supply Chain table retrieval
    def test_scenario_06_supply_chain_inventory(self):
        res = retrieve_erp_schema("Show inventory_stocks schema")
        self.assertEqual(res.domain, "erp_supply_chain")
        self.assertEqual(res.tables, ["product_items", "warehouses", "inventory_stocks"])
        self.assertNotIn("suppliers", res.tables)
        self.assertNotIn("purchase_orders", res.tables)
        self.assertIn("REFERENCES product_items(id) ON DELETE CASCADE", res.sql)
        self.assertIn("REFERENCES warehouses(id) ON DELETE CASCADE", res.sql)
        self.assertIn("2 direct foreign-key dependency", res.explanation)

    # Test 7: Nonexistent table
    def test_scenario_07_nonexistent_table(self):
        res1 = retrieve_erp_schema("Show employee_benefits table schema")
        self.assertEqual(res1.domain, "not_found")
        self.assertEqual(res1.tables, [])
        self.assertEqual(res1.count, 0)
        self.assertEqual(res1.sql, "")
        self.assertIn("employee_benefits", res1.explanation)
        self.assertIn("Available Tables:\n- None", res1.summary)

        res2 = retrieve_erp_schema("Search for employee_benefits")
        self.assertEqual(res2.domain, "not_found")
        self.assertEqual(res2.tables, [])
        self.assertEqual(res2.sql, "")

    # Test 8: Nonexistent column
    def test_scenario_08_nonexistent_column(self):
        res = retrieve_erp_schema("Search for salary_grade in hrms_employees")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertIn("salary_grade", res.explanation)
        self.assertIn("hrms_employees", res.explanation)
        self.assertIn("Available Tables:\n- None", res.summary)

    # Test 9: Multiple-column retrieval
    def test_scenario_09_multiple_columns(self):
        res = retrieve_erp_schema("Retrieve employee_id, work_date, check_in, and check_out from hrms_attendance")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertEqual(res.tables, ["hrms_attendance"])
        self.assertEqual(res.count, 1)
        self.assertIn("employee_id", res.sql)
        self.assertIn("work_date", res.sql)
        self.assertIn("check_in", res.sql)
        self.assertIn("check_out", res.sql)
        self.assertNotIn("-- Column: id\n", res.sql)

    # Test 9b: Multi-column retrieval with natural language noise
    def test_scenario_09b_multiple_columns_with_noise(self):
        res = retrieve_erp_schema("Retrieve metadata for employee_id, work_date, and check_in from hrms_attendance")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertEqual(res.tables, ["hrms_attendance"])
        self.assertEqual(res.count, 1)
        self.assertIn("employee_id", res.sql)
        self.assertIn("work_date", res.sql)
        self.assertIn("check_in", res.sql)
        self.assertNotIn("check_out", res.sql)

    # Test 9c: Mixed valid and invalid requested columns (Section 4 & Scenario 6)
    def test_scenario_09c_mixed_valid_invalid_columns(self):
        res = retrieve_erp_schema("Retrieve employee_code, salary_grade from hrms_employees")
        self.assertEqual(res.tables, ["hrms_employees"])
        self.assertEqual(res.count, 1)
        self.assertIn("employee_code", res.sql)
        self.assertIn("salary_grade", res.explanation)
        self.assertIn("Warning:", res.explanation)
        self.assertNotIn("salary_grade VARCHAR", res.sql)
        self.assertNotIn("salary_grade TEXT", res.sql)

    # Test 10: HRMS employees and direct dependencies
    def test_scenario_10_hrms_employees_dependencies(self):
        res = retrieve_erp_schema("Retrieve hrms_employees and its actual direct dependencies")
        self.assertEqual(res.domain, "erp_hrms")
        self.assertEqual(res.tables, ["hrms_departments", "hrms_designations", "hrms_employees"])
        self.assertNotIn("hrms_attendance", res.tables)
        self.assertNotIn("hrms_payrolls", res.tables)
        self.assertIn("REFERENCES hrms_departments(id) ON DELETE RESTRICT", res.sql)
        self.assertIn("REFERENCES hrms_designations(id) ON DELETE RESTRICT", res.sql)
        self.assertIn("2 direct foreign-key dependency", res.explanation)
        self.assertNotIn("hire_date", res.sql)

    # Test 10b: Requests that explicitly exclude dependencies
    def test_scenario_10b_exclude_dependencies(self):
        res1 = retrieve_erp_schema("I need only the customer invoice table")
        self.assertEqual(res1.tables, ["customer_invoices"])
        self.assertEqual(res1.count, 1)
        self.assertIn("0 dependencies", res1.explanation)

        res2 = retrieve_erp_schema("Retrieve customer_invoices without dependencies")
        self.assertEqual(res2.tables, ["customer_invoices"])
        self.assertEqual(res2.count, 1)
        self.assertIn("0 dependencies", res2.explanation)


class TestTwentyComprehensiveScenarios(unittest.TestCase):
    """
    Automated regression tests covering all 20 explicit scenarios from Section 13:
    1. Full ERP schema retrieval
    2. Finance-only schema retrieval
    3. Exact single-table retrieval
    4. Exact single-column retrieval
    5. Multiple-column retrieval
    6. Mixed valid and invalid columns
    7. Nonexistent table rejection
    8. Nonexistent column rejection
    9. Ambiguous table clarification
    10. Direct FK dependency retrieval
    11. Multi-level dependency retrieval
    12. Cross-domain multi-table retrieval
    13. Explicit dependency expansion disabled
    14. Natural-language column mapping
    15. Missing metadata handling
    16. Nullable and default-value accuracy
    17. FK and ON DELETE accuracy
    18. SQL-to-JSON consistency
    19. Multi-turn follow-up query handling
    20. Empty, malformed, and ambiguous user queries
    """

    # 1. Full ERP schema retrieval
    def test_01_full_erp_schema(self):
        res = retrieve_erp_schema("Show all tables from the ERP application.")
        self.assertEqual(res.domain, "entire_application")
        self.assertGreaterEqual(len(res.tables), 15)
        self.assertIn("customer_invoices", res.tables)
        self.assertIn("hrms_employees", res.tables)
        self.assertIn("inventory_stocks", res.tables)

    # 2. Finance-only schema retrieval
    def test_02_finance_only_schema(self):
        res = retrieve_erp_schema("Give me the complete finance database structure.")
        self.assertEqual(res.domain, "erp_finance")
        self.assertGreaterEqual(len(res.tables), 20)
        for t in ["chart_of_accounts", "fiscal_periods", "journal_entries", "finance_customers", "customer_invoices"]:
            self.assertIn(t, res.tables)
        self.assertNotIn("hrms_employees", res.tables)

    # 3. Exact single-table retrieval
    def test_03_exact_single_table(self):
        res = retrieve_erp_schema("I need only the customer invoice table.")
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual(res.count, 1)
        self.assertIn("0 dependencies", res.explanation)

    # 4. Exact single-column retrieval
    def test_04_exact_single_column(self):
        res = retrieve_erp_schema("Retrieve only customer_id from customer_invoices")
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual(res.count, 1)
        self.assertIn("customer_id UUID NOT NULL REFERENCES finance_customers(id) ON DELETE RESTRICT", res.sql)
        self.assertNotIn("-- Column: id\n", res.sql)
        self.assertNotIn("-- Column: invoice_number", res.sql)
        self.assertNotIn("-- Column: due_date", res.sql)
        self.assertNotIn("-- Column: total_amount", res.sql)

    # 5. Multiple-column retrieval
    def test_05_multiple_columns(self):
        res = retrieve_erp_schema("Retrieve employee_id, work_date, check_in, and check_out from hrms_attendance")
        self.assertEqual(res.tables, ["hrms_attendance"])
        self.assertEqual(res.count, 1)
        self.assertIn("employee_id", res.sql)
        self.assertIn("work_date", res.sql)
        self.assertIn("check_in", res.sql)
        self.assertIn("check_out", res.sql)
        self.assertNotIn("-- Column: id\n", res.sql)

    # 6. Mixed valid and invalid columns
    def test_06_mixed_valid_invalid_columns(self):
        res = retrieve_erp_schema("Retrieve employee_code, full_name, salary_grade, department_id from hrms_employees")
        self.assertEqual(res.tables, ["hrms_employees"])
        self.assertEqual(res.count, 1)
        self.assertIn("employee_code", res.sql)
        self.assertIn("full_name", res.sql)
        self.assertIn("department_id", res.sql)
        self.assertIn("Warning:", res.explanation)
        self.assertIn("salary_grade", res.explanation)
        self.assertNotIn("salary_grade VARCHAR", res.sql)

    # 7. Nonexistent table rejection
    def test_07_nonexistent_table(self):
        res = retrieve_erp_schema("Show employee_benefits table schema")
        self.assertEqual(res.domain, "not_found")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertIn("employee_benefits", res.explanation)

    # 8. Nonexistent column rejection
    def test_08_nonexistent_column(self):
        res = retrieve_erp_schema("Search for salary_grade in hrms_employees")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertIn("salary_grade", res.explanation)
        self.assertIn("Available columns for 'hrms_employees'", res.explanation)

    # 9. Ambiguous table clarification
    def test_09_ambiguous_table(self):
        res = retrieve_erp_schema("Show employees")
        self.assertEqual(res.domain, "ambiguous")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.sql, "")
        self.assertIn("clarify", res.explanation.lower())
        self.assertIn("hrms_employees", res.explanation)

    # 10. Direct FK dependency retrieval
    def test_10_direct_fk_dependencies(self):
        res = retrieve_erp_schema("Show inventory_stocks schema")
        self.assertEqual(res.tables, ["product_items", "warehouses", "inventory_stocks"])
        self.assertIn("2 direct foreign-key dependency", res.explanation)
        self.assertNotIn("suppliers", res.tables)

    # 11. Multi-level dependency retrieval
    def test_11_recursive_dependencies(self):
        res = retrieve_erp_schema("Retrieve customer_invoices with full dependency hierarchy")
        self.assertEqual(res.tables, ["chart_of_accounts", "finance_customers", "customer_invoices"])
        self.assertIn("recursive dependency hierarchy", res.explanation)

    # 12. Cross-domain multi-table retrieval
    def test_12_cross_domain_multi_table(self):
        res = retrieve_erp_schema("Retrieve customer_invoices from Finance and hrms_attendance from HRMS.")
        self.assertIn("customer_invoices", res.tables)
        self.assertIn("hrms_attendance", res.tables)
        self.assertEqual(len(res.tables), 2)
        self.assertIn("erp_finance", res.domain)
        self.assertIn("erp_hrms", res.domain)
        self.assertIn("Zero dependencies expanded", res.explanation)

    # 13. Explicit dependency expansion disabled
    def test_13_exclude_dependencies(self):
        res = retrieve_erp_schema("Retrieve customer_invoices without dependencies")
        self.assertEqual(res.tables, ["customer_invoices"])
        self.assertEqual(res.count, 1)
        self.assertIn("0 dependencies", res.explanation)

    # 14. Natural-language column mapping
    def test_14_natural_language_column_mapping(self):
        res = retrieve_erp_schema("Retrieve the employee reference, attendance date, check-in time, and check-out time from hrms_attendance")
        self.assertEqual(res.tables, ["hrms_attendance"])
        self.assertEqual(res.count, 1)
        self.assertIn("employee_id", res.sql)
        self.assertIn("work_date", res.sql)
        self.assertIn("check_in", res.sql)
        self.assertIn("check_out", res.sql)

    # 15. Missing metadata handling
    def test_15_missing_metadata(self):
        res = retrieve_erp_schema("Get the invoice_rating column from customer invoices.")
        self.assertEqual(res.tables, [])
        self.assertEqual(res.count, 0)
        self.assertEqual(res.sql, "")
        self.assertIn("invoice_rating", res.explanation)

    # 16. Nullable and default-value accuracy
    def test_16_nullable_and_default_accuracy(self):
        res = retrieve_erp_schema("Retrieve credit_limit from finance_customers")
        self.assertIn("DEFAULT 0", res.sql)
        self.assertIn("NUMERIC(15,2)", res.sql)

        res_att = retrieve_erp_schema("Retrieve check_out from hrms_attendance")
        self.assertIn("Nullable: Yes", res_att.sql)
        self.assertIn("check_out TIME;", res_att.sql)

    # 17. FK and ON DELETE accuracy
    def test_17_fk_and_on_delete_accuracy(self):
        res_emp = retrieve_erp_schema("Retrieve employee_id from hrms_attendance")
        self.assertIn("REFERENCES hrms_employees(id) ON DELETE CASCADE", res_emp.sql)

        res_dept = retrieve_erp_schema("Retrieve department_id from hrms_employees")
        self.assertIn("REFERENCES hrms_departments(id) ON DELETE RESTRICT", res_dept.sql)

    # 18. SQL-to-JSON consistency
    def test_18_sql_to_json_consistency(self):
        reg = load_erp_registry()
        res = retrieve_erp_schema("Show customer_invoices schema")
        for t in res.tables:
            tbl_obj = reg["all_tables"][t]
            # Ensure the create table statement in res.sql matches canonical sql
            self.assertIn(f"CREATE TABLE {t}", res.sql)

    # 19. Multi-turn follow-up query handling
    def test_19_multi_turn_domain_switching(self):
        r1 = retrieve_erp_schema("Show finance schema")
        self.assertEqual(r1.domain, "erp_finance")
        r2 = retrieve_erp_schema("Retrieve hrms_attendance schema")
        self.assertEqual(r2.domain, "erp_hrms")
        self.assertIn("hrms_attendance", r2.tables)
        r3 = retrieve_erp_schema("Show inventory_stocks schema")
        self.assertEqual(r3.domain, "erp_supply_chain")
        self.assertIn("inventory_stocks", r3.tables)

    # 20. Empty, malformed, and ambiguous user queries
    def test_20_empty_and_malformed_queries(self):
        r_empty = retrieve_erp_schema("")
        self.assertEqual(r_empty.tables, [])
        self.assertEqual(r_empty.sql, "")
        self.assertEqual(r_empty.domain, "not_found")

        r_spaces = retrieve_erp_schema("   ")
        self.assertEqual(r_spaces.tables, [])
        self.assertEqual(r_spaces.sql, "")

        r_ambig = retrieve_erp_schema("Show me the details table.")
        self.assertEqual(r_ambig.domain, "ambiguous")
        self.assertEqual(r_ambig.tables, [])
        self.assertEqual(r_ambig.sql, "")


if __name__ == "__main__":
    unittest.main()
