import unittest
from src.rag.service import retrieve_erp_schema

class TestUserSpecRequirements(unittest.TestCase):

    def test_specified_columns_adds_id_field(self):
        """User requirement: specified columns should add id field."""
        res = retrieve_erp_schema("create customer_invoices table with invoice_number and total_amount")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["customer_invoices"])
        col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(col_names, ["id", "invoice_number", "total_amount"])
        self.assertIn("id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY", res.sql)
        self.assertIn("invoice_number", res.sql)
        self.assertIn("total_amount", res.sql)

    def test_specified_columns_without_create_keyword(self):
        """User requirement: customer_invoices with invoice_number and total_amount should add id."""
        res = retrieve_erp_schema("customer_invoices with invoice_number and total_amount")
        self.assertEqual(res.status, "accepted")
        col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(col_names, ["id", "invoice_number", "total_amount"])
        self.assertIn("id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY", res.sql)

    def test_specified_columns_explicit_without_id(self):
        """If user explicitly asks without id / without primary key, do not add id."""
        res = retrieve_erp_schema("customer_invoices with invoice_number and total_amount without primary key")
        self.assertEqual(res.status, "accepted")
        col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(col_names, ["invoice_number", "total_amount"])
        self.assertNotIn("id UUID PRIMARY KEY", res.sql)

    def test_2_columns_customer_invoices(self):
        """Any 2 columns for a table returns primary id + top business column."""
        res = retrieve_erp_schema("create 2 columns for customer_invoices")
        self.assertEqual(res.status, "accepted")
        col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(col_names, ["id", "invoice_number"])

    def test_3_columns_customer_invoices(self):
        """Any 3 columns for a table returns primary id + top 2 business columns."""
        res = retrieve_erp_schema("create 3 columns for customer_invoices")
        self.assertEqual(res.status, "accepted")
        col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(col_names, ["id", "invoice_number", "total_amount"])

    def test_2_columns_without_pk(self):
        """Any 2 columns without pk returns top 2 business columns."""
        res = retrieve_erp_schema("create 2 columns for customer_invoices without primary key")
        self.assertEqual(res.status, "accepted")
        col_names = [c["name"] for c in res.columns_metadata]
        self.assertEqual(col_names, ["invoice_number", "total_amount"])

    def test_2_tables_finance(self):
        """2 tables for finance returns primary master tables: chart_of_accounts & finance_customers."""
        res = retrieve_erp_schema("create 2 tables for finance")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["chart_of_accounts", "finance_customers"])

    def test_3_tables_finance(self):
        """3 tables for finance returns top 3 primary tables in valid topological order."""
        res = retrieve_erp_schema("create 3 tables for finance")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["chart_of_accounts", "finance_customers", "customer_invoices"])

    def test_2_tables_supply_chain(self):
        """2 tables for supply chain returns product_items & warehouses."""
        res = retrieve_erp_schema("create 2 tables for supply chain")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["product_items", "warehouses"])

    def test_3_tables_supply_chain(self):
        """3 tables for supply chain returns product_items, warehouses & inventory_stocks."""
        res = retrieve_erp_schema("create 3 tables for supply chain")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["product_items", "warehouses", "inventory_stocks"])

    def test_2_tables_hrms(self):
        """2 tables for hrms returns hrms_departments & hrms_designations."""
        res = retrieve_erp_schema("create 2 tables for hrms")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["hrms_departments", "hrms_designations"])

    def test_3_tables_hrms(self):
        """3 tables for hrms returns hrms_departments, hrms_designations & hrms_employees."""
        res = retrieve_erp_schema("create 3 tables for hrms")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["hrms_departments", "hrms_designations", "hrms_employees"])

    def test_2_tables_ecommerce(self):
        """2 tables for ecommerce returns categories & customers."""
        res = retrieve_erp_schema("create 2 tables for ecommerce")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["categories", "customers"])

    def test_3_tables_ecommerce(self):
        """3 tables for ecommerce returns categories, customers & products."""
        res = retrieve_erp_schema("create 3 tables for ecommerce")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["categories", "customers", "products"])

    def test_2_tables_healthcare(self):
        """2 tables for healthcare returns departments & doctors."""
        res = retrieve_erp_schema("create 2 tables for healthcare")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["departments", "doctors"])

    def test_3_tables_healthcare(self):
        """3 tables for healthcare returns departments, doctors & patients."""
        res = retrieve_erp_schema("create 3 tables for healthcare")
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.tables, ["departments", "doctors", "patients"])

if __name__ == "__main__":
    unittest.main()
