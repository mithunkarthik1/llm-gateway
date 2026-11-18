"""
Domain and Table Entity Priority Configurations.
Defines core master entities and primary business column rankings for ERP schema retrieval.
"""
from typing import Dict, List

# =====================================================================
# Domain Primary Tables Prioritization (Top Core Master Entities)
# =====================================================================

DOMAIN_PRIMARY_TABLES: Dict[str, List[str]] = {
    "erp_finance": [
        "chart_of_accounts",
        "finance_customers",
        "customer_invoices",
        "journal_entries",
        "payments"
    ],
    "erp_supply_chain": [
        "product_items",
        "warehouses",
        "inventory_stocks",
        "purchase_orders",
        "suppliers"
    ],
    "erp_hrms": [
        "hrms_departments",
        "hrms_designations",
        "hrms_employees",
        "hrms_attendance",
        "hrms_payrolls"
    ],
    "erp_ecommerce": [
        "categories",
        "customers",
        "products",
        "orders",
        "order_items"
    ],
    "erp_healthcare": [
        "departments",
        "doctors",
        "patients",
        "appointments",
        "prescriptions"
    ]
}

# =====================================================================
# Table Primary Columns Prioritization (PK + Core Business Identifiers)
# =====================================================================

TABLE_PRIMARY_COLUMNS_MAP: Dict[str, List[str]] = {
    "chart_of_accounts": ["id", "account_code", "account_name", "account_type", "is_active"],
    "finance_customers": ["id", "customer_number", "customer_name", "credit_limit", "gl_account_id"],
    "customer_invoices": ["id", "invoice_number", "total_amount", "due_date", "customer_id", "status"],
    "journal_entries": ["id", "entry_number", "entry_date", "description", "posted"],
    "payments": ["id", "payment_number", "amount", "payment_date", "payment_method"],
    "hrms_departments": ["id", "dept_code", "dept_name", "is_active"],
    "hrms_designations": ["id", "designation_name", "grade_level"],
    "hrms_employees": ["id", "employee_code", "full_name", "department_id", "designation_id", "salary"],
    "hrms_attendance": ["id", "employee_id", "work_date", "check_in", "check_out", "status"],
    "hrms_payrolls": ["id", "employee_id", "pay_month", "net_salary", "basic_salary"],
    "product_items": ["id", "sku_code", "product_name", "unit_price", "category"],
    "warehouses": ["id", "warehouse_code", "warehouse_name", "location_city"],
    "inventory_stocks": ["id", "product_id", "warehouse_id", "quantity_on_hand", "reorder_level"],
    "suppliers": ["id", "supplier_code", "supplier_name", "contact_email"],
    "purchase_orders": ["id", "po_number", "order_date", "total_amount", "supplier_id"],
    "categories": ["id", "name", "slug", "parent_id"],
    "customers": ["id", "first_name", "last_name", "email", "phone"],
    "products": ["id", "name", "sku", "price", "stock_quantity", "category_id"],
    "orders": ["id", "order_number", "total_amount", "status", "customer_id"],
    "order_items": ["id", "order_id", "product_id", "quantity", "unit_price"],
    "departments": ["id", "dept_code", "dept_name", "floor_no"],
    "doctors": ["id", "full_name", "specialization", "license_no", "department_id"],
    "patients": ["id", "mrn", "full_name", "date_of_birth", "gender", "phone"],
    "appointments": ["id", "patient_id", "doctor_id", "appointment_date", "status"],
    "prescriptions": ["id", "patient_id", "doctor_id", "prescription_date", "notes"]
}

__all__ = [
    "DOMAIN_PRIMARY_TABLES",
    "TABLE_PRIMARY_COLUMNS_MAP",
]
