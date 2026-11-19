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

def prioritize_table_columns(table_name: str, all_cols: List[str], include_pk: bool = True) -> List[str]:
    """
    Ranks columns by business importance for a given table:
    1. Primary Key ('id') if include_pk is True
    2. Primary business code/identifier/name
    3. Core business value/amount/status/date
    4. Foreign keys / attributes
    5. Audit columns (created_at, updated_at, etc.)
    """
    tname_lower = table_name.lower()
    if tname_lower in TABLE_PRIMARY_COLUMNS_MAP:
        ranked = [c for c in TABLE_PRIMARY_COLUMNS_MAP[tname_lower] if c in all_cols]
        for c in all_cols:
            if c not in ranked:
                ranked.append(c)
    else:
        def col_rank(c: str) -> int:
            cl = c.lower()
            if cl in ("id", "primary_key"):
                return 0
            if any(k in cl for k in ["code", "number", "sku", "name", "title"]):
                return 1
            if any(k in cl for k in ["amount", "total", "price", "salary", "date", "status", "type", "email"]):
                return 2
            if cl.endswith("_id"):
                return 3
            if any(k in cl for k in ["created_at", "updated_at", "password_hash", "deleted_at"]):
                return 5
            return 4
        ranked = sorted(all_cols, key=col_rank)

    if not include_pk:
        ranked = [c for c in ranked if c.lower() not in ("id", "primary_key")]
    elif "id" in all_cols and (not ranked or ranked[0].lower() != "id"):
        ranked = ["id"] + [c for c in ranked if c.lower() != "id"]

    return ranked


__all__ = [
    "DOMAIN_PRIMARY_TABLES",
    "TABLE_PRIMARY_COLUMNS_MAP",
    "prioritize_table_columns",
]
