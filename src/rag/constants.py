from typing import Dict, List

# =====================================================================
# RAG Prompts
# =====================================================================

RAG_SYSTEM_PROMPT = """You are an Enterprise ERP Database Architect specializing in PostgreSQL and MySQL relational schemas.
You have access to a validated, production-grade enterprise schema knowledge base.

Rules:
1. Adhere strictly to the domain relationships, primary/foreign key hierarchies, and column data types provided in the retrieved schema context.
2. Preserve double-entry accounting integrity, financial cutoffs, audit trails, and referential integrity.
3. When extending or modifying the schema, maintain topological ordering so parent tables precede dependent child tables.
4. Output clean, syntactically correct SQL DDL with appropriate constraints and indices.
"""

RAG_CONTEXT_TEMPLATE = """### RETRIEVED ENTERPRISE ERP SCHEMA CONTEXT:
Domain: {domain_name} ({domain_id})
Available Tables: {tables}

{schema_ddl}

### ACTIVE SCHEMA SUMMARY LEDGER:
{summary}

### USER INSTRUCTION:
{user_query}

Generate the exact SQL schema DDL satisfying the instruction while respecting the architectural standards and relationships above.
"""

# =====================================================================
# Domain Primary Tables Prioritization (Top-3 Core Master Entities)
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

# =====================================================================
# Canonical Schema Synonyms
# =====================================================================

TABLE_SYNONYMS: Dict[str, str] = {
    "customer invoices": "customer_invoices",
    "customer invoice": "customer_invoices",
    "invoices": "customer_invoices",
    "invoice": "customer_invoices",
    "finance customers": "finance_customers",
    "finance customer": "finance_customers",
    "customers": "finance_customers",
    "customer": "finance_customers",
    "chart of accounts": "chart_of_accounts",
    "chart of account": "chart_of_accounts",
    "gl accounts": "chart_of_accounts",
    "gl account": "chart_of_accounts",
    "fiscal periods": "fiscal_periods",
    "fiscal period": "fiscal_periods",
    "fiscal": "fiscal_periods",
    "journal entries": "journal_entries",
    "journal entry": "journal_entries",
    "accounting entry": "journal_entries",
    "accounting entries": "journal_entries",
    "hrms departments": "hrms_departments",
    "hrms department": "hrms_departments",
    "departments": "hrms_departments",
    "department": "hrms_departments",
    "hrms designations": "hrms_designations",
    "hrms designation": "hrms_designations",
    "designations": "hrms_designations",
    "designation": "hrms_designations",
    "hrms employees": "hrms_employees",
    "hrms employee": "hrms_employees",
    "attendance": "hrms_attendance",
    "employee attendance": "hrms_attendance",
    "attendance records": "hrms_attendance",
    "hrms attendance": "hrms_attendance",
    "payrolls": "hrms_payrolls",
    "payroll": "hrms_payrolls",
    "hrms payroll": "hrms_payrolls",
    "hrms payrolls": "hrms_payrolls",
    "employee payroll": "hrms_payrolls",
    "product items": "product_items",
    "product item": "product_items",
    "product details": "product_items",
    "inventory items": "product_items",
    "inventory item": "product_items",
    "inventory_items": "product_items",
    "products table": "product_items",
    "items table": "product_items",
    "items": "product_items",
    "inventory stocks": "inventory_stocks",
    "inventory stock": "inventory_stocks",
    "inventory_stocks": "inventory_stocks",
    "inventory_stock": "inventory_stocks",
    "stock information": "inventory_stocks",
    "stock details": "inventory_stocks",
    "inventory": "inventory_stocks",
    "stocks": "inventory_stocks",
    "stock": "inventory_stocks",
    "stock levels": "inventory_stocks",
    "purchase orders": "purchase_orders",
    "purchase order": "purchase_orders",
    "po": "purchase_orders",
    "orders": "purchase_orders",
    "order": "purchase_orders",
    "warehouse information": "warehouses",
    "warehouse details": "warehouses",
    "warehouses": "warehouses",
    "warehouse": "warehouses",
    "supplier information": "suppliers",
    "supplier details": "suppliers",
    "suppliers": "suppliers",
    "supplier": "suppliers",
    # E-Commerce domain synonyms
    "categories": "categories",
    "category": "categories",
    "product categories": "categories",
    "product category": "categories",
    "ecommerce categories": "categories",
    "addresses": "addresses",
    "address": "addresses",
    "customer addresses": "addresses",
    "shipping addresses": "addresses",
    "product variants": "product_variants",
    "product variant": "product_variants",
    "variants": "product_variants",
    "variant": "product_variants",
    "product images": "product_images",
    "product image": "product_images",
    "carts": "carts",
    "cart": "carts",
    "shopping carts": "carts",
    "shopping cart": "carts",
    "cart items": "cart_items",
    "cart item": "cart_items",
    "coupons": "coupons",
    "coupon": "coupons",
    "promo codes": "coupons",
    "discount codes": "coupons",
    "order items": "order_items",
    "order item": "order_items",
    "product reviews": "product_reviews",
    "product review": "product_reviews",
    "reviews": "product_reviews",
    "wishlists": "wishlists",
    "wishlist": "wishlists",
    "saved items": "wishlists",
    "order refunds": "order_refunds",
    "order refund": "order_refunds",
    "refunds": "order_refunds",
    # Healthcare domain synonyms
    "doctors": "doctors",
    "doctor": "doctors",
    "physicians": "doctors",
    "physician": "doctors",
    "patients": "patients",
    "patient": "patients",
    "hospital rooms": "hospital_rooms",
    "hospital room": "hospital_rooms",
    "rooms": "hospital_rooms",
    "patient admissions": "patient_admissions",
    "patient admission": "patient_admissions",
    "admissions": "patient_admissions",
    "vital signs": "vital_signs",
    "vital sign": "vital_signs",
    "vitals": "vital_signs",
    "patient allergies": "patient_allergies",
    "patient allergy": "patient_allergies",
    "allergies": "patient_allergies",
    "appointments": "appointments",
    "appointment": "appointments",
    "doctor appointments": "appointments",
    "medical records": "medical_records",
    "medical record": "medical_records",
    "clinical records": "medical_records",
    "medications": "medications",
    "medication": "medications",
    "drugs": "medications",
    "prescriptions": "prescriptions",
    "prescription": "prescriptions",
    "prescription items": "prescription_items",
    "prescription item": "prescription_items",
    "lab tests": "lab_tests",
    "lab test": "lab_tests",
    "laboratory tests": "lab_tests",
    "lab orders": "lab_orders",
    "lab order": "lab_orders",
    "insurance claims": "insurance_claims",
    "insurance claim": "insurance_claims",
    "patient bills": "patient_bills",
    "patient bill": "patient_bills",
    "hospital bills": "patient_bills",
}

NONEXISTENT_TABLE_PATTERNS: List[str] = [
    "employee_benefits", "employee benefits", "benefits table",
    "salary_history", "salary history", "payment_history", "payment history",
    "sales_orders", "sales order", "sales orders", "tax_management", "tax management",
    "leave_requests", "leave request", "leave requests",
    "supplier_payment", "supplier payment", "customer_complaints", "customer complaints", "customer payment",
    "customer payment history"
]
