#!/usr/bin/env python3
"""DB Nexus LLM end-to-end test runner.

Runs individual SQL-generation tests, multi-turn conversations, and schema
summary/Q&A tests against a FastAPI chat endpoint. Conversations reuse one
session_id so the server can retain context.

Default endpoint: http://127.0.0.1:8000/api/v1/chat
Default request JSON: {"message": "...", "session_id": "..."}

Examples:
  python db_nexus_llm_test_suite.py --list
  python db_nexus_llm_test_suite.py --suite all
  python db_nexus_llm_test_suite.py --suite conversations --endpoint http://localhost:8000/api/v1/chat
  python db_nexus_llm_test_suite.py --message-field prompt --session-field conversation_id
  python db_nexus_llm_test_suite.py --suite summary --api-key YOUR_KEY

Optional SQL parsing:
  pip install sqlglot
  python db_nexus_llm_test_suite.py --validate-sql

IMPORTANT:
- Use a test/staging endpoint, not production.
- Keep the same session ID for every turn in a conversation scenario.
- This script's automatic checks are keyword/evidence checks, not a substitute
  for semantic review or executing generated SQL against a disposable database.
- The script does not execute SQL or modify your database.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

@dataclass
class TestStep:
    prompt: str
    expected_terms: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    mode: str = "answer"  # answer | sql | clarify | either
    note: str = ""


@dataclass
class TestCase:
    case_id: str
    category: str
    prompt: str
    expected_terms: List[str]
    setup: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    mode: str = "sql"  # sql | answer | clarify | either
    note: str = ""


@dataclass
class Scenario:
    scenario_id: str
    category: str
    title: str
    steps: List[TestStep]


# 52 single-request test cases. M13-M15 are represented by multi-turn
# scenarios below, giving 55 base cases when those 3 scenarios are included.
INDIVIDUAL_CASES: List[TestCase] = [
    # A. Single-table creation (B01-B06)
    TestCase("B01", "single_table", "Create a users table with id, name, email and phone_number.",
             ["users", "id", "name", "email", "phone_number"]),
    TestCase("B02", "single_table", "Create a products table with product_id, product_name and price.",
             ["products", "product_id", "product_name", "price"]),
    TestCase("B03", "single_table", "Generate an employees table with id, name, salary and joining_date.",
             ["employees", "id", "name", "salary", "joining_date"]),
    TestCase("B04", "single_table", "Create a customers table with customer_id as the primary key, name and email as mandatory fields.",
             ["customers", "customer_id", "primary key", "not null"]),
    TestCase("B05", "single_table", "Create a categories table with id and category_name. category_name must be unique.",
             ["categories", "category_name", "unique"]),
    TestCase("B06", "single_table", "Create an orders table with id, order_number and created_at. Set created_at to the current timestamp by default.",
             ["orders", "order_number", "created_at", "current_timestamp"]),

    # B. Data types and constraints (B07-B10)
    TestCase("B07", "data_types_constraints", "Create a students table with id, name, age, email and date_of_birth. Use appropriate data types.",
             ["students", "id", "name", "age", "email", "date_of_birth"]),
    TestCase("B08", "data_types_constraints", "Create an accounts table with id, account_number and balance. Account number must be unique.",
             ["accounts", "account_number", "balance", "unique"]),
    TestCase("B09", "data_types_constraints", "Create an employees table with id, name, salary and status. Salary must be greater than zero.",
             ["employees", "salary", "check"]),
    TestCase("B10", "data_types_constraints", "Create a users table with an auto-incrementing primary key and a unique email address.",
             ["users", "auto_increment", "primary key", "email", "unique"]),

    # C. Multiple-table creation (B11-B14)
    TestCase("B11", "multiple_tables", "Create customers and orders tables. Each order should belong to a customer.",
             ["customers", "orders", "customer_id", "references"]),
    TestCase("B12", "multiple_tables", "Create departments and employees tables. One department can have multiple employees, but each employee belongs to one department.",
             ["departments", "employees", "department_id", "references"]),
    TestCase("B13", "multiple_tables", "Create categories and products tables. Each product belongs to one category.",
             ["categories", "products", "category_id", "references"]),
    TestCase("B14", "multiple_tables", "Create users, roles and user_roles tables. A user can have multiple roles, and a role can belong to multiple users.",
             ["users", "roles", "user_roles", "user_id", "role_id"]),

    # D. Real-world schemas (M01-M06)
    TestCase("M01", "real_world_schema", "Create an employee management database with employees, departments and job_roles. Each employee belongs to one department and one job role. Include employee name, salary, joining date, department name and job title.",
             ["employees", "departments", "job_roles", "department_id", "job_role_id", "salary", "joining_date"]),
    TestCase("M02", "real_world_schema", "Create an e-commerce database with customers, products, categories and orders. Include appropriate IDs, prices, customer information and relationships.",
             ["customers", "products", "categories", "orders", "references"]),
    TestCase("M03", "real_world_schema", "Create a library management database with books, authors, members and borrow_records. A book can have multiple authors and a member can borrow multiple books.",
             ["books", "authors", "book_authors", "members", "borrow_records"]),
    TestCase("M04", "real_world_schema", "Create a hospital database with patients, doctors, appointments and prescriptions. Each appointment should reference a patient and a doctor.",
             ["patients", "doctors", "appointments", "prescriptions", "patient_id", "doctor_id"]),
    TestCase("M05", "real_world_schema", "Create a school database with students, teachers, courses and enrollments. A student can enroll in multiple courses and a course can have multiple students.",
             ["students", "teachers", "courses", "enrollments"]),
    TestCase("M06", "real_world_schema", "Create a banking database with customers, accounts and transactions. One customer can have multiple accounts and each account can have multiple transactions.",
             ["customers", "accounts", "transactions", "customer_id", "account_id"]),

    # E. Advanced constraints and relationships (M07-M12)
    TestCase("M07", "relationships_constraints", "Create an employees table with id, name and manager_id. manager_id should reference another employee in the same table.",
             ["employees", "manager_id", "references"]),
    TestCase("M08", "relationships_constraints", "Create orders and order_items tables. Each order can contain multiple items. Each item should have product_id, quantity and unit_price.",
             ["orders", "order_items", "order_id", "product_id", "quantity", "unit_price"]),
    TestCase("M09", "relationships_constraints", "Create a products table with id, category_id, SKU, price and stock_quantity. SKU must be unique and price must be greater than zero.",
             ["products", "category_id", "sku", "price", "stock_quantity", "unique", "check"]),
    TestCase("M10", "relationships_constraints", "Create a customers table with email, phone_number and address. Email must be unique. Phone number should be optional.",
             ["customers", "email", "unique", "phone_number", "address"]),
    TestCase("M11", "relationships_constraints", "Create a payments table with id, order_id, amount and payment_status. Default payment_status to pending and ensure amount is greater than zero.",
             ["payments", "order_id", "amount", "payment_status", "pending", "check"]),
    TestCase("M12", "relationships_constraints", "Create a student_courses table with student_id and course_id. Use both columns together as the composite primary key.",
             ["student_courses", "student_id", "course_id", "primary key"]),

    # G. ALTER operations (M16-M22); setup is run first in same session.
    TestCase("M16", "alter", "Add an address column of type VARCHAR(255) to the customers table.",
             ["alter table", "customers", "address", "varchar"],
             setup=["Create a customers table with id INT PRIMARY KEY, name VARCHAR(100), and email VARCHAR(150)."]),
    TestCase("M17", "alter", "Change the employees salary column to DECIMAL(12,2).",
             ["alter table", "employees", "salary", "decimal"],
             setup=["Create an employees table with id INT PRIMARY KEY, name VARCHAR(100), salary INT."]),
    TestCase("M18", "alter", "Rename the phone column to phone_number in the customers table.",
             ["alter table", "customers", "phone", "phone_number"],
             setup=["Create a customers table with id INT PRIMARY KEY, name VARCHAR(100), phone VARCHAR(20)."]),
    TestCase("M19", "alter", "Make the email column NOT NULL in the users table.",
             ["alter table", "users", "email", "not null"],
             setup=["Create a users table with id INT PRIMARY KEY and email VARCHAR(150) NULL."]),
    TestCase("M20", "alter", "Add a unique constraint to the SKU column in the products table.",
             ["alter table", "products", "sku", "unique"],
             setup=["Create a products table with id INT PRIMARY KEY and sku VARCHAR(80)."]),
    TestCase("M21", "alter", "Add a department_id foreign key to employees referencing departments(id).",
             ["alter table", "employees", "department_id", "references", "departments"],
             setup=["Create a departments table with id INT PRIMARY KEY and name VARCHAR(100). Create an employees table with id INT PRIMARY KEY and department_id INT."]),
    TestCase("M22", "alter", "Remove the temporary_notes column from the customers table.",
             ["alter table", "customers", "drop", "temporary_notes"],
             setup=["Create a customers table with id INT PRIMARY KEY, name VARCHAR(100), and temporary_notes TEXT."]),

    # H. DROP and RENAME operations (M23-M27)
    TestCase("M23", "drop_rename", "Drop the temporary_logs table.",
             ["drop table", "temporary_logs"], setup=["Create a temporary_logs table with id INT PRIMARY KEY."]),
    TestCase("M24", "drop_rename", "Rename the employees table to staff.",
             ["rename table", "employees", "staff"], setup=["Create an employees table with id INT PRIMARY KEY and name VARCHAR(100)."]),
    TestCase("M25", "drop_rename", "Drop the email column from the users table.",
             ["alter table", "users", "drop", "email"], setup=["Create a users table with id INT PRIMARY KEY and email VARCHAR(150)."]),
    TestCase("M26", "drop_rename", "Remove the foreign key constraint named fk_employees_department_id from employees.",
             ["alter table", "employees", "drop", "fk_employees_department_id"],
             setup=["Create departments with id INT PRIMARY KEY. Create employees with id INT PRIMARY KEY and department_id INT, and add a foreign key constraint named fk_employees_department_id on employees(department_id) referencing departments(id)."]),
    TestCase("M27", "drop_rename", "Drop the old_orders table if it exists.",
             ["drop table", "old_orders", "if exists"], setup=["Create an old_orders table with id INT PRIMARY KEY."]),

    # I. Invalid/ambiguous prompts (M28-M33)
    TestCase("M28", "validation", "Create a table.",
             ["table"], mode="clarify", forbidden_terms=["create table"], note="Should ask for table name and/or columns; SQL generation is not required."),
    TestCase("M29", "validation", "Create a users table with columns id, id and name. Both id columns should be primary keys.",
             ["duplicate", "id"], mode="clarify", forbidden_terms=["create table users"], note="Should flag duplicate column names."),
    TestCase("M30", "validation", "Create an orders table with customer_id as a foreign key referencing customers(id). Do not create the customers table.",
             ["orders", "customers", "customer_id"], mode="sql", forbidden_terms=["create table customers"], note="May create orders referencing an existing customers table, but must not create customers."),
    TestCase("M31", "validation", "Create a table with salary as VARCHAR(100), and make sure salary is greater than zero.",
             ["salary"], mode="either", note="Should flag the type/constraint mismatch or clearly explain/correct it."),
    TestCase("M32", "validation", "Create a products table with an invalid data type called MONEYTEXT.",
             ["moneytext"], mode="clarify", forbidden_terms=["create table products"], note="Should reject/clarify unsupported type rather than silently invent a type."),
    TestCase("M33", "validation", "Drop the table.",
             ["which", "table"], mode="clarify", forbidden_terms=["drop table"], note="Must ask which table; must not drop an unspecified table."),

    # J. Complex real-world schemas (M34-M38)
    TestCase("M34", "complex_schema", "Create an e-commerce database with users, addresses, categories, products, carts, cart_items, orders, order_items and payments. Include all appropriate primary keys, foreign keys and relationships.",
             ["users", "addresses", "categories", "products", "carts", "cart_items", "orders", "order_items", "payments"]),
    TestCase("M35", "complex_schema", "Create a project management database with users, projects, tasks, comments and task_assignments. A task can be assigned to multiple users.",
             ["users", "projects", "tasks", "comments", "task_assignments"]),
    TestCase("M36", "complex_schema", "Create an employee payroll system with employees, departments, salaries, payroll_records and deductions. Include appropriate relationships and salary-related data types.",
             ["employees", "departments", "salaries", "payroll_records", "deductions"]),
    TestCase("M37", "complex_schema", "Create an inventory management system with warehouses, products, suppliers, stock_movements and purchase_orders. Include the necessary relationships.",
             ["warehouses", "products", "suppliers", "stock_movements", "purchase_orders"]),
    TestCase("M38", "complex_schema", "Create a social media database with users, posts, comments, likes and followers. A user can follow multiple users, and a user can be followed by multiple users.",
             ["users", "posts", "comments", "likes", "followers"]),

    # K. Robustness (M39-M41)
    TestCase("M39", "robustness", "Gerante a user tabel with id, name and email.",
             ["users", "id", "name", "email"]),
    TestCase("M40", "robustness", "Create a table to store employee information. Choose suitable columns, data types and constraints.",
             ["employees", "id", "name"]),
    TestCase("M41", "robustness", "Create an employees table. Actually, use staff as the table name instead. Include id, name and salary.",
             ["staff", "id", "name", "salary"], forbidden_terms=["create table employees"]),
]


# Multi-turn cases. Every scenario gets exactly one session ID and all prompts
# in it are sent sequentially with that same ID.
CONVERSATIONS: List[Scenario] = [
    Scenario("M13", "followup_creation", "Basic table creation and follow-up", [
        TestStep("Create a customers table with id, first_name, last_name, email and phone_number.", ["customers", "first_name", "last_name", "email"], mode="sql"),
        TestStep("Now create an orders table with id, customer_id, order_date and total_amount. Each order should belong to a customer.", ["orders", "customer_id", "references"], mode="sql"),
        TestStep("Now create an order_items table with id, order_id, product_id, quantity and unit_price.", ["order_items", "order_id", "product_id", "quantity", "unit_price"], mode="sql"),
        TestStep("Summarize all the tables we have created so far and explain their relationships.", ["customers", "orders", "order_items", "one-to-many"], mode="answer"),
    ]),
    Scenario("M14", "followup_alter", "Follow-up constraint modification", [
        TestStep("Create an employees table with id, name, email and salary.", ["employees", "id", "name", "email", "salary"], mode="sql"),
        TestStep("Make email unique.", ["email", "unique"], mode="either"),
        TestStep("Also make salary mandatory.", ["salary", "not null"], mode="either"),
    ]),
    Scenario("M15", "followup_context", "Context reference and rename a column", [
        TestStep("Create a products table with id, name, price and stock.", ["products", "id", "name", "price", "stock"], mode="sql"),
        TestStep("Add a description column to that table.", ["products", "description", "add"], mode="sql"),
        TestStep("Rename that column to product_description.", ["products", "description", "product_description", "rename"], mode="sql"),
    ]),
    Scenario("F01", "followup_creation", "Users, profiles and relationship", [
        TestStep("Create a users table with id, name and email.", ["users", "id", "name", "email"], mode="sql"),
        TestStep("Create a profiles table. Each profile should belong to one user.", ["profiles", "user_id", "references"], mode="sql"),
        TestStep("Add date_of_birth and profile_image_url to that table.", ["date_of_birth", "profile_image_url"], mode="sql"),
        TestStep("Make its user_id unique because each user should have only one profile.", ["user_id", "unique"], mode="either"),
        TestStep("What is the relationship between these two tables?", ["users", "profiles", "one-to-one"], mode="answer"),
        TestStep("List the columns of the second table.", ["profiles", "user_id", "date_of_birth", "profile_image_url"], mode="answer"),
    ]),
    Scenario("F02", "followup_alter", "Employee schema and memory recall", [
        TestStep("Create an employee management schema with employees, departments and job_roles. Include primary keys and foreign keys.", ["employees", "departments", "job_roles"], mode="sql"),
        TestStep("Add email, phone_number, salary and joining_date to the employees table if they are not already included.", ["email", "phone_number", "salary", "joining_date"], mode="either"),
        TestStep("Add a unique constraint to the employees email column.", ["email", "unique"], mode="either"),
        TestStep("What tables have we created so far?", ["employees", "departments", "job_roles"], mode="answer"),
        TestStep("What columns are available in the employees table? Include their data types and constraints.", ["employees", "email", "phone_number", "salary", "joining_date"], mode="answer"),
    ]),
    Scenario("F03", "schema_summary", "E-commerce schema summary and relationship explanation", [
        TestStep("Create an e-commerce database with customers, products, categories and orders.", ["customers", "products", "categories", "orders"], mode="sql"),
        TestStep("Each product must belong to one category. Each order must belong to one customer.", ["products", "categories", "orders", "customers"], mode="either"),
        TestStep("Each order can contain multiple order_items. Each order_item should reference one product.", ["order_items", "order_id", "product_id"], mode="either"),
        TestStep("Explain the complete database structure in simple English.", ["customers", "products", "categories", "orders", "order_items"], mode="answer"),
        TestStep("How many tables have we created? List their names.", ["5", "customers", "products", "categories", "orders", "order_items"], mode="answer", note="Expected table count is five if order_items was created."),
        TestStep("Explain every foreign key relationship and identify the parent and child tables.", ["customers", "orders", "categories", "products", "order_items"], mode="answer"),
    ]),
    Scenario("F04", "schema_modification", "Preserve schema through multiple ALTER operations", [
        TestStep("Create a products table with id, product_name, price and stock_quantity.", ["products", "product_name", "price", "stock_quantity"], mode="sql"),
        TestStep("Add a description column to that table.", ["description", "add"], mode="sql"),
        TestStep("Make the product_name column unique.", ["product_name", "unique"], mode="either"),
        TestStep("Change the price column to DECIMAL(12,2).", ["price", "decimal"], mode="sql"),
        TestStep("Rename description to product_description.", ["description", "product_description", "rename"], mode="sql"),
        TestStep("Show me the updated structure of the products table.", ["products", "id", "product_name", "price", "stock_quantity", "product_description"], mode="answer"),
        TestStep("Summarize all modifications performed on this table.", ["unique", "decimal", "product_description"], mode="answer"),
    ]),
    Scenario("F05", "drop_and_history", "DROP operation and schema history", [
        TestStep("Create a database with users, products, orders and order_items. Include their appropriate relationships.", ["users", "products", "orders", "order_items"], mode="sql"),
        TestStep("Drop the order_items table.", ["drop table", "order_items"], mode="sql"),
        TestStep("What happens to the orders table when order_items is dropped?", ["orders", "remains"], mode="answer"),
        TestStep("Which tables currently remain in the schema?", ["users", "products", "orders"], mode="answer"),
        TestStep("Summarize the changes we made to the database.", ["order_items", "dropped"], mode="answer"),
        TestStep("Did we remove any foreign key relationships? Explain which relationships remain.", ["order_items", "orders", "products"], mode="answer"),
    ]),
    Scenario("F06", "rename_and_history", "Rename table and retain context", [
        TestStep("Create an employees table with id, name, salary and joining_date.", ["employees", "id", "name", "salary", "joining_date"], mode="sql"),
        TestStep("Rename employees to staff.", ["employees", "staff", "rename"], mode="sql"),
        TestStep("Add an email column to the renamed table.", ["staff", "email", "add"], mode="sql"),
        TestStep("What was the original table name, and what is its current name?", ["employees", "staff"], mode="answer"),
        TestStep("Summarize the complete schema modifications we performed.", ["employees", "staff", "email"], mode="answer"),
    ]),
    Scenario("F07", "clarification", "Missing information and clarification", [
        TestStep("Create a customers table with id, name and email.", ["customers", "id", "name", "email"], mode="sql"),
        TestStep("Add a relationship to the orders table.", ["orders"], mode="clarify", note="A clarification is acceptable because cardinality/key details are missing."),
        TestStep("What information do you need from me to create this relationship correctly?", ["customer", "relationship"], mode="answer"),
        TestStep("Each customer can have multiple orders, and each order belongs to exactly one customer. Use customer_id in orders.", ["orders", "customer_id", "customers"], mode="sql"),
        TestStep("Summarize the relationship and explain what changed.", ["customers", "orders", "one-to-many"], mode="answer"),
    ]),
    Scenario("F08", "long_context", "Long-context retention across ten turns", [
        TestStep("Create an e-commerce database with customers, products and categories.", ["customers", "products", "categories"], mode="sql"),
        TestStep("Add addresses and connect them to customers.", ["addresses", "customer_id"], mode="either"),
        TestStep("Add orders and connect them to customers.", ["orders", "customer_id"], mode="either"),
        TestStep("Add order_items and connect them to orders and products.", ["order_items", "order_id", "product_id"], mode="either"),
        TestStep("Add payments and connect them to orders.", ["payments", "order_id"], mode="either"),
        TestStep("Add reviews. Each review should belong to a customer and a product.", ["reviews", "customer_id", "product_id"], mode="either"),
        TestStep("Add a created_at timestamp to all tables that do not already have it.", ["created_at"], mode="either"),
        TestStep("Which tables did we create in the first prompt?", ["customers", "products", "categories"], mode="answer"),
        TestStep("Which table connects orders and products?", ["order_items"], mode="answer"),
        TestStep("Give me the complete final schema summary, including all tables, their important columns, primary keys, foreign keys and relationships.", ["customers", "products", "categories", "addresses", "orders", "order_items", "payments", "reviews", "primary key", "foreign key"], mode="answer"),
    ]),
    Scenario("F09", "instruction_correction", "Correct previous instructions", [
        TestStep("Create a users table with id, username, email and password_hash. Make username unique.", ["users", "username", "email", "password_hash", "unique"], mode="sql"),
        TestStep("Create a roles table with id and role_name. Make role_name unique.", ["roles", "role_name", "unique"], mode="sql"),
        TestStep("Create a user_roles table to connect users and roles using a composite primary key.", ["user_roles", "user_id", "role_id", "primary key"], mode="sql"),
        TestStep("Actually, rename username to display_name.", ["username", "display_name", "rename"], mode="sql"),
        TestStep("Keep email unique as well. Do not change any other columns.", ["email", "unique"], mode="either"),
        TestStep("Summarize the final schema and list all the changes made since the first prompt.", ["display_name", "email", "password_hash", "user_roles", "roles"], mode="answer"),
    ]),
    Scenario("F10", "schema_qa", "Natural-language database understanding without SQL", [
        TestStep("Create an employee management database with employees, departments, job_roles and salary_history. An employee belongs to one department and one job role. An employee can have multiple salary history records.", ["employees", "departments", "job_roles", "salary_history"], mode="sql"),
        TestStep("Explain the purpose of each table.", ["employees", "departments", "job_roles", "salary_history"], mode="answer"),
        TestStep("Which table stores the employee's salary history?", ["salary_history"], mode="answer"),
        TestStep("Can one employee have multiple salary records? Explain based on the schema.", ["yes", "salary_history"], mode="answer"),
        TestStep("Which column connects employees with departments?", ["department_id"], mode="answer"),
        TestStep("Which tables have foreign key relationships?", ["employees", "departments", "job_roles", "salary_history"], mode="answer"),
        TestStep("What changes would be needed to allow an employee to work in multiple departments?", ["junction"], mode="answer", note="Accept a junction/association table explanation; manually review the proposed keys and cardinality."),
        TestStep("Summarize this database for a beginner without writing SQL.", ["employees", "departments", "job_roles", "salary_history"], mode="answer"),
    ]),
    Scenario("F11", "summary_qa", "Twenty-five schema summary and Q&A prompts", [
        TestStep("Create an online shopping database with customers, addresses, products, categories, orders, order_items, payments and temp_logs. Include primary keys, foreign keys and suitable constraints. Make customers.email unique and products.sku unique.", ["customers", "addresses", "products", "categories", "orders", "order_items", "payments", "temp_logs"], mode="sql"),
        TestStep("Add delivery_status to orders with a default value of pending.", ["delivery_status", "pending"], mode="either"),
        TestStep("Add a nullable, unique tracking_number column to orders.", ["tracking_number", "unique"], mode="either"),
        TestStep("Add discount_amount DECIMAL(10,2) to order_items with a default value of zero.", ["discount_amount", "decimal", "default"], mode="either"),
        TestStep("Rename temp_logs to audit_logs.", ["temp_logs", "audit_logs", "rename"], mode="sql"),
        TestStep("Drop the audit_logs table.", ["drop table", "audit_logs"], mode="sql"),
        TestStep("S01: What have we created so far?", ["customers", "addresses", "products", "categories", "orders", "order_items", "payments"], mode="answer"),
        TestStep("S02: Give me a summary of our database.", ["customers", "products", "orders", "order_items"], mode="answer"),
        TestStep("S03: How many tables are currently available?", ["7"], mode="answer", note="Seven current tables, assuming each requested table was created."),
        TestStep("S04: List all the table names.", ["customers", "addresses", "products", "categories", "orders", "order_items", "payments"], mode="answer"),
        TestStep("S05: What columns are available in the customers table?", ["customers", "id", "email"], mode="answer"),
        TestStep("S06: What is the primary key of each table?", ["primary key", "customers", "orders", "products"], mode="answer"),
        TestStep("S07: List all foreign key relationships.", ["foreign key", "references"], mode="answer"),
        TestStep("S08: Which table is connected to products?", ["products", "order_items"], mode="answer"),
        TestStep("S09: Explain the relationships in simple English.", ["customers", "orders", "products", "order_items"], mode="answer"),
        TestStep("S10: Which tables have one-to-many relationships?", ["one-to-many"], mode="answer"),
        TestStep("S11: Which tables have many-to-many relationships?", ["many-to-many"], mode="answer", note="If the schema has no many-to-many relationship, it should say none rather than invent one."),
        TestStep("S12: Do we have any self-referencing relationships?", ["self-referenc"], mode="answer", note="Should report none if no self-reference exists."),
        TestStep("S13: Which columns are unique?", ["email", "sku", "tracking_number"], mode="answer"),
        TestStep("S14: Which columns allow NULL values?", ["null"], mode="answer"),
        TestStep("S15: Which columns have default values?", ["default", "delivery_status", "discount_amount"], mode="answer"),
        TestStep("S16: What changes did we make after the initial creation?", ["delivery_status", "tracking_number", "discount_amount"], mode="answer"),
        TestStep("S17: What was the last operation we performed?", ["drop", "audit_logs"], mode="answer"),
        TestStep("S18: What was the original name of the renamed table?", ["temp_logs", "audit_logs"], mode="answer"),
        TestStep("S19: Which tables have been dropped?", ["audit_logs"], mode="answer"),
        TestStep("S20: Are there any tables that do not have foreign key relationships?", ["customers", "categories"], mode="answer"),
        TestStep("S21: Explain the complete schema without writing SQL.", ["customers", "addresses", "products", "categories", "orders", "order_items", "payments"], mode="answer"),
        TestStep("S22: Show me the final updated schema after all modifications.", ["delivery_status", "tracking_number", "discount_amount"], mode="answer"),
        TestStep("S23: What is the purpose of the order_items table?", ["order_items", "orders", "products"], mode="answer"),
        TestStep("S24: If I delete a customer, what related records might be affected?", ["customers", "orders", "addresses"], mode="answer", note="Should explain FK dependencies and not assume cascade behavior unless specified."),
        TestStep("S25: What information is missing from our current schema?", ["missing"], mode="answer", note="Open-ended; manually review whether gaps are grounded and clearly labeled as suggestions."),
    ]),
]


# ---------------------------------------------------------------------------
# HTTP and response handling
# ---------------------------------------------------------------------------

def build_payload(prompt: str, session_id: str, args: argparse.Namespace,
                  history: List[Dict[str, str]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    try:
        extra = json.loads(args.extra_payload) if args.extra_payload else {}
        if not isinstance(extra, dict):
            raise ValueError("--extra-payload must be a JSON object")
        payload.update(extra)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --extra-payload JSON: {exc}") from exc

    if args.history_mode == "messages":
        payload[args.messages_field] = history + [{"role": "user", "content": prompt}]
    else:
        if args.message_field:
            payload[args.message_field] = prompt
    if args.history_mode != "none" and args.session_field:
        payload[args.session_field] = session_id
    return payload


def call_api(prompt: str, session_id: str, args: argparse.Namespace,
             history: List[Dict[str, str]]) -> Tuple[Optional[Any], Optional[str], float, int]:
    payload = build_payload(prompt, session_id, args, history)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if args.api_key:
        headers[args.api_key_header] = f"{args.api_key_prefix}{args.api_key}"
    for item in args.header:
        if ":" not in item:
            raise ValueError(f"Invalid --header {item!r}; expected 'Name: Value'")
        key, value = item.split(":", 1)
        headers[key.strip()] = value.strip()

    request = Request(args.endpoint, data=body, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
        elapsed = time.perf_counter() - start
        return None, f"HTTP {status}: {raw[:1500]}", elapsed, status
    except (URLError, TimeoutError, OSError) as exc:
        elapsed = time.perf_counter() - start
        return None, f"Connection error: {exc}", elapsed, 0
    elapsed = time.perf_counter() - start
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        data = {"raw_text": raw}
    return data, None, elapsed, status


def flatten_values(value: Any) -> List[str]:
    output: List[str] = []
    if isinstance(value, dict):
        for v in value.values():
            output.extend(flatten_values(v))
    elif isinstance(value, list):
        for v in value:
            output.extend(flatten_values(v))
    elif isinstance(value, (str, int, float, bool)):
        output.append(str(value))
    return output


def response_text(data: Any, error: Optional[str] = None) -> str:
    if error:
        return error
    parts = flatten_values(data)
    return "\n".join(parts)


SQL_KEYS = {"sql", "sql_query", "generated_sql", "ddl", "statement", "statements", "query"}
SQL_START = re.compile(r"(?im)^\s*(CREATE|ALTER|DROP|RENAME|TRUNCATE|COMMENT|USE)\b")


def extract_sql(data: Any) -> str:
    found: List[str] = []

    def walk(obj: Any, parent_key: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in SQL_KEYS and isinstance(value, str) and value.strip():
                    found.append(value.strip())
                else:
                    walk(value, key)
        elif isinstance(obj, list):
            for value in obj:
                walk(value, parent_key)

    walk(data)
    # Prefer values explicitly returned as SQL fields.
    unique: List[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    if unique:
        return "\n\n".join(unique)

    # Fallback for APIs that return SQL only inside a message/string.
    text = "\n".join(flatten_values(data))
    matches = list(SQL_START.finditer(text))
    if not matches:
        return ""
    chunks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[match.start():end].strip()
        if chunk:
            chunks.append(chunk)
    return "\n\n".join(chunks)


def normalize_for_search(text: str) -> str:
    text = text.lower().replace("`", "").replace('"', "").replace("'", "")
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def check_terms(text: str, expected: Sequence[str], forbidden: Sequence[str]) -> Tuple[List[str], List[str]]:
    normalized = normalize_for_search(text)
    matched = [term for term in expected if normalize_for_search(term) in normalized]
    missing = [term for term in expected if normalize_for_search(term) not in normalized]
    forbidden_found = [term for term in forbidden if normalize_for_search(term) in normalized]
    return matched, missing + [f"FORBIDDEN:{term}" for term in forbidden_found]


def parse_sql_status(sql: str, dialect: str) -> Tuple[Optional[bool], str]:
    if not sql.strip():
        return None, "No SQL extracted"
    try:
        import sqlglot  # type: ignore
    except ImportError:
        return None, "sqlglot not installed; syntax parsing skipped"
    try:
        parsed = sqlglot.parse(sql, read=dialect)
        if not parsed or any(statement is None for statement in parsed):
            return False, "Parser returned an empty/invalid statement"
        return True, f"Parsed {len(parsed)} SQL statement(s)"
    except Exception as exc:  # sqlglot raises several parser exception types
        return False, f"SQL parse error: {exc}"


def score_step(step: TestStep, data: Any, error: Optional[str], status_code: int,
               args: argparse.Namespace) -> Dict[str, Any]:
    text = response_text(data, error)
    sql = extract_sql(data) if data is not None else ""
    matched, problems = check_terms(text, step.expected_terms, [])
    sql_normalized = normalize_for_search(sql)
    forbidden_found = [term for term in step.forbidden_terms if normalize_for_search(term) in sql_normalized]
    problems.extend([f"FORBIDDEN_SQL:{term}" for term in forbidden_found])
    sql_required = step.mode == "sql"
    clarification_required = step.mode == "clarify"
    if sql_required and not sql:
        problems.append("EXPECTED_SQL_NOT_FOUND")
    if clarification_required and sql:
        problems.append("UNEXPECTED_SQL_FOR_CLARIFICATION_TEST")
    parse_ok, parse_message = (None, "Not requested")
    if args.validate_sql and sql:
        parse_ok, parse_message = parse_sql_status(sql, args.dialect)
        if parse_ok is False:
            problems.append("SQL_PARSE_FAILED")
    success = error is None and 200 <= status_code < 300
    passed = success and not problems and len(matched) == len(step.expected_terms)
    return {
        "passed": passed,
        "http_ok": success,
        "http_status": status_code,
        "matched_terms": matched,
        "missing_or_forbidden": problems,
        "expected_terms": step.expected_terms,
        "sql": sql,
        "sql_parse_ok": parse_ok,
        "sql_parse_message": parse_message,
        "response": text,
    }


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def run_one_case(case: TestCase, args: argparse.Namespace, results: List[Dict[str, Any]]) -> None:
    session_id = f"nexus-{case.case_id.lower()}-{uuid.uuid4().hex[:8]}"
    history: List[Dict[str, str]] = []
    blocked = False
    for index, setup_prompt in enumerate(case.setup, start=1):
        data, error, elapsed, status = call_api(setup_prompt, session_id, args, history)
        setup_text = response_text(data, error)
        history.append({"role": "user", "content": setup_prompt})
        history.append({"role": "assistant", "content": setup_text})
        if error or not (200 <= status < 300):
            results.append({
                "test_id": f"{case.case_id}-SETUP{index}", "suite": "individual", "category": case.category,
                "title": "Prerequisite setup", "prompt": setup_prompt, "passed": False, "blocked": True,
                "http_status": status, "elapsed_seconds": round(elapsed, 3), "expected_terms": [],
                "matched_terms": [], "missing_or_forbidden": [error or f"HTTP {status}"],
                "sql": extract_sql(data) if data is not None else "", "sql_parse_ok": None,
                "sql_parse_message": "Setup failed", "response": setup_text,
                "manual_score_0_10": "", "manual_notes": "",
            })
            blocked = True
            break
    if blocked:
        results.append({
            "test_id": case.case_id, "suite": "individual", "category": case.category,
            "title": "Main test blocked by failed setup", "prompt": case.prompt, "passed": False,
            "blocked": True, "http_status": 0, "elapsed_seconds": 0, "expected_terms": case.expected_terms,
            "matched_terms": [], "missing_or_forbidden": ["PREREQUISITE_SETUP_FAILED"], "sql": "",
            "sql_parse_ok": None, "sql_parse_message": "Not run", "response": "",
            "manual_score_0_10": "", "manual_notes": "",
        })
        return

    step = TestStep(case.prompt, case.expected_terms, case.forbidden_terms, case.mode, case.note)
    data, error, elapsed, status = call_api(case.prompt, session_id, args, history)
    checked = score_step(step, data, error, status, args)
    results.append({
        "test_id": case.case_id, "suite": "individual", "category": case.category,
        "title": case.note or case.category, "prompt": case.prompt, "passed": checked["passed"],
        "blocked": False, "http_status": status, "elapsed_seconds": round(elapsed, 3),
        "expected_terms": case.expected_terms, "matched_terms": checked["matched_terms"],
        "missing_or_forbidden": checked["missing_or_forbidden"], "sql": checked["sql"],
        "sql_parse_ok": checked["sql_parse_ok"], "sql_parse_message": checked["sql_parse_message"],
        "response": checked["response"], "manual_score_0_10": "", "manual_notes": "",
    })


def run_scenario(scenario: Scenario, args: argparse.Namespace, results: List[Dict[str, Any]]) -> None:
    session_id = f"nexus-{scenario.scenario_id.lower()}-{uuid.uuid4().hex[:8]}"
    history: List[Dict[str, str]] = []
    for index, step in enumerate(scenario.steps, start=1):
        data, error, elapsed, status = call_api(step.prompt, session_id, args, history)
        checked = score_step(step, data, error, status, args)
        results.append({
            "test_id": f"{scenario.scenario_id}-T{index:02d}", "suite": "conversation", "category": scenario.category,
            "title": f"{scenario.title} / turn {index}", "prompt": step.prompt, "passed": checked["passed"],
            "blocked": False, "http_status": status, "elapsed_seconds": round(elapsed, 3),
            "expected_terms": step.expected_terms, "matched_terms": checked["matched_terms"],
            "missing_or_forbidden": checked["missing_or_forbidden"], "sql": checked["sql"],
            "sql_parse_ok": checked["sql_parse_ok"], "sql_parse_message": checked["sql_parse_message"],
            "response": checked["response"], "manual_score_0_10": "", "manual_notes": step.note,
        })
        assistant_text = response_text(data, error)
        history.append({"role": "user", "content": step.prompt})
        history.append({"role": "assistant", "content": assistant_text})


def print_list() -> None:
    print("INDIVIDUAL TESTS")
    for case in INDIVIDUAL_CASES:
        print(f"{case.case_id:5} [{case.category:24}] {case.prompt}")
    print("\nCONVERSATION SCENARIOS")
    for scenario in CONVERSATIONS:
        print(f"{scenario.scenario_id:5} [{scenario.category:24}] {scenario.title} ({len(scenario.steps)} turns)")
        for i, step in enumerate(scenario.steps, 1):
            print(f"      {i:02}. {step.prompt}")


def write_reports(results: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"nexus_test_report_{stamp}.json"
    csv_path = output_dir / f"nexus_test_report_{stamp}.csv"
    manual_path = output_dir / f"nexus_manual_review_{stamp}.csv"

    summary = calculate_summary(results)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "suite": args.suite,
        "request_configuration": {
            "message_field": args.message_field,
            "session_field": args.session_field,
            "history_mode": args.history_mode,
            "messages_field": args.messages_field,
            "dialect": args.dialect,
            "sql_validation_enabled": args.validate_sql,
        },
        "summary": summary,
        "interpretation": "Automated pass rate and expected-term coverage are screening metrics. Review responses and schema semantics manually before treating them as true model accuracy.",
        "results": results,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    columns = ["test_id", "suite", "category", "title", "passed", "blocked", "http_status",
               "elapsed_seconds", "expected_terms", "matched_terms", "missing_or_forbidden",
               "sql_parse_ok", "sql_parse_message", "prompt", "sql", "response",
               "manual_score_0_10", "manual_notes"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            converted = dict(row)
            for key in ("expected_terms", "matched_terms", "missing_or_forbidden"):
                converted[key] = " | ".join(map(str, converted.get(key, [])))
            writer.writerow(converted)

    with manual_path.open("w", newline="", encoding="utf-8-sig") as handle:
        manual_columns = ["test_id", "suite", "category", "prompt", "response", "manual_score_0_10", "manual_notes"]
        writer = csv.DictWriter(handle, fieldnames=manual_columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return json_path, csv_path, manual_path


def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in results if not str(r["test_id"]).endswith(tuple(f"SETUP{i}" for i in range(1, 20)))]
    total = len(scored)
    passed = sum(1 for r in scored if r.get("passed"))
    blocked = sum(1 for r in scored if r.get("blocked"))
    expected_count = sum(len(r.get("expected_terms", [])) for r in scored)
    matched_count = sum(len(r.get("matched_terms", [])) for r in scored)
    http_ok = sum(1 for r in scored if 200 <= int(r.get("http_status", 0)) < 300)
    sql_parseable = [r for r in scored if r.get("sql_parse_ok") is not None]
    sql_valid = sum(1 for r in sql_parseable if r.get("sql_parse_ok") is True)

    by_category: Dict[str, Dict[str, Any]] = {}
    for row in scored:
        cat = row["category"]
        stats = by_category.setdefault(cat, {"total": 0, "passed": 0, "expected_terms": 0, "matched_terms": 0})
        stats["total"] += 1
        stats["passed"] += int(bool(row.get("passed")))
        stats["expected_terms"] += len(row.get("expected_terms", []))
        stats["matched_terms"] += len(row.get("matched_terms", []))
    for stats in by_category.values():
        stats["automated_pass_rate_percent"] = round(100 * stats["passed"] / stats["total"], 2) if stats["total"] else 0
        stats["expected_term_coverage_percent"] = round(100 * stats["matched_terms"] / stats["expected_terms"], 2) if stats["expected_terms"] else None

    conv = [r for r in scored if r["suite"] == "conversation"]
    individual = [r for r in scored if r["suite"] == "individual"]
    summary_steps = [r for r in conv if r["category"] == "summary_qa"]
    return {
        "total_scored_turns": total,
        "passed_turns": passed,
        "failed_turns": total - passed,
        "blocked_turns": blocked,
        "automated_pass_rate_percent": round(100 * passed / total, 2) if total else 0,
        "expected_term_coverage_percent": round(100 * matched_count / expected_count, 2) if expected_count else None,
        "http_success_rate_percent": round(100 * http_ok / total, 2) if total else 0,
        "sql_parse_valid_percent": round(100 * sql_valid / len(sql_parseable), 2) if sql_parseable else None,
        "sql_parseable_turns": len(sql_parseable),
        "individual_turns": len(individual),
        "conversation_turns": len(conv),
        "summary_qa_turns": len(summary_steps),
        "by_category": by_category,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DB Nexus LLM individual and multi-turn test cases.")
    parser.add_argument("--endpoint", default=os.getenv("NEXUS_ENDPOINT", "http://127.0.0.1:8000/api/v1/chat"),
                        help="FastAPI chat endpoint (default: %(default)s)")
    parser.add_argument("--suite", choices=["all", "individual", "conversations", "summary"], default="all",
                        help="Which test suite to run")
    parser.add_argument("--message-field", default="message", help="Request field for the current prompt; set empty to omit")
    parser.add_argument("--session-field", default="session_id", help="Request field for session ID; set empty to omit")
    parser.add_argument("--history-mode", choices=["session", "messages", "none"], default="session",
                        help="session: server stores context by session ID; messages: send accumulated messages; none: stateless")
    parser.add_argument("--messages-field", default="messages", help="Request field used when --history-mode messages")
    parser.add_argument("--extra-payload", default="{}", help="Extra JSON fields merged into every request")
    parser.add_argument("--header", action="append", default=[], help="Extra HTTP header, e.g. X-Client: test (repeatable)")
    parser.add_argument("--api-key", default=os.getenv("NEXUS_API_KEY", ""), help="Optional bearer/API key")
    parser.add_argument("--api-key-header", default="Authorization", help="API key header name")
    parser.add_argument("--api-key-prefix", default="Bearer ", help="Prefix before API key; use empty string for raw key")
    parser.add_argument("--timeout", type=float, default=90.0, help="Request timeout in seconds")
    parser.add_argument("--output-dir", default="nexus_test_reports", help="Where JSON/CSV reports are written")
    parser.add_argument("--validate-sql", action="store_true", help="Parse extracted SQL using optional sqlglot package")
    parser.add_argument("--dialect", default="mysql", help="sqlglot dialect, default mysql")
    parser.add_argument("--list", action="store_true", help="List all prompts without calling the API")
    parser.add_argument("--pause", type=float, default=0.0, help="Optional delay between API calls in seconds")
    args = parser.parse_args()

    if args.list:
        print_list()
        print(f"\nIndividual cases: {len(INDIVIDUAL_CASES)}")
        print(f"Conversation scenarios: {len(CONVERSATIONS)}")
        print(f"Conversation turns: {sum(len(s.steps) for s in CONVERSATIONS)}")
        return 0

    if not args.endpoint.startswith(("http://", "https://")):
        print("ERROR: --endpoint must start with http:// or https://", file=sys.stderr)
        return 2

    if args.history_mode == "none":
        print("WARNING: history-mode=none disables server-side session context. Follow-up tests will probably fail unless your API manages context another way.")
    if args.history_mode == "messages":
        print("INFO: sending the accumulated messages array on every turn. Confirm that your API accepts this format.")

    results: List[Dict[str, Any]] = []
    total_calls = 0
    if args.suite in ("all", "individual"):
        total_calls += sum(len(c.setup) + 1 for c in INDIVIDUAL_CASES)
    if args.suite in ("all", "conversations", "summary"):
        selected = CONVERSATIONS if args.suite != "summary" else [s for s in CONVERSATIONS if s.scenario_id == "F11"]
        total_calls += sum(len(s.steps) for s in selected)
    print(f"DB Nexus test run | suite={args.suite} | endpoint={args.endpoint}")
    print(f"Estimated API calls: {total_calls}")
    print("Each conversation scenario will use a separate session ID; turns inside it share that ID.\n")

    if args.suite in ("all", "individual"):
        for index, case in enumerate(INDIVIDUAL_CASES, 1):
            print(f"[{index}/{len(INDIVIDUAL_CASES)}] {case.case_id}: {case.prompt[:95]}", flush=True)
            before = len(results)
            run_one_case(case, args, results)
            newly_scored = [r for r in results[before:] if r.get("test_id") == case.case_id]
            if newly_scored:
                print("   ", "PASS" if newly_scored[-1]["passed"] else "FAIL", flush=True)
            if args.pause:
                time.sleep(args.pause)

    if args.suite in ("all", "conversations", "summary"):
        selected_scenarios = CONVERSATIONS if args.suite != "summary" else [s for s in CONVERSATIONS if s.scenario_id == "F11"]
        for index, scenario in enumerate(selected_scenarios, 1):
            print(f"\n[{index}/{len(selected_scenarios)}] Scenario {scenario.scenario_id}: {scenario.title} ({len(scenario.steps)} turns)")
            start = len(results)
            run_scenario(scenario, args, results)
            scenario_results = results[start:]
            passed = sum(bool(r["passed"]) for r in scenario_results)
            print(f"   {passed}/{len(scenario_results)} turns passed automated checks")
            if args.pause:
                time.sleep(args.pause)

    json_path, csv_path, manual_path = write_reports(results, args)
    summary = calculate_summary(results)
    print("\n" + "=" * 65)
    print("DB NEXUS TEST SUMMARY")
    print("=" * 65)
    print(f"Scored turns:              {summary['total_scored_turns']}")
    print(f"Automated pass rate:       {summary['automated_pass_rate_percent']:.2f}%")
    if summary["expected_term_coverage_percent"] is not None:
        print(f"Expected-term coverage:    {summary['expected_term_coverage_percent']:.2f}%")
    print(f"HTTP success rate:         {summary['http_success_rate_percent']:.2f}%")
    if summary["sql_parse_valid_percent"] is not None:
        print(f"SQL parser validity:       {summary['sql_parse_valid_percent']:.2f}% ({summary['sql_parseable_turns']} parsed turns)")
    else:
        print("SQL parser validity:       Not measured (install sqlglot and use --validate-sql)")
    print("\nCategory breakdown:")
    for category, stats in sorted(summary["by_category"].items()):
        print(f"  {category:25} {stats['passed']:>3}/{stats['total']:<3}  {stats['automated_pass_rate_percent']:>6.2f}%")
    print("\nReports created:")
    print(f"  JSON:         {json_path.resolve()}")
    print(f"  CSV:          {csv_path.resolve()}")
    print(f"  Manual review:{manual_path.resolve()}")
    print("\nNote: Automatic term matching is a screening score, not true semantic accuracy.")
    print("Review the response text, verify expected schema structures, and use a disposable MySQL database for execution testing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
