import os
import sys
import time
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.rag.service import retrieve_erp_schema, load_erp_registry

def run_tests():
    print("=" * 80)
    print("ENTERPRISE ERP RAG KNOWLEDGE BASE - 5-DOMAIN COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    reg = load_erp_registry(force_reload=True)
    print(f"Schema Registry Loaded:")
    print(f"  Total Domains: {len(reg['domains'])}")
    for dname, dinfo in reg['domains'].items():
        tbls = dinfo.get('tables', [])
        print(f"  - {dname}: {len(tbls)} tables")
    print(f"  Total Master Tables: {len(reg['all_tables'])}")
    print(f"  Total Master Columns: {len(reg['all_columns'])}\n")

    test_scenarios = [
        {
            "category": "Full Domain Generation (21 Tables Each)",
            "tests": [
                ("Finance Domain (21 tables)", "create complete database schema for ERP FINANCE", 21, "erp_finance"),
                ("Supply Chain Domain (21 tables)", "generate complete supply chain database schema", 21, "erp_supply_chain"),
                ("HRMS Domain (21 tables)", "build complete schema for HRMS human resources domain", 21, "erp_hrms"),
                ("E-Commerce Domain (21 tables)", "create complete database schema for ecommerce", 21, "erp_ecommerce"),
                ("Healthcare Domain (21 tables)", "build complete database schema for healthcare domain", 21, "erp_healthcare"),
            ]
        },
        {
            "category": "Specific Tables & Referential Hierarchy (Topological Order)",
            "tests": [
                ("Multi-parent FK Expansion", "Show inventory_stocks schema", 3, "erp_supply_chain"),
                ("Direct Dependency Ordering", "Show hrms_attendance schema", 2, "erp_hrms"),
                ("Customer Invoice with Parent", "Retrieve customer_invoices and its actual direct dependency", 2, "erp_finance"),
                ("Explicit Exclusion of Dependencies", "Retrieve customer_invoices without dependencies", 1, "erp_finance"),
            ]
        },
        {
            "category": "Granular Column Scope & Projection (No PK Leakage)",
            "tests": [
                ("Single Column Isolation", "Retrieve only customer_id from customer_invoices", 1, "erp_finance"),
                ("Multiple Columns without ID", "Retrieve employee_id, work_date, check_in, and check_out from hrms_attendance", 1, "erp_hrms"),
                ("Natural Language Column Mapping", "Retrieve the employee reference, attendance date, check-in time, and check-out time from hrms_attendance", 1, "erp_hrms"),
                ("Dot-Notation Column Retrieval", "Retrieve customer_invoices.customer_id", 1, "erp_finance"),
            ]
        },
        {
            "category": "Zero-Hallucination & Anti-Hallucination Boundary Checks",
            "tests": [
                ("Nonexistent Table Rejection", "Search for employee_benefits", 0, "not_found"),
                ("Nonexistent Column Rejection", "Retrieve salary_grade from hrms_employees", 0, "rejected"),
                ("Mixed Valid & Invalid Tables (2 in KB + 2 Not)", "create customer_invoices, finance_customers, user_tokens, and analytics_events", 2, "erp_finance"),
                ("Mixed Valid & Invalid Columns in Table", "Retrieve employee_code, full_name, salary_grade, department_id from hrms_employees", 1, "erp_hrms"),
            ]
        },
        {
            "category": "Query Variations, Misspellings & Typo Tolerance",
            "tests": [
                ("Enhanced Phrasing via Gateway", "create database schema for ERP finance", 21, "erp_finance"),
                ("Case Insensitive Search", "Retrieve HRMS_Employees", 1, "erp_hrms"),
                ("Singular Near-Miss Clarification", "Retrieve the employees table.", 0, "needs_clarification"),
                ("Leading FROM Clause Variation", "From customer_invoices, retrieve only customer_id", 1, "erp_finance"),
            ]
        },
        {
            "category": "Multi-turn State Continuity & Active Summary Ledger",
            "tests": [
                ("Turn 1: Standalone COA", "create chart_of_accounts table", 1, "erp_finance"),
                ("Turn 2: Customers referencing active COA", "create finance_customers table", 1, "erp_finance"),
                ("Turn 3: Invoices referencing active Customers", "create customer_invoices table", 1, "erp_finance"),
            ]
        }
    ]

    total_executed = 0
    total_passed = 0
    active_summary = ""

    results_table = []

    for group in test_scenarios:
        print(f"\n--- Category: {group['category']} ---")
        for test in group["tests"]:
            name = test[0]
            query = test[1]
            exp_tbl_count = test[2]
            exp_domain_or_status = test[3]

            summary_input = ""
            if name.startswith("Turn 2"):
                summary_input = active_summary
            elif name.startswith("Turn 3"):
                summary_input = active_summary

            t0 = time.time()
            res = retrieve_erp_schema(query, summary=summary_input)
            latency_ms = (time.time() - t0) * 1000

            total_executed += 1

            # Checks
            tbl_count = len(res.tables)
            passed = True
            reasons = []

            if name == "Nonexistent Table Rejection":
                if res.status != "rejected" or tbl_count != 0 or res.sql != "":
                    passed = False
                    reasons.append(f"Expected rejected with 0 tables, got status={res.status}, count={tbl_count}")
            elif name == "Nonexistent Column Rejection":
                if res.status != "rejected" or tbl_count != 0:
                    passed = False
                    reasons.append(f"Expected rejected, got status={res.status}")
            elif name == "Singular Near-Miss Clarification":
                if res.status != "needs_clarification" or "hrms_employees" not in res.suggestions:
                    passed = False
                    reasons.append(f"Expected clarification with suggestions, got status={res.status}")
            elif name == "Mixed Valid & Invalid Tables (2 in KB + 2 Not)":
                if tbl_count != 2 or not any("user_tokens" in w for w in res.warnings):
                    passed = False
                    reasons.append(f"Expected 2 tables and ungrounded warnings, got count={tbl_count}")
            elif name == "Mixed Valid & Invalid Columns in Table":
                if res.status != "partial" or "salary_grade" not in res.explanation:
                    passed = False
                    reasons.append(f"Expected partial with salary_grade warning, got status={res.status}")
            else:
                if tbl_count < exp_tbl_count:
                    passed = False
                    reasons.append(f"Expected >={exp_tbl_count} tables, got {tbl_count}")

            # SQL and Explanation check
            has_sql = bool(res.sql.strip()) or res.status in ("rejected", "needs_clarification")
            has_expl = bool(res.explanation.strip())
            has_sum = bool(res.summary.strip()) or res.status == "needs_clarification"

            if not has_expl:
                passed = False
                reasons.append("Missing explanation")

            if passed:
                total_passed += 1
                verdict = "PASS [OK]"
            else:
                verdict = f"FAIL: {'; '.join(reasons)}"

            if name.startswith("Turn 1") or name.startswith("Turn 2"):
                active_summary = res.summary

            print(f"[{verdict[:4]}] {name}")
            print(f"       Query: \"{query}\"")
            print(f"       Status: {res.status} | Tables Found: {tbl_count} | Latency: {latency_ms:.1f}ms")
            if res.warnings:
                print(f"       Warnings: {res.warnings}")

            results_table.append({
                "category": group["category"],
                "name": name,
                "query": query,
                "status": res.status,
                "tables_count": tbl_count,
                "latency_ms": round(latency_ms, 1),
                "has_sql": bool(res.sql.strip()),
                "has_explanation": bool(res.explanation.strip()),
                "has_summary": bool(res.summary.strip()),
                "verdict": "PASS" if passed else "FAIL"
            })

    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY: {total_passed} / {total_executed} SCENARIOS PASSED ({(total_passed/total_executed)*100:.1f}%)")
    print("=" * 80)

    # Save benchmark json artifact
    out_file = BASE_DIR / "scripts" / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"Results JSON written to {out_file}")

if __name__ == "__main__":
    run_tests()
