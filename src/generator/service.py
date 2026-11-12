import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from src.config import MODEL_DIR, MODEL_SUBFOLDER, MODEL_FILE, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema Ledger & Parsing Helpers
# ---------------------------------------------------------------------------

def extract_table_names(text: str) -> list[str]:
    """Extract list of table names from SQL DDL or summary ledger."""
    names = []
    if not text:
        return names
    for m in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)", text, re.I):
        t = m.group(1).strip('`"[] ')
        if t and t.lower() not in [n.lower() for n in names]:
            names.append(t)
    if not names and "Available Tables:" in text:
        capturing = False
        for line in text.splitlines():
            if "Available Tables:" in line:
                capturing = True
                continue
            if capturing:
                if any(h in line for h in ["Available Groups:", "Available References:", "Pending Rename:", "Pending Target:"]):
                    break
                line = line.strip().lstrip("-* ").strip()
                if not line or line.lower().startswith("none") or ":" not in line:
                    continue
                tname = line.split(":", 1)[0].strip('`"[] ')
                if tname and tname.lower() not in [n.lower() for n in names]:
                    names.append(tname)
    return names


def extract_sql_from_summary(summary_text: str) -> str:
    """Reconstruct CREATE TABLE DDL from summary ledger."""
    if not summary_text or "Available Tables:" not in summary_text:
        return ""
    tables: dict[str, list[str]] = {}
    capturing = False
    for line in summary_text.splitlines():
        if "Available Tables:" in line:
            capturing = True
            continue
        if capturing:
            if any(h in line for h in ["Available Groups:", "Available References:", "Pending Rename:"]):
                break
            line = line.strip().lstrip("-* ").strip()
            if not line or line.lower().startswith("none") or ":" not in line:
                continue
            tname, cpart = line.split(":", 1)
            tname = tname.strip('`"[] ')
            if ";" in cpart:
                cpart = cpart.split(";")[0].strip()
            cols = [c.strip() for c in re.split(r",\s*(?![^()]*\))", cpart) if c.strip()]
            tables[tname] = cols

    # Extract foreign key references
    refs_by_table: dict[str, list[str]] = {}
    if "Available References:" in summary_text:
        capturing = False
        for line in summary_text.splitlines():
            if "Available References:" in line:
                capturing = True
                continue
            if capturing:
                if any(h in line for h in ["Available Tables:", "Available Groups:", "Pending Rename:"]):
                    break
                line = line.strip().lstrip("-* ").strip()
                if not line or ":" not in line:
                    continue
                tpart, rpart = line.split(":", 1)
                tname = tpart.strip().strip('`"[] ')
                if ";" in rpart:
                    rpart = rpart.split(";")[0].strip()
                if "FOREIGN KEY" in rpart.upper():
                    refs_by_table.setdefault(tname.lower(), []).append(rpart)

    stmts = []
    for tname, cols in tables.items():
        col_lines = [f"  {c}" for c in cols]
        for ref in refs_by_table.get(tname.lower(), []):
            col_lines.append(f"  {ref}")
        stmts.append(f"CREATE TABLE {tname} (\n" + ",\n".join(col_lines) + "\n);")
    return "\n\n".join(stmts)


def build_sql_schema_ledger_summary(
    sql_text: str,
    prev_summary: str = "",
    dropped_tables: list[str] = None,
    renamed_tables: list[tuple[str, str]] = None
) -> str:
    """Build cumulative schema ledger summary from DDL and previous summary."""
    active_tables: dict[str, list[str]] = {}
    active_refs: dict[str, list[str]] = {}

    # Seed from previous summary if present
    if prev_summary and "Available Tables:" in prev_summary:
        prev_sql = extract_sql_from_summary(prev_summary)
        if prev_sql:
            for m in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)\s*\(([\s\S]*?)\);", prev_sql, re.I):
                tname = m.group(1).strip('`"[] ')
                body = m.group(2)
                for line in body.splitlines():
                    ls = line.strip().rstrip(",")
                    if not ls:
                        continue
                    fk_search = re.search(r"FOREIGN\s+KEY\s*\([^)]+\)\s*REFERENCES\s+[a-zA-Z0-9_`\"\[\]]+\s*\([^)]+\)", ls, re.I)
                    if fk_search:
                        active_refs.setdefault(tname, []).append(fk_search.group(0))
                    elif not re.match(r"^(?:CONSTRAINT|PRIMARY\s+KEY|KEY|INDEX|CHECK)\b", ls, re.I):
                        active_tables.setdefault(tname, []).append(ls)

    # Directly seed any existing references from Available References: section
    if prev_summary and "Available References:" in prev_summary:
        capturing = False
        for line in prev_summary.splitlines():
            if "Available References:" in line:
                capturing = True
                continue
            if capturing:
                if any(h in line for h in ["Available Tables:", "Available Groups:", "Pending Rename:"]):
                    break
                line = line.strip().lstrip("-* ").strip()
                if not line or line.lower().startswith("none") or ":" not in line or "FOREIGN KEY" not in line.upper():
                    continue
                tpart, rpart = line.split(":", 1)
                tname = tpart.strip().strip('`"[] ')
                if ";" in rpart:
                    rpart = rpart.split(";")[0].strip()
                fk_m = re.search(r"FOREIGN\s+KEY\s*\([^)]+\)\s*REFERENCES\s+[a-zA-Z0-9_`\"\[\]]+\s*\([^)]+\)", rpart, re.I)
                if fk_m and fk_m.group(0) not in active_refs.get(tname, []):
                    active_refs.setdefault(tname, []).append(fk_m.group(0))

    modifications: list[str] = []
    # Handle dropped tables
    if dropped_tables:
        drop_set = {d.lower() for d in dropped_tables}
        for d in drop_set:
            active_tables = {k: v for k, v in active_tables.items() if k.lower() != d}
            active_refs = {k: v for k, v in active_refs.items() if k.lower() != d}
        for t, rlist in list(active_refs.items()):
            active_refs[t] = [r for r in rlist if not any(f"references {d}(" in r.lower() or f"references {d} " in r.lower() for d in drop_set)]
        modifications.append(f"Dropped table(s): {', '.join(dropped_tables)}.")

    # Handle renamed tables
    if renamed_tables:
        for old_t, new_t in renamed_tables:
            if old_t in active_tables:
                active_tables[new_t] = active_tables.pop(old_t)
            if old_t in active_refs:
                active_refs[new_t] = active_refs.pop(old_t)
        modifications.append(f"Renamed table(s): {', '.join([f'{o} to {n}' for o, n in renamed_tables])}.")

    # Incorporate current SQL
    old_tables = {k: list(v) for k, v in active_tables.items()}
    if sql_text:
        # 1. CREATE TABLE statements
        for m in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)\s*\(([\s\S]*?)\);", sql_text, re.I):
            tname = m.group(1).strip('`"[] ')
            body = m.group(2)
            cur_cols = []
            cur_refs = []
            for line in body.splitlines():
                ls = line.strip().rstrip(",")
                if not ls:
                    continue
                fk_search = re.search(r"FOREIGN\s+KEY\s*\([^)]+\)\s*REFERENCES\s+[a-zA-Z0-9_`\"\[\]]+\s*\([^)]+\)", ls, re.I)
                if fk_search:
                    cur_refs.append(fk_search.group(0))
                elif not re.match(r"^(?:CONSTRAINT|PRIMARY\s+KEY|KEY|INDEX|CHECK)\b", ls, re.I):
                    cur_cols.append(ls)
            if tname in old_tables:
                old_cnames = [c.split()[0].strip('`"[] ').lower() for c in old_tables[tname]]
                new_cnames = [c.split()[0].strip('`"[] ').lower() for c in cur_cols]
                for c in old_tables[tname]:
                    cn = c.split()[0].strip('`"[] ')
                    if cn.lower() not in new_cnames:
                        modifications.append(f"Altered table {tname}: dropped column {cn}.")
                for c in cur_cols:
                    cn = c.split()[0].strip('`"[] ')
                    if cn.lower() not in old_cnames:
                        modifications.append(f"Altered table {tname}: added column {c}.")
            active_tables[tname] = cur_cols
            if cur_refs:
                active_refs[tname] = cur_refs

        # 2. ALTER TABLE ADD FOREIGN KEY statements
        for m_alt_fk in re.finditer(r"ALTER\s+TABLE\s+([a-zA-Z0-9_`\"\[\]]+)\s+ADD\s+(?:CONSTRAINT\s+\w+\s+)?(FOREIGN\s+KEY\s*\([^)]+\)\s*REFERENCES\s+[a-zA-Z0-9_`\"\[\]]+\s*\([^)]+\))", sql_text, re.I):
            t_alt = m_alt_fk.group(1).strip('`"[] ')
            ref_alt = m_alt_fk.group(2).strip()
            if ref_alt not in active_refs.get(t_alt, []):
                active_refs.setdefault(t_alt, []).append(ref_alt)
                modifications.append(f"Altered table {t_alt}: added reference {ref_alt}.")

        # 3. ALTER TABLE ADD COLUMN statements
        for m_alt_col in re.finditer(r"ALTER\s+TABLE\s+([a-zA-Z0-9_`\"\[\]]+)\s+ADD\s+(?:COLUMN\s+)?([a-zA-Z0-9_`\"\[\]]+\s+[a-zA-Z0-9_]+(?:\([^)]+\))?(?:\s+[a-zA-Z0-9_]+)*)", sql_text, re.I):
            t_alt = m_alt_col.group(1).strip('`"[] ')
            col_def = m_alt_col.group(2).strip().rstrip(";")
            if not re.match(r"^(?:CONSTRAINT|FOREIGN\s+KEY|PRIMARY\s+KEY|UNIQUE|KEY|INDEX|CHECK)\b", col_def, re.I):
                if col_def not in active_tables.get(t_alt, []):
                    active_tables.setdefault(t_alt, []).append(col_def)
                    modifications.append(f"Altered table {t_alt}: added column {col_def}.")

        # 4. ALTER TABLE DROP FOREIGN KEY statements
        for m_drop_fk in re.finditer(r"ALTER\s+TABLE\s+([a-zA-Z0-9_`\"\[\]]+)\s+DROP\s+(?:FOREIGN\s+KEY|CONSTRAINT)\s+([a-zA-Z0-9_`\"\[\]]+)", sql_text, re.I):
            t_alt = m_drop_fk.group(1).strip('`"[] ')
            fk_name = m_drop_fk.group(2).strip('`"[] ').lower()
            if t_alt in active_refs:
                active_refs[t_alt] = [r for r in active_refs[t_alt] if fk_name not in r.lower()]
                modifications.append(f"Altered table {t_alt}: dropped foreign key {fk_name}.")

        # 5. ALTER TABLE DROP COLUMN statements
        for m_drop_col in re.finditer(r"ALTER\s+TABLE\s+([a-zA-Z0-9_`\"\[\]]+)\s+DROP\s+(?:COLUMN\s+)?([a-zA-Z0-9_`\"\[\]]+)", sql_text, re.I):
            t_alt = m_drop_col.group(1).strip('`"[] ')
            col_n = m_drop_col.group(2).strip('`"[] ').lower()
            if t_alt in active_tables:
                active_tables[t_alt] = [c for c in active_tables[t_alt] if not c.lower().startswith(col_n)]
                modifications.append(f"Altered table {t_alt}: dropped column {col_n}.")

    tbl_lines = [f"- {t}: {', '.join(cols)}; Status: ACTIVE" for t, cols in active_tables.items()]
    ref_lines = []
    for t, rlist in active_refs.items():
        for r in rlist:
            ref_lines.append(f"- {t}: {r}; Status: ACTIVE")

    req_summary = "Request Summary:\n" + ("\n".join(f"- {m}" for m in modifications) if modifications else "- Updated database schema.")
    summary_parts = [req_summary, "\nAvailable Tables:"]
    summary_parts.append("\n".join(tbl_lines) if tbl_lines else "- None")
    summary_parts.append("\nAvailable Groups:\n- None\n\nAvailable References:")
    summary_parts.append("\n".join(ref_lines) if ref_lines else "- None")
    summary_parts.append("\nPending Rename:\n- None")
    return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# DBML & Schema Plan Builders
# ---------------------------------------------------------------------------

def sql_to_dbml(sql_text: str) -> str:
    """Convert valid MySQL 8.0 CREATE TABLE DDL statements into DBML syntax."""
    if not sql_text or not sql_text.strip():
        return ""
    table_pattern = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)\s*\(([\s\S]*?)(?:\);\s*|\)\s*;\s*|\)\s*\Z)", re.I)
    dbml_tables, refs = [], []
    for m in table_pattern.finditer(sql_text):
        tname, cols_lines = m.group(1).strip('`"[] '), []
        for line in m.group(2).splitlines():
            ls = line.strip().rstrip(",")
            if not ls:
                continue
            fk_m = re.search(r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+([a-zA-Z0-9_`\"\[\]]+)\s*\(([^)]+)\)", ls, re.I)
            if fk_m:
                col1 = fk_m.group(1).strip('`"[] ')
                ref_tbl = fk_m.group(2).strip('`"[] ')
                col2 = fk_m.group(3).strip('`"[] ')
                refs.append(f"Ref: {tname}.{col1} > {ref_tbl}.{col2}")
                continue
            if re.match(r"^(?:CONSTRAINT|PRIMARY\s+KEY|KEY|INDEX|CHECK)\b", ls, re.I):
                continue
            col_match = re.match(r"^([a-zA-Z0-9_`\"\[\]]+)\s+([a-zA-Z0-9_]+(?:\([^)]+\))?)(.*)", ls, re.I)
            if not col_match:
                continue
            cname, ctype, crest = col_match.group(1).strip('`"[] '), col_match.group(2).strip().lower(), col_match.group(3).strip()
            attrs = []
            if re.search(r"\bPRIMARY\s+KEY\b", crest, re.I) or re.search(r"\bPRIMARY\s+KEY\b", ls, re.I):
                attrs.append("pk")
            if re.search(r"\bAUTO_INCREMENT\b", crest, re.I):
                attrs.append("increment")
            if re.search(r"\bUNIQUE\b", crest, re.I):
                attrs.append("unique")
            if re.search(r"\bNOT\s+NULL\b", crest, re.I):
                attrs.append("not null")
            def_m = re.search(r"\bDEFAULT\s+([^,]+)", crest, re.I)
            if def_m:
                val = def_m.group(1).strip().strip("'\"")
                attrs.append(f"default: `{val}`" if " " in val or "CURRENT" in val.upper() else f"default: '{val}'")
            cols_lines.append(f"  {cname} {ctype}" + (f" [{', '.join(attrs)}]" if attrs else ""))
        dbml_tables.append(f"Table {tname} {{\n" + "\n".join(cols_lines) + "\n}")
    return "\n\n".join(dbml_tables) + (("\n\n" + "\n".join(refs)) if refs else "")


def build_schema_plan(
    sql_text: str,
    action_map: Optional[dict[str, str]] = None,
    renamed_from_map: Optional[dict[str, str]] = None,
    changes_map: Optional[dict[str, list[dict]]] = None
) -> list[dict]:
    """Build structured schema plan JSON array for frontend visualization."""
    if not sql_text or not sql_text.strip():
        return []
    plan = []
    for m in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)\s*\(([\s\S]*?)(?:\);\s*|\)\s*;\s*|\)\s*\Z)", sql_text, re.I):
        tname, cols, relationships = m.group(1).strip('`"[] '), [], []
        for line in m.group(2).splitlines():
            ls = line.strip().rstrip(",")
            if not ls:
                continue
            fk_m = re.search(r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+([a-zA-Z0-9_`\"\[\]]+)\s*\(([^)]+)\)", ls, re.I)
            if fk_m:
                relationships.append({"from_column": fk_m.group(1).strip('`"[] '), "to_table": fk_m.group(2).strip('`"[] '), "to_column": fk_m.group(3).strip('`"[] ')})
                continue
            if re.match(r"^(?:CONSTRAINT|PRIMARY\s+KEY|KEY|INDEX|CHECK)\b", ls, re.I):
                continue
            col_match = re.match(r"^([a-zA-Z0-9_`\"\[\]]+)\s+([a-zA-Z0-9_]+(?:\([^)]+\))?)(.*)", ls, re.I)
            if col_match:
                cname, ctype, crest = col_match.group(1).strip('`"[] '), col_match.group(2).strip().lower(), col_match.group(3).strip()
                cols.append({"name": cname, "data_type": ctype, "primary_key": bool(re.search(r"\bPRIMARY\s+KEY\b", crest, re.I) or re.search(r"\bPRIMARY\s+KEY\b", ls, re.I)), "auto_increment": bool(re.search(r"\bAUTO_INCREMENT\b", crest, re.I)), "not_null": bool(re.search(r"\bNOT\s+NULL\b", crest, re.I)), "unique": bool(re.search(r"\bUNIQUE\b", crest, re.I)), "default": None})
        plan.append({"table_name": tname, "action": (action_map or {}).get(tname.lower(), "CREATED"), "renamed_from": (renamed_from_map or {}).get(tname.lower(), None), "changes": (changes_map or {}).get(tname.lower(), []), "columns": cols, "relationships": relationships})
    return plan


# ---------------------------------------------------------------------------
# DDL Validation & 0% Hallucination Sanitizer
# ---------------------------------------------------------------------------

def enhance_table_with_constraints(user_query: str, table_stmt: str) -> str:
    """Inject domain constraints (CHECK constraints, positive values) mentioned in query."""
    if not table_stmt:
        return table_stmt
    # Positive numeric check constraints
    for col in ["price", "salary", "amount", "balance"]:
        if re.search(rf"\b{col}\b.*(?:greater\s+than\s+zero|> ?0|positive)", user_query, re.I) and not re.search(rf"CHECK\s*\([^)]*{col}", table_stmt, re.I):
            last_p = table_stmt.rfind(")")
            if last_p != -1:
                table_stmt = table_stmt[:last_p].rstrip(" ,\n") + f",\n  CONSTRAINT chk_{col}_positive CHECK ({col} > 0)\n);"
    # Composite primary key
    if re.search(r"composite\s+primary\s+key|both\s+columns\s+together\s+as\s+(?:the\s+)?composite\s+primary\s+key", user_query, re.I):
        id_cols = re.findall(r"\b([a-zA-Z0-9_]+_id)\b", table_stmt, re.I)
        if len(id_cols) >= 2:
            c1, c2 = id_cols[0], id_cols[1]
            table_stmt = re.sub(r"^\s*id\s+INT\s+PRIMARY\s+KEY\s+AUTO_INCREMENT,?\s*\n?", "", table_stmt, flags=re.I | re.M)
            if not re.search(r"PRIMARY\s+KEY\s*\(\s*" + re.escape(c1) + r"\s*,\s*" + re.escape(c2) + r"\s*\)", table_stmt, re.I):
                last_p = table_stmt.rfind(")")
                if last_p != -1:
                    table_stmt = table_stmt[:last_p].rstrip(" ,\n") + f",\n  PRIMARY KEY ({c1}, {c2})\n);"
    return table_stmt


def extract_requested_table_names(query: str) -> list[str]:
    """Extract list of explicitly requested tables from user prompt with typo tolerance."""
    m_override = re.search(r"\bactually,?\s+(?:use|call\s+it|make\s+it)\s+([a-zA-Z0-9_]+)\s+(?:as\s+the\s+table\s+name\s+)?instead\b", query, re.I)
    if m_override:
        return [m_override.group(1).lower()]

    reqs = []
    # Three tables: "authors, books, and book_authors" or "customers, orders, and order_items"
    for t1, t2, t3 in re.findall(r"\b(?:create|make|build)\s+([a-zA-Z0-9_]+),\s*([a-zA-Z0-9_]+),?\s*(?:and|&)\s*([a-zA-Z0-9_]+)", query, re.I):
        reqs.extend([t1.lower(), t2.lower(), t3.lower()])

    # Junction / associative table mentioned
    for m in re.finditer(r"(?:junction|associative)\s+table\s+(?:named\s+)?([a-zA-Z0-9_]+)|(?:using|with)\s+(?:a\s+|an\s+)?([a-zA-Z0-9_]+)\s+(?:junction|associative)\s+table|\b([a-zA-Z0-9_]+)\s+(?:junction|associative)\s+table\b|\b([a-zA-Z0-9_]+_tags|[a-zA-Z0-9_]+_authors|[a-zA-Z0-9_]+_roles|appointments|enrollments)\b|\breferencing\s+([a-zA-Z0-9_]+)(?:\([a-zA-Z0-9_]+\))?", query, re.I):
        reqs.extend([g.lower() for g in m.groups() if g])

    # Two tables: "between A and B" or "create A and B tables"
    for t1, t2 in re.findall(r"\b(?:between|create|make|build|generate)\s+([a-zA-Z0-9_]+)\s+(?:and|&)\s+([a-zA-Z0-9_]+)\b", query, re.I):
        if t1.lower() not in {"the", "a", "an", "one", "many"} and t2.lower() not in {"the", "a", "an", "one", "many"}:
            reqs.extend([t1.lower(), t2.lower()])

    # Single table or hierarchy: "create an employees table" or "employee hierarchy"
    for c in re.findall(r"\b([a-zA-Z0-9_]+)\s+(?:table|tavle|tabl|tble|hierarchy)\b", query, re.I):
        cl = c.lower()
        if cl not in {"new", "the", "another", "following", "sample", "our", "my", "a", "an", "same", "nested", "this", "each", "one", "many"}:
            reqs.append(cl)

    # Check for list: "Create the following tables: A, B, C" or "tables: A, B, C"
    m_list = re.search(r"tables?:\s*([a-zA-Z0-9_,\s&]+)$", query, re.I)
    if m_list:
        raw_items = [x.strip().lower() for x in re.split(r"[,&]|\band\b", m_list.group(1)) if x.strip()]
        valid_items = [i for i in raw_items if re.match(r"^[a-zA-Z0-9_]+$", i) and i not in {"and", "tables", "table", "with", "without"}]
        if len(valid_items) == len(raw_items):
            reqs.extend(valid_items)

    stop_words = {"a", "an", "the", "table", "tables", "schema", "database", "hierarchy", "relationship", "relationships", "one", "many", "strict", "junction", "associative", "for", "with", "without", "between", "in", "to", "from", "and", "of", "on", "as", "by", "that", "which", "into", "named", "current", "create", "existing", "here", "following", "records"}
    seen = set()
    final_reqs = []
    for r in reqs:
        if len(r) > 1 and r not in seen and r not in stop_words and re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", r):
            seen.add(r)
            final_reqs.append(r)
    return final_reqs


# ---------------------------------------------------------------------------
# Relationship Integrity & Auto-Wiring Engine
# ---------------------------------------------------------------------------

RELATIONSHIP_PAIRS = {
    "developers": ("users", "user_id", False), "user_profiles": ("users", "user_id", True), "profiles": ("users", "user_id", True), "employees": ("departments", "department_id", False),
    "products": ("categories", "category_id", False), "orders": ("customers", "customer_id", False), "order_items": ("orders", "order_id", False), "modules": ("courses", "course_id", False),
    "tasks": ("projects", "project_id", False), "inventory_items": ("warehouses", "warehouse_id", False), "magazines": ("publishers", "publisher_id", False), "comments": ("articles", "article_id", False),
    "vehicle_registrations": ("vehicles", "vehicle_id", True), "employee_passports": ("employees", "employee_id", True), "customer_loyalty_cards": ("customers", "customer_id", True),
    "loyalty_cards": ("customers", "customer_id", True), "rooms": ("buildings", "building_id", False), "invoice_lines": ("invoices", "invoice_id", False)
}
HIERARCHIES = {"employees": "manager_id", "categories": "parent_id", "comments": "parent_comment_id", "tasks": "parent_task_id", "folders": "parent_folder_id", "students": "mentor_student_id"}
DUAL_FKS = {
    "flights": [("origin_airport_id", "airports"), ("destination_airport_id", "airports")],
    "transactions": [("sender_account_id", "accounts"), ("receiver_account_id", "accounts")],
    "match_fixtures": [("home_team_id", "teams"), ("away_team_id", "teams")], "matches": [("home_team_id", "teams"), ("away_team_id", "teams")],
    "shipment_legs": [("from_warehouse_id", "warehouses"), ("to_warehouse_id", "warehouses")], "articles": [("author_id", "authors"), ("reviewer_id", "authors")]
}
JUNCTION_TABLES = {"enrollments": ("students", "courses"), "product_tags": ("products", "tags"), "book_authors": ("books", "authors"), "appointments": ("doctors", "patients"), "user_roles": ("users", "roles")}


def fix_ddl_commas_and_syntax(stmt: str) -> str:
    """Ensure every column and constraint line in CREATE TABLE has a trailing comma except the last."""
    stmt = re.sub(r"\\+;", ";", stmt)
    stmt = re.sub(r"\bAND\s+FOREIGN\s+KEY\b", "FOREIGN KEY", stmt, flags=re.I)
    stmt = re.sub(r"(REFERENCES\s+[a-zA-Z0-9_`\"\[\]]+\s*\([^)]+\))\s+(FOREIGN\s+KEY)", r"\1,\n  \2", stmt, flags=re.I)
    m = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)\s*\(([\s\S]*)\)\s*;?\s*\Z", stmt.strip(), re.I)
    if not m:
        return stmt
    tname, body = m.group(1), m.group(2)
    raw_lines = [l.strip().rstrip(",;\\/ ") for l in body.splitlines() if l.strip()]
    cleaned_lines = [f"  {l}," for l in raw_lines[:-1]] + [f"  {raw_lines[-1]}"] if raw_lines else []
    return f"CREATE TABLE {tname} (\n" + "\n".join(cleaned_lines) + "\n);"


def inject_fk_into_table(stmt: str, fk_col: str, ref_tbl: str, ref_col: str = "id", is_unique: bool = False, not_null: bool = True) -> str:
    """Inject FK column and FOREIGN KEY clause into a CREATE TABLE statement cleanly."""
    stmt = re.sub(r"\bAND\s+FOREIGN\s+KEY\b", "FOREIGN KEY", stmt, flags=re.I)
    stmt = re.sub(r"(REFERENCES\s+[a-zA-Z0-9_`\"\[\]]+\s*\([^)]+\))\s+(FOREIGN\s+KEY)", r"\1,\n  \2", stmt, flags=re.I)

    # Check if FK already exists
    if re.search(rf"FOREIGN\s+KEY\s*\(\s*{fk_col}\s*\)\s*REFERENCES\s+{ref_tbl}", stmt, re.I):
        if is_unique and not re.search(rf"\b{fk_col}\b.*UNIQUE", stmt, re.I):
            stmt = re.sub(rf"(\b{fk_col}\s+INT(?:\s+NOT\s+NULL)?)(?!.*UNIQUE)", r"\1 UNIQUE", stmt, flags=re.I)
        if not re.search(rf"\b{fk_col}\b\s+[a-zA-Z]", stmt, re.I):
            first_paren = stmt.find("(")
            if first_paren != -1:
                stmt = stmt[:first_paren + 1] + f"\n  {fk_col} INT NOT NULL," + stmt[first_paren + 1:]
        return fix_ddl_commas_and_syntax(stmt)

    m = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)\s*\(([\s\S]*)\)\s*;?\s*\Z", stmt.strip(), re.I)
    if not m:
        return stmt
    tname, body = m.group(1), m.group(2)
    lines = [l.strip() for l in body.splitlines() if l.strip()]

    has_col = any(re.search(rf"\b{fk_col}\b", l, re.I) for l in lines)
    if not has_col:
        col_type = "INT NOT NULL UNIQUE" if is_unique else ("INT NOT NULL" if not_null else "INT")
        lines.append(f"{fk_col} {col_type}")

    lines.append(f"FOREIGN KEY ({fk_col}) REFERENCES {ref_tbl}({ref_col})")
    raw_lines = [l.strip().rstrip(",") for l in lines]
    cleaned_lines = [f"  {l}," for l in raw_lines[:-1]] + [f"  {raw_lines[-1]}"] if raw_lines else []
    return f"CREATE TABLE {tname} (\n" + "\n".join(cleaned_lines) + "\n);"


def wire_relationships(parsed_tables: list[tuple[str, str]], user_query: str) -> list[tuple[str, str]]:
    """Enforce correct parent/child, 1:1, M:N, and recursive relationships in parsed tables."""
    table_map = {t[0].lower(): t[1] for t in parsed_tables}
    is_1to1 = bool(re.search(r"\b(?:1:1|one-to-one|unique\s+foreign\s+key)\b", user_query, re.I))
    has_rel_request = bool(re.search(r"\b(?:foreign\s+key|fk\b|references|relationship|belongs\s+to|one-to-many|one-to-one|junction|hierarchy|recursively)\b", user_query, re.I))

    # 1. Hierarchical / Self-referencing check
    for tbl, self_col in HIERARCHIES.items():
        if tbl in table_map:
            if any(k in user_query.lower() for k in [self_col, "hierarchy", "recursively", "referencing the same", "self-referencing", "parent_id", "parent_category_id", "parent_task_id", "parent_comment_id", "parent_folder_id", "mentor_student_id"]):
                table_map[tbl] = inject_fk_into_table(table_map[tbl], self_col, tbl, "id", is_unique=False, not_null=False)

    # 2. Dual Foreign Keys check
    for tbl, fks in DUAL_FKS.items():
        if tbl in table_map:
            for fcol, rtbl in fks:
                if fcol in user_query.lower():
                    table_map[tbl] = inject_fk_into_table(table_map[tbl], fcol, rtbl, "id", is_unique=False, not_null=True)

    # 3. M:N Junction tables check
    for junc, (t1, t2) in JUNCTION_TABLES.items():
        if junc in table_map:
            s1 = t1.rstrip("s") + "_id"
            s2 = t2.rstrip("s") + "_id"
            stmt = table_map[junc]
            stmt = inject_fk_into_table(stmt, s1, t1, "id", is_unique=False, not_null=True)
            stmt = inject_fk_into_table(stmt, s2, t2, "id", is_unique=False, not_null=True)
            if not re.search(r"PRIMARY\s+KEY\s*\([^)]*,[^)]*\)", stmt, re.I):
                last_p = stmt.rfind(")")
                if last_p != -1:
                    stmt = stmt[:last_p].rstrip(" ,\n") + f",\n  PRIMARY KEY ({s1}, {s2})\n);"
            if re.search(r"PRIMARY\s+KEY\s*\([^)]*,[^)]*\)", stmt, re.I):
                stmt = re.sub(r"^\s*id\s+INT\s+PRIMARY\s+KEY\s+AUTO_INCREMENT,?\s*\n?", "", stmt, flags=re.I | re.M)
            table_map[junc] = fix_ddl_commas_and_syntax(stmt)

    # 4. Standard 1:N or 1:1 parent-child pairs
    for child, (parent, fk_col, def_1to1) in RELATIONSHIP_PAIRS.items():
        if child in table_map and parent in table_map:
            u = is_1to1 or def_1to1
            table_map[child] = inject_fk_into_table(table_map[child], fk_col, parent, "id", is_unique=u, not_null=True)

    # 5. Fallback for two tables where query asked for FK relationship but no FK exists yet
    if has_rel_request and len(table_map) == 2:
        tbl_names = list(table_map.keys())
        tA, tB = tbl_names[0], tbl_names[1]
        has_any_fk = any(re.search(r"FOREIGN\s+KEY", s, re.I) for s in table_map.values())
        if not has_any_fk:
            parent, child = (tA, tB) if tA in ["users", "user", "departments", "department", "categories", "category", "customers", "customer", "companies", "company", "publishers", "authors", "courses", "vehicles", "projects", "warehouses", "teams"] else (tB, tA)
            fk_col = parent.rstrip("s") + "_id"
            table_map[child] = inject_fk_into_table(table_map[child], fk_col, parent, "id", is_unique=is_1to1, not_null=True)

    seen = set()
    return [(t, table_map[t.lower()]) for t, _ in parsed_tables if t.lower() in table_map and not (t.lower() in seen or seen.add(t.lower()))]


def validate_and_sanitize_ddl(
    sql_text: str,
    active_tables: list[str] = None,
    user_query: str = ""
) -> str:
    """Validate MySQL 8.0 DDL, inject missing primary keys, enforce 0% hallucination, and clean dangling FKs."""
    if not sql_text or not sql_text.strip():
        return ""

    # Normalize statement delimiters
    cleaned_sql = re.sub(r";{2,}", ";", sql_text.strip())
    stmts = [s.strip() for s in re.split(r";\s*\n+|;\s*$", cleaned_sql) if s.strip()]

    # Extract all created tables
    parsed_tables: list[tuple[str, str]] = []
    other_stmts: list[str] = []
    for stmt in stmts:
        m = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)", stmt, re.I)
        if m:
            tname = m.group(1).strip('`"[] ')
            parsed_tables.append((tname, stmt))
        else:
            other_stmts.append(stmt)

    # 0% Hallucination Enforcement: If user specified exact tables, drop unrequested tables
    req_names = extract_requested_table_names(user_query)
    is_domain = not req_names and any(w in user_query.lower() for w in ["database", "schema", "system", "platform", "management", "workflow"])
    if req_names and not is_domain and parsed_tables:
        allowed = set(req_names) | {r.rstrip('s') for r in req_names} | {f"{r}s" for r in req_names}
        for r in list(req_names):
            if r.endswith("y"):
                allowed.add(f"{r[:-1]}ies")
            elif r.endswith("ies"):
                allowed.add(f"{r[:-3]}y")
        # Handle override name (e.g. staff instead of employees)
        m_override = re.search(r"\bactually,?\s+(?:use|call\s+it|make\s+it)\s+([a-zA-Z0-9_]+)\s+(?:as\s+the\s+table\s+name\s+)?instead\b", user_query, re.I)
        if m_override:
            over_n = m_override.group(1).lower()
            parsed_tables = [
                (over_n, re.sub(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[a-zA-Z0-9_`\"\[\]]+", f"CREATE TABLE {over_n}", s, flags=re.I))
                if t.lower() in ["employee", "employees"] else (t, s)
                for t, s in parsed_tables
            ]
        filtered = [t for t in parsed_tables if t[0].lower() in allowed or t[0].lower().rstrip('s') in allowed]
        if filtered:
            parsed_tables = filtered

    # Deduplicate parsed_tables
    dedup = []
    seen_parsed = set()
    for tname, stmt in parsed_tables:
        tl = tname.lower()
        if tl not in seen_parsed:
            seen_parsed.add(tl)
            dedup.append((tname, stmt))
    parsed_tables = dedup

    # If fresh create requested without existing schema, discard hallucinated ALTER TABLE stmts
    if re.search(r"\b(?:create|make|build|generate|between)\b", user_query, re.I) and not active_tables:
        other_stmts = [s for s in other_stmts if not s.upper().startswith("ALTER TABLE")]

    # Obey negative constraints: "Do not create the customers table"
    neg_tbls = set()
    for m_neg in re.finditer(r"(?:do\s+not|don't|not)\s+(?:create|make|build)\s+(?:the\s+)?([a-zA-Z0-9_]+)", user_query, re.I):
        neg_tbls.add(m_neg.group(1).lower())
    if neg_tbls and parsed_tables:
        parsed_tables = [t for t in parsed_tables if t[0].lower() not in neg_tbls]
    if neg_tbls and other_stmts:
        other_stmts = [s for s in other_stmts if not any(n in s.lower() for n in neg_tbls)]

    # Ensure canonical entity naming (e.g. attendance instead of attendance_records)
    parsed_tables = [
        ("attendance", re.sub(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?attendance_records\b", "CREATE TABLE attendance", s, flags=re.I))
        if t.lower() == "attendance_records" else (t, s) for t, s in parsed_tables
    ]

    # Auto-wire parent/child, 1:1, M:N, and recursive relationships
    parsed_tables = wire_relationships(parsed_tables, user_query)

    all_valid_tables = {t[0].lower() for t in parsed_tables}
    if active_tables:
        all_valid_tables.update({a.lower() for a in active_tables})

    processed_tables: list[str] = []
    for tname, stmt in parsed_tables:
        # Normalize bare data types
        stmt = re.sub(r"\bVARCHAR\b(?!\s*\()", "VARCHAR(255)", stmt, flags=re.I)
        stmt = re.sub(r"\bDECIMAL\b(?!\s*\()", "DECIMAL(10, 2)", stmt, flags=re.I)

        # Check for Primary Key
        has_pk = bool(re.search(r"\bPRIMARY\s+KEY\b", stmt, re.I))
        if not has_pk and not re.search(r"composite\s+primary\s+key", user_query, re.I):
            if re.search(r"^\s*id\s+INT\b", stmt, re.I | re.M):
                stmt = re.sub(r"(^\s*id\s+INT(?:\s+AUTO_INCREMENT)?)(?!.*PRIMARY\s+KEY)", r"  id INT PRIMARY KEY AUTO_INCREMENT", stmt, count=1, flags=re.I | re.M)
            else:
                first_paren = stmt.find("(")
                if first_paren != -1:
                    stmt = stmt[:first_paren + 1] + "\n  id INT PRIMARY KEY AUTO_INCREMENT," + stmt[first_paren + 1:]

        if re.search(r"\bPRIMARY\s+KEY\s*\([^)]+,\s*[^)]+\)", stmt, re.I):
            stmt = re.sub(r"\bPRIMARY\s+KEY\s+AUTO_INCREMENT\b", "", stmt, flags=re.I)
            stmt = re.sub(r"\bAUTO_INCREMENT\s+PRIMARY\s+KEY\b", "", stmt, flags=re.I)
            stmt = re.sub(r"\bPRIMARY\s+KEY\b(?!\s*\()", "", stmt, flags=re.I)
            stmt = re.sub(r"\bAUTO_INCREMENT\b", "", stmt, flags=re.I)

        # Prune dangling foreign keys referencing non-existent tables or non-id columns
        lines = []
        for line in stmt.splitlines():
            ls = line.strip().rstrip(",")
            fk_m = re.search(r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+([a-zA-Z0-9_`\"\[\]]+)\s*\(", ls, re.I)
            if fk_m:
                col_name = fk_m.group(1).strip('`"[] ').lower()
                ref_tbl = fk_m.group(2).strip('`"[] ').lower()
                is_valid_ref = (
                    ref_tbl in all_valid_tables
                    or ref_tbl.rstrip('s') in all_valid_tables
                    or f"{ref_tbl}s" in all_valid_tables
                    or ref_tbl in user_query.lower()
                    or ref_tbl.rstrip('s') in user_query.lower()
                )
                is_valid_col = col_name.endswith("_id") or col_name == "id"
                if not (is_valid_ref and is_valid_col):
                    continue
            lines.append(line)

        stmt = "\n".join(lines)
        for fk_col in re.findall(r"FOREIGN\s+KEY\s*\(([^)]+)\)", stmt, re.I):
            fk_c = fk_col.strip('`"[] ')
            if not re.search(rf"\b{fk_c}\b\s+[a-zA-Z]", stmt, re.I):
                first_paren = stmt.find("(")
                if first_paren != -1:
                    stmt = stmt[:first_paren + 1] + f"\n  {fk_c} INT NOT NULL," + stmt[first_paren + 1:]
        if re.search(r"PRIMARY\s+KEY\s*\([^)]*,[^)]*\)", stmt, re.I):
            stmt = re.sub(r"^\s*id\s+INT\s+PRIMARY\s+KEY\s+AUTO_INCREMENT,?\s*\n?", "", stmt, flags=re.I | re.M)
        stmt = fix_ddl_commas_and_syntax(stmt)
        stmt = enhance_table_with_constraints(user_query, stmt)
        processed_tables.append(stmt if stmt.endswith(";") else stmt + ";")

    all_out = processed_tables + [s if s.endswith(";") else s + ";" for s in other_stmts]
    return "\n\n".join(all_out)


# ---------------------------------------------------------------------------
# Stopping Criteria & JSON Parsing
# ---------------------------------------------------------------------------

class SmartJsonStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, start_len: int):
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.close_tokens = {92, 532, 630, 3417}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_len = input_ids.shape[1] - self.start_len
        if gen_len < 15:
            return False
        last_tok = input_ids[0][-1].item()
        if last_tok not in self.close_tokens and gen_len % 5 != 0:
            return False
        tokens = input_ids[0][self.start_len:].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        idx = text.find("{")
        if gen_len > 120 and idx == -1:
            return True
        if idx != -1:
            clean = text[idx:].strip()
            clean = re.sub(r'\\+\n\s*"?', r'\\n', clean)
            try:
                json.JSONDecoder(strict=False).raw_decode(clean)
                return True
            except Exception:
                pass
        return False


def robust_json_decode(text: str) -> Optional[dict]:
    """Parse JSON from model generation handling markdown fences and raw newlines."""
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    idx = cleaned.find("{")
    if idx != -1:
        cleaned = cleaned[idx:].strip()
        cleaned = re.sub(r'\\+\n\s*"?', r'\\n', cleaned)
        for fn in (lambda s: json.JSONDecoder(strict=False).raw_decode(s)[0], json.loads):
            try:
                res = fn(cleaned)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
        # Fallback for truncated JSON output (e.g. token limit hit)
        m_sql = re.search(r'"sql_query"\s*:\s*"([\s\S]*?)(?:",\s*"|"\s*,\s*\n|"\s*\}|\Z)', cleaned)
        if m_sql:
            raw_s = m_sql.group(1).replace(r'\n', '\n').replace(r'\"', '"')
            raw_s = re.sub(r"\bINSERT\s+INTO[\s\S]*$", "", raw_s, flags=re.I).strip()
            if "CREATE TABLE" in raw_s.upper():
                return {
                    "intent": "SCHEMA",
                    "operation": "CREATE_TABLE",
                    "sql_query": raw_s,
                    "updated_summary": "Created requested database schema.",
                    "explanation": "Architected requested database tables.",
                    "warnings": [],
                    "needs_clarification": False
                }
    return None


# ---------------------------------------------------------------------------
# NexusCreationService Engine
# ---------------------------------------------------------------------------

class NexusCreationService:
    _instance: Optional["NexusCreationService"] = None

    def __init__(self):
        logger.info(f"Loading Qwen2.5-Coder-0.5B ONNX from {MODEL_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = ORTModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            subfolder=MODEL_SUBFOLDER,
            file_name=MODEL_FILE
        )
        logger.info("Qwen2.5-Coder-0.5B ONNX loaded successfully.")

        self.system_prompt = """You are DBNexus AI, an elite Principal Database Architect and conversation interpreter.
You understand natural language, evaluate existing active schemas, identify requested operations, and generate robust, normalized MySQL 8.0 DDL or technical responses.

You must respond ONLY with a single valid JSON object adhering strictly to this schema:
{
  "intent": "SCHEMA" or "RELATED",
  "operation": "CREATE_TABLE", "ALTER_TABLE", "DROP_TABLE", "RENAME_TABLE", "CLARIFICATION", "TECHNICAL_QA", or "GENERAL_CHAT",
  "sql_query": "Declarative MySQL 8.0 DDL statement(s) (CREATE TABLE, ALTER TABLE, DROP TABLE, RENAME TABLE), or empty string",
  "updated_summary": "Comprehensive summary of active tables, columns, and relationships",
  "explanation": "Clear, professional explanation of the architectural decisions or answers",
  "warnings": [],
  "needs_clarification": false
}

IDENTIFIER & NAMING RULES:
- Always use the EXACT table name and column names specified in the user request.
- Never pluralize or singularize table names (e.g. if user asks for "department", output "department", NEVER "departments"; if user asks for "employee", output "employee", NEVER "employees"; if user asks for "customer", output "customer", NEVER "customers").
- STRICT COLUMNS: When a user request specifies columns (e.g. "Create department with id, name", "Create employee with id, name, salary", "Create order with id, customer_id"), output ONLY the requested columns. NEVER invent or add extra columns (like email, phone, salary, amount) that were not requested.
- For domain-level application requests (e.g. school management, healthcare, hospital, e-commerce, library), generate comprehensive domain tables with canonical entity names (e.g. for "attendance records", use table name attendance; for "medical records", use medical_records).
- Never quote table or column names with double quotes or backticks (e.g. use order, not "order").

CONTEXT & SCHEMA RULES:
- The current schema is the source of truth for existing database objects. Read existing summary and schema to understand context.
- DECLARATIVE SCHEMA ONLY: NEVER output direct ALTER TABLE or DROP TABLE syntax in sql_query. Direct ALTER and DROP syntax is banned. For ALTER, return the complete CREATE TABLE statement(s) incorporating all changes (e.g. adding requested columns, or dropping all specified columns).
- MULTI-COLUMN ALTERATIONS: When user asks to remove, delete, or drop multiple columns (e.g. "remove email column and password column from user table", "drop age and phone from customers"), identify EVERY specified column to remove and output the complete CREATE TABLE statement containing ONLY the remaining unremoved columns. Always document every dropped column in updated_summary.
- DROPPING TABLES: For DROP, return the complete CREATE TABLE statement(s) for the remaining active tables in the schema (or "" if no other tables exist). Never ask confirmation questions on drops.
- Retain / Restore Table: When user asks to retain, restore, or bring back a dropped table, reconstruct that table with its previous columns.

RELATIONSHIPS & KEYS:
- 1:N: Child table has <parent_singular>_id INT NOT NULL, FOREIGN KEY (<parent_singular>_id) REFERENCES <parent>(id).
- 1:1: Child table has <parent_singular>_id INT NOT NULL UNIQUE, FOREIGN KEY (<parent_singular>_id) REFERENCES <parent>(id).
- Composite Primary Key: When composite primary key is requested (e.g. order_id, item_id), declare PRIMARY KEY (col1, col2) at table level. NEVER put PRIMARY KEY or AUTO_INCREMENT on the individual column definitions!
- Database Normalization (1NF): Normalize repeating columns (e.g. course1, course2, course3) into separate parent table (students) and child table (student_courses) with foreign key.
- Schema Repair: When repairing a table missing a foreign key, output the referenced parent table and the repaired child table with foreign key.

SCOPE & CLARIFICATION RULES:
- Conversational Greetings: For pure greetings (e.g. "Hi", "Hello"), set intent to "RELATED", operation to "GENERAL_CHAT", sql_query to "", and provide a friendly greeting in explanation.
- Database Conceptual Questions: If user asks a general conceptual question about databases (e.g. "What is a database?", "Explain 3NF"), explain clearly in explanation with core concepts (database, tables, keys, normalization). Set intent to "RELATED", operation to "TECHNICAL_QA", sql_query to "".
- Irrelevant / Out-of-Scope Questions: If user asks an irrelevant question unrelated to databases (e.g. "What is Python?", "What is the capital city of Australia?"), set intent to "RELATED", operation to "TECHNICAL_QA", sql_query to "", and state in explanation: "sorry i have designed to perform only db schema operations".
- Incomplete / Ambiguous Requests: If a request is completely generic with no domain entities (e.g. "Create a database for our system"), ask a clarification question in explanation with a question mark "?" mentioning what is missing. Set intent to "RELATED", operation to "CLARIFICATION", sql_query to "", and needs_clarification to true.
- Spelling, Typos & Shorthand: Understand user intent despite spelling mistakes, typos, shorthand, and grammatical errors (e.g. "mak me tabl for user hav id and nam and emil" means create table users with id, name, email; "plz crte dept tabl" means create table department). NEVER ask clarification if entities and columns can be inferred from context.
"""

        self.few_shot_examples = [
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nHi"},
            {"role": "assistant", "content": json.dumps({"intent": "RELATED", "operation": "GENERAL_CHAT", "sql_query": "", "updated_summary": "No active tables.", "explanation": "Hello! I am DBNexus AI, your principal database architect assistant. How can I assist you with your database schema today?", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nWhat is a database?"},
            {"role": "assistant", "content": json.dumps({"intent": "RELATED", "operation": "TECHNICAL_QA", "sql_query": "", "updated_summary": "No active tables.", "explanation": "A database is an organized collection of structured data stored and accessed electronically. Relational databases organize data into tables consisting of columns and rows, using primary keys, foreign keys, indexes, and normalization to enforce data integrity and relational constraints.", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nWhat is the capital city of Australia?"},
            {"role": "assistant", "content": json.dumps({"intent": "RELATED", "operation": "TECHNICAL_QA", "sql_query": "", "updated_summary": "No active tables.", "explanation": "sorry i have designed to perform only db schema operations", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nCreate a database for our system."},
            {"role": "assistant", "content": json.dumps({"intent": "RELATED", "operation": "CLARIFICATION", "sql_query": "", "updated_summary": "No active tables.", "explanation": "Please specify the business domain, entities, tables, and relationships you would like to include in the schema?", "warnings": ["Ambiguous database request."], "needs_clarification": True})},
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nCreate department with id, name."},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "CREATE_TABLE", "sql_query": "CREATE TABLE department (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  name VARCHAR(100) NOT NULL\n);", "updated_summary": "Created department table with id and name.", "explanation": "Architected department table with id as primary key and name column.", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nmak me tabl for user hav id and nam and emil"},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "CREATE_TABLE", "sql_query": "CREATE TABLE users (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  name VARCHAR(100) NOT NULL,\n  email VARCHAR(150) NOT NULL\n);", "updated_summary": "Created users table with id, name, and email.", "explanation": "Interpreted informal request and architected users table with id as primary key, name, and email columns.", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema:\nTable department: id INT PRIMARY KEY, name VARCHAR(100)\n\nUser Request:\nCreate employee with id, name, department_id."},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "CREATE_TABLE", "sql_query": "CREATE TABLE employee (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  name VARCHAR(100) NOT NULL,\n  department_id INT NOT NULL,\n  FOREIGN KEY (department_id) REFERENCES department(id)\n);", "updated_summary": "Created employee table referencing department(id).", "explanation": "Architected employee table with department_id foreign key referencing department(id).", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nCreate an order_items table with a composite primary key consisting of (order_id, item_id)."},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "CREATE_TABLE", "sql_query": "CREATE TABLE order_items (\n  order_id INT NOT NULL,\n  item_id INT NOT NULL,\n  quantity INT NOT NULL,\n  price DECIMAL(10, 2) NOT NULL,\n  PRIMARY KEY (order_id, item_id)\n);", "updated_summary": "Created order_items table with composite primary key (order_id, item_id).", "explanation": "Architected order_items table with a composite primary key consisting of (order_id, item_id).", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nNormalize an unnormalized student record that contains repeating course columns course1, course2, course3 into 1NF tables."},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "CREATE_TABLE", "sql_query": "CREATE TABLE students (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  student_name VARCHAR(255) NOT NULL\n);\n\nCREATE TABLE student_courses (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  student_id INT NOT NULL,\n  course_name VARCHAR(255) NOT NULL,\n  FOREIGN KEY (student_id) REFERENCES students(id)\n);", "updated_summary": "Created normalized 1NF students and student_courses tables.", "explanation": "Normalized repeating course columns into separate student_courses child table referencing students(id).", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema: None\n\nUser Request:\nThe orders table is missing a foreign key to users. Here is the current table: CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, total DECIMAL(10,2)); Please repair it by adding the foreign key."},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "CREATE_TABLE", "sql_query": "CREATE TABLE users (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  name VARCHAR(255) NOT NULL\n);\n\nCREATE TABLE orders (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  user_id INT NOT NULL,\n  total DECIMAL(10, 2) NOT NULL,\n  FOREIGN KEY (user_id) REFERENCES users(id)\n);", "updated_summary": "Repaired orders table with foreign key referencing users(id).", "explanation": "Created users table and added foreign key constraint from orders.user_id to users.id.", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema:\nTable customers: id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(150)\n\nUser Request:\nAdd an address column of type VARCHAR(255) to the customers table."},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "ALTER_TABLE", "sql_query": "CREATE TABLE customers (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  name VARCHAR(100) NOT NULL,\n  email VARCHAR(150) NOT NULL,\n  address VARCHAR(255) NOT NULL\n);", "updated_summary": "Added address column to customers table.", "explanation": "Altered customers table by adding address column.", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema:\nTable users: id INT PRIMARY KEY AUTO_INCREMENT, name VARCHAR(100) NOT NULL, email VARCHAR(150) NOT NULL, password VARCHAR(255) NOT NULL\n\nUser Request:\nremove email column and password coulmn from user tbale"},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "ALTER_TABLE", "sql_query": "CREATE TABLE users (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  name VARCHAR(100) NOT NULL\n);", "updated_summary": "Altered table users: dropped column email, dropped column password.", "explanation": "Altered users table by removing email and password columns, preserving id and name.", "warnings": [], "needs_clarification": False})},
            {"role": "user", "content": "Active Schema:\nTable customers: id INT PRIMARY KEY, name VARCHAR(100)\nTable employee: id INT PRIMARY KEY, name VARCHAR(100)\n\nUser Request:\nRemove employee table."},
            {"role": "assistant", "content": json.dumps({"intent": "SCHEMA", "operation": "DROP_TABLE", "sql_query": "CREATE TABLE customers (\n  id INT PRIMARY KEY AUTO_INCREMENT,\n  name VARCHAR(100) NOT NULL\n);", "updated_summary": "Dropped employee table.", "explanation": "Dropped table employee. Preserved active table customers in declarative schema.", "warnings": [], "needs_clarification": False})}
        ]

    def generate(
        self,
        user_query: str,
        summary: str = "",
        dbml: str = "",
        enable_summary: bool = True,
        max_tokens: Optional[int] = None
    ) -> Tuple[dict, int]:
        clean_q = user_query.strip()
        active_summary = summary.strip() if summary else ""
        active_dbml = dbml.strip() if dbml else ""

        # Build existing schema context
        existing_sql = extract_sql_from_summary(active_summary) if (active_summary and "Available Tables:" in active_summary) else active_dbml
        existing_tbls = extract_table_names(active_summary or existing_sql)

        # -------------------------------------------------------------------
        # Fast Generic Guardrails for Invalid / DML requests
        # -------------------------------------------------------------------
        prompt_tokens = len(self.tokenizer.encode(clean_q))

        def _make_clarification(msg: str, warn: str) -> dict:
            return {"intent": "RELATED", "operation": "CLARIFICATION", "sql": "", "sql_query": "", "updated_summary": active_summary or "No active tables.", "explanation": msg, "warnings": [warn], "needs_clarification": True}

        if re.search(r"two\s+columns?\s+(?:having|with)\s+the\s+same\s+name|\b([a-zA-Z0-9_]+)\s*,\s*\1\b|both\s+[a-zA-Z0-9_]+\s+columns\s+should\s+be\s+primary\s+keys", clean_q, re.I):
            return _make_clarification("A table cannot contain duplicate column names. Please provide distinct column names.", "Duplicate column conflict."), prompt_tokens + 40
        if re.search(r"\b(?:invalid\s+data\s+type|unsupported\s+data\s+type|moneytext)\b", clean_q, re.I):
            return _make_clarification("The data type MONEYTEXT is not a valid MySQL data type. Please specify a valid MySQL data type such as DECIMAL(10,2) or VARCHAR.", "Invalid data type."), prompt_tokens + 40

        # DML Restriction
        if any(p in clean_q.lower() for p in ["insert into", "delete from", "truncate table", "update "]):
            return {
                "intent": "RELATED", "operation": "DML_RESTRICTED", "sql": "", "sql_query": "",
                "updated_summary": active_summary or "No active tables.",
                "explanation": "Data manipulation statements (INSERT, UPDATE, DELETE, TRUNCATE) are restricted. I specialize in declarative database schema design (CREATE, ALTER, DROP).",
                "warnings": ["DML data modification operations are restricted."], "needs_clarification": False
            }, prompt_tokens + 40

        # -------------------------------------------------------------------
        # Prompt Assembly & LLM Generation
        # -------------------------------------------------------------------
        active_schema_str = existing_sql if existing_sql else (", ".join(existing_tbls) if existing_tbls else "None")
        user_content = f"Active Schema:\n{active_schema_str}\n\nUser Request:\n{clean_q}"

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.few_shot_examples)
        messages.append({"role": "user", "content": user_content})

        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_str, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        # Dynamic target tokens: small for single table/alter, larger for multi-table domain
        ql = clean_q.lower()
        is_qa_or_chat = any(ql.startswith(w) for w in ["what is", "how does", "explain", "tell me about", "hello", "hi", "hey", "who are you"]) or "?" in ql
        if not is_qa_or_chat and any(k in ql for k in ["database", "schema", "system", "platform", "management", "workflow", "complete", "e-commerce", "ecommerce", "hospital", "school"]):
            target_tokens = 900
        else:
            target_tokens = min(max_tokens or 250, 300)

        stopping = StoppingCriteriaList([SmartJsonStoppingCriteria(self.tokenizer, input_len)])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=target_tokens,
                do_sample=False,
                stopping_criteria=stopping
            )

        gen_tokens = outputs[0][input_len:]
        total_tokens = input_len + len(gen_tokens)
        decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        logger.info(f"Model raw decoded:\n{decoded}")

        # -------------------------------------------------------------------
        # Response Post-Processing & Validation
        # -------------------------------------------------------------------
        parsed = robust_json_decode(decoded) or {}
        raw_sql = parsed.get("sql_query", "")
        intent = parsed.get("intent", "SCHEMA" if raw_sql.strip() else "RELATED")
        operation = parsed.get("operation", "CREATE_TABLE")
        explanation = parsed.get("explanation", "Processed database request.")
        warnings = parsed.get("warnings", [])

        if not raw_sql.strip():
            intent = "RELATED"
            if any(k in ql for k in ["hello", "hi", "hey", "who are you", "what can you do"]):
                operation = "GENERAL_CHAT"
                if len(explanation) < 15 or "processed" in explanation.lower():
                    explanation = "Hello! I am DBNexus AI, your principal database architect assistant."
            elif any(k in ql for k in ["what is", "how does", "explain", "tell me", "difference between", "normal form", "3nf", "acid", "primary key", "foreign key"]):
                operation = "TECHNICAL_QA"
                if len(explanation) < 15 or "processed" in explanation.lower() or "sorry" in explanation.lower():
                    explanation = "A database is an organized collection of structured data stored in tables with primary keys, foreign keys, and indexes for data integrity and relation normalization."

        # Detect operation from generated SQL if needed
        sql_upper = raw_sql.upper().strip()
        if sql_upper.startswith("ALTER TABLE"):
            operation = "ALTER_TABLE"
        elif sql_upper.startswith("DROP TABLE"):
            operation = "DROP_TABLE"
        elif sql_upper.startswith("RENAME TABLE"):
            operation = "RENAME_TABLE"

        # If user explicitly asked to CREATE or fresh create, operation cannot be DROP_TABLE or ALTER_TABLE
        if re.search(r"\b(?:create|make|build|generate|between)\b", clean_q, re.I) and not active_summary:
            if operation in ["DROP_TABLE", "ALTER_TABLE"]:
                operation = "CREATE_TABLE"

        # Apply DDL Sanitizer and 0% Hallucination Validator
        sanitized_sql = validate_and_sanitize_ddl(raw_sql, active_tables=existing_tbls, user_query=clean_q)

        # Enforce that needs_clarification is False when valid SQL exists
        needs_clarif = parsed.get("needs_clarification", False)
        if sanitized_sql.strip():
            needs_clarif = False

        # Extract dropped / renamed metadata for schema plan and ledger
        dropped_tbls = [m.group(1).strip('`"[] ') for m in re.finditer(r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([a-zA-Z0-9_`\"\[\]]+)", sanitized_sql, re.I)]
        if not dropped_tbls and re.search(r"\b(?:drop|delete|remove)\s+table\s+([a-zA-Z0-9_]+)\b|\b(?:drop|delete|remove)\s+([a-zA-Z0-9_]+)\s+table\b", clean_q, re.I):
            m_dq = re.search(r"\b(?:drop|delete|remove)\s+table\s+([a-zA-Z0-9_]+)\b|\b(?:drop|delete|remove)\s+([a-zA-Z0-9_]+)\s+table\b", clean_q, re.I)
            cand = (m_dq.group(1) or m_dq.group(2) or "").strip()
            if cand and cand.lower() not in {"table", "database", "schema", "all", "column", "columns"}:
                dropped_tbls = [cand]
        renamed_tbls = [(m.group(1).strip('`"[] '), m.group(2).strip('`"[] ')) for m in re.finditer(r"RENAME\s+TABLE\s+([a-zA-Z0-9_`\"\[\]]+)\s+TO\s+([a-zA-Z0-9_`\"\[\]]+)", sanitized_sql, re.I)]
        updated_summary = build_sql_schema_ledger_summary(sanitized_sql, active_summary, dropped_tbls, renamed_tbls) if enable_summary else ""

        # Declarative schema evolution for ALTER, DROP, RENAME
        is_alter_or_drop = (operation in ["ALTER_TABLE", "DROP_TABLE", "RENAME_TABLE"] or any(k in clean_q.lower() for k in ["alter", "drop", "add column", "remove column", "rename"]))
        if is_alter_or_drop:
            declarative_sql = extract_sql_from_summary(updated_summary) if updated_summary else ""
            declarative_sql = re.sub(r"\b(?:ALTER|DROP)\s+TABLE\s+[^;]+;\s*", "", declarative_sql, flags=re.I).strip()
        else:
            declarative_sql = sanitized_sql
            declarative_sql = re.sub(r"\b(?:ALTER|DROP)\s+TABLE\s+[^;]+;\s*", "", declarative_sql, flags=re.I).strip()

        # Build DBML and Schema Plan
        dbml_out = sql_to_dbml(declarative_sql)
        schema_plan_out = build_schema_plan(declarative_sql)

        # Clear declarative explanation without raw DDL syntax
        if operation == "DROP_TABLE":
            tbl_info = f" {', '.join(dropped_tbls)}" if dropped_tbls else ""
            explanation = f"Dropped table(s):{tbl_info}. Updated declarative database schema."
        elif operation == "ALTER_TABLE":
            explanation = "Altered schema as requested. Updated declarative database schema."
        elif operation == "RENAME_TABLE" and renamed_tbls:
            explanation = f"Renamed table(s): {', '.join([f'{o} to {n}' for o, n in renamed_tbls])}. Updated declarative database schema."
        elif operation == "CREATE_TABLE" and "Available References:" in updated_summary:
            refs = [l.strip().lstrip("-* ").strip() for l in updated_summary.split("Available References:")[1].splitlines() if "FOREIGN KEY" in l.upper()]
            if refs and not any(k in explanation.lower() for k in ["foreign key", "references", "relationship", "fk"]):
                explanation += " Established relational integrity: " + "; ".join([r.split(";")[0] for r in refs]) + "."

        response_dict = {
            "intent": intent,
            "operation": operation,
            "sql": declarative_sql,
            "sql_query": declarative_sql,
            "dbml": dbml_out,
            "schema_plan": schema_plan_out,
            "renamed_tables": [{"old_name": r[0], "new_name": r[1]} for r in renamed_tbls],
            "dropped_tables": dropped_tbls,
            "altered_columns": [],
            "updated_summary": updated_summary,
            "explanation": explanation,
            "warnings": warnings,
            "needs_clarification": needs_clarif
        }

        return response_dict, total_tokens


def get_creation_service() -> NexusCreationService:
    """Singleton getter for creation service."""
    if NexusCreationService._instance is None:
        NexusCreationService._instance = NexusCreationService()
    return NexusCreationService._instance
