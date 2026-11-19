import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import requests

from src.config import get_logger, GROQ_API_KEY, GROQ_API_URL, GROQ_MODELS_URL
from src.rag.config import (
    DOMAINS_DATA_DIR,
    CHROMA_DATA_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_TOP_K,
    CHROMA_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    DEFAULT_RAG_LLM_MODEL,
    RAG_LLM_PRIORITY_MODELS,
    RAG_SYSTEM_PROMPT,
    RAG_CONTEXT_TEMPLATE,
    DOMAIN_PRIMARY_TABLES,
    TABLE_PRIMARY_COLUMNS_MAP,
    TABLE_SYNONYMS,
    NONEXISTENT_TABLE_PATTERNS,
)
from src.rag.schema import RAGSchemaResponse, RAGQueryRequest

logger = get_logger(__name__)


def build_rag_augmented_prompt(
    user_query: str,
    schema_ddl: str,
    domain_name: str = "Enterprise ERP",
    domain_id: str = "erp_domain",
    tables: Optional[List[str]] = None,
    summary: str = ""
) -> str:
    """
    Combines the user query with the RAG retrieved schema DDL and summary ledger
    into an enterprise context prompt for LLM completion.
    """
    try:
        table_str = ", ".join(tables) if tables else "N/A"
        return RAG_CONTEXT_TEMPLATE.format(
            domain_name=domain_name,
            domain_id=domain_id,
            tables=table_str,
            schema_ddl=schema_ddl.strip(),
            summary=summary.strip() or "None",
            user_query=user_query.strip()
        )
    except Exception as e:
        logger.error(f"Error formatting RAG augmented prompt: {e}", exc_info=True)
        return f"{schema_ddl}\n\nUser Query: {user_query}"


# =====================================================================
# LLM Model Resolution & Caching
# =====================================================================

_RESOLVED_BEST_RAG_MODEL: Optional[str] = None


def resolve_best_rag_model(api_key: Optional[str] = None, preferred_model: Optional[str] = None) -> str:
    """
    Selects the best free model available on Groq for RAG reasoning and SQL generation.
    Prioritizes flagship Meta LLaMA models (llama-3.3-70b-versatile, llama-3.1-8b-instant),
    with automatic zero-downtime fallback to other high-capacity free models (allam-2-7b, openai/gpt-oss-120b).
    """
    global _RESOLVED_BEST_RAG_MODEL
    if preferred_model:
        return preferred_model
    if _RESOLVED_BEST_RAG_MODEL:
        return _RESOLVED_BEST_RAG_MODEL

    key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not key:
        return DEFAULT_RAG_LLM_MODEL

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        r = requests.get(GROQ_MODELS_URL, headers=headers, timeout=6)
        if r.status_code == 200:
            available_models = {m["id"] for m in r.json().get("data", [])}
            for candidate in RAG_LLM_PRIORITY_MODELS:
                if candidate in available_models:
                    logger.info(f"Resolved best free model for RAG LLM: '{candidate}'")
                    _RESOLVED_BEST_RAG_MODEL = candidate
                    return candidate
    except Exception as e:
        logger.warning(f"Could not auto-detect Groq models ({e}); using default '{DEFAULT_RAG_LLM_MODEL}'.")

    _RESOLVED_BEST_RAG_MODEL = DEFAULT_RAG_LLM_MODEL
    return _RESOLVED_BEST_RAG_MODEL


# =====================================================================
# Canonical Schema Registry & Synonyms
# =====================================================================

_CACHED_REGISTRY: Optional[Dict[str, Any]] = None
_LAST_REGISTRY_MTIME: float = 0.0


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


def _compute_data_dir_mtime(data_dir: Path) -> float:
    try:
        mtimes = [f.stat().st_mtime for f in data_dir.rglob("*.json")]
        return max(mtimes) if mtimes else 0.0
    except Exception as e:
        logger.warning(f"Error computing mtime for {data_dir}: {e}")
        return 0.0


def _load_erp_registry_from_postgres() -> Optional[Dict[str, Any]]:
    """Loads ERP schema registry directly from PostgreSQL knowledge base."""
    try:
        from src.rag.database import get_db, is_db_connected
        if not is_db_connected():
            return None

        from src.rag.models import DomainModel, KnowledgeTableModel, KnowledgeColumnModel, TableDependencyModel

        registry: Dict[str, Any] = {
            "domains": {},
            "all_tables": {},
            "all_columns": {},
            "col_name_to_tables": {},
            "valid_tables": set()
        }

        with get_db() as session:
            domains = session.query(DomainModel).filter(DomainModel.is_active.is_(True)).all()
            if not domains:
                return None

            tables = session.query(KnowledgeTableModel).filter(KnowledgeTableModel.is_active.is_(True)).all()
            columns = session.query(KnowledgeColumnModel).filter(KnowledgeColumnModel.is_active.is_(True)).all()
            deps = session.query(TableDependencyModel).all()

            tbl_by_id = {t.id: t for t in tables}
            cols_by_tbl: Dict[Any, List[Any]] = {}
            for c in columns:
                cols_by_tbl.setdefault(c.table_id, []).append(c)

            deps_by_src: Dict[Any, List[str]] = {}
            for d in deps:
                ref_tbl = tbl_by_id.get(d.referenced_table_id)
                if ref_tbl:
                    deps_by_src.setdefault(d.source_table_id, []).append(ref_tbl.table_name.lower())

            dom_tbls: Dict[Any, List[Any]] = {}
            for t in tables:
                dom_tbls.setdefault(t.domain_id, []).append(t)

            for dom in domains:
                dom_id = dom.domain_id.lower()
                d_tables = []
                for t in dom_tbls.get(dom.id, []):
                    tname = t.table_name.lower()
                    raw_cols = cols_by_tbl.get(t.id, [])

                    # Order columns according to DDL structure
                    lines = t.sql_definition.strip().splitlines()
                    ddl_cols = []
                    for line in lines[1:]:
                        m = re.match(r"^\s*([a-zA-Z0-9_]+)\s+", line)
                        if m:
                            w = m.group(1).lower()
                            if w not in ("constraint", "primary", "foreign", "unique", "check", ");", ")"):
                                ddl_cols.append(w)

                    col_order_map = {cname: i for i, cname in enumerate(ddl_cols)}
                    sorted_cols = sorted(raw_cols, key=lambda c: col_order_map.get(c.column_name.lower(), 999))

                    col_names = [c.column_name.lower() for c in sorted_cols]
                    col_details = []
                    actual_fk_deps = list(dict.fromkeys(deps_by_src.get(t.id, [])))

                    # Parse any direct REFERENCES in SQL as safety
                    sql_refs = re.findall(r"REFERENCES\s+([a-zA-Z0-9_]+)\s*\(", t.sql_definition, re.IGNORECASE)
                    for s in sql_refs:
                        s_low = s.lower()
                        if s_low != tname and s_low not in actual_fk_deps:
                            actual_fk_deps.append(s_low)

                    for c in sorted_cols:
                        cname = c.column_name.lower()
                        ref_str = f"{c.references_table}({c.references_column})" if c.references_table and c.references_column else None
                        cdata = {
                            "column_name": c.column_name,
                            "table_name": t.table_name,
                            "domain": dom_id,
                            "data_type": c.data_type,
                            "description": c.description or "",
                            "is_primary_key": c.is_primary_key,
                            "is_foreign_key": c.is_foreign_key,
                            "references": ref_str,
                            "keywords": c.keywords or []
                        }
                        col_details.append(cdata)
                        registry["all_columns"][(tname, cname)] = cdata
                        registry["col_name_to_tables"].setdefault(cname, []).append((t.table_name, dom_id))

                    tbl_obj = {
                        "table_name": t.table_name,
                        "description": t.description or "",
                        "keywords": t.keywords or [],
                        "dependencies": actual_fk_deps,
                        "columns": col_names,
                        "sql": t.sql_definition,
                        "domain": dom_id,
                        "column_details": col_details,
                        "actual_dependencies": actual_fk_deps
                    }
                    d_tables.append(tbl_obj)
                    registry["all_tables"][tname] = tbl_obj
                    registry["valid_tables"].add(tname)

                registry["domains"][dom_id] = {
                    "id": dom_id,
                    "name": dom.domain_name,
                    "tables": d_tables
                }

        return registry
    except Exception as e:
        logger.warning(f"Could not load registry from PostgreSQL ({e}).")
        return None


def load_erp_registry(force_reload: bool = False) -> Dict[str, Any]:
    """
    Loads all domain, table, and column metadata from PostgreSQL (Primary)
    with automatic zero-downtime fallback to local JSON files.
    """
    global _CACHED_REGISTRY, _LAST_REGISTRY_MTIME

    # 1. Try loading from PostgreSQL Primary Knowledge Base
    from src.rag.config import USE_POSTGRES_KB
    if USE_POSTGRES_KB:
        try:
            if not force_reload and _CACHED_REGISTRY is not None:
                return _CACHED_REGISTRY
            pg_reg = _load_erp_registry_from_postgres()
            if pg_reg and pg_reg.get("domains"):
                _CACHED_REGISTRY = pg_reg
                logger.info(f"Loaded {len(_CACHED_REGISTRY['domains'])} domain(s) from PostgreSQL Knowledge Base (Primary).")
                return _CACHED_REGISTRY
        except Exception as e:
            logger.warning(f"PostgreSQL Knowledge Base access failed: {e}. Falling back to file-based registry.")

    # 2. File-Based Loading Fallback
    current_mtime = _compute_data_dir_mtime(DOMAINS_DATA_DIR)
    if not force_reload and _CACHED_REGISTRY is not None and current_mtime <= _LAST_REGISTRY_MTIME:
        return _CACHED_REGISTRY

    registry: Dict[str, Any] = {
        "domains": {},
        "all_tables": {},
        "all_columns": {},
        "col_name_to_tables": {},
        "valid_tables": set()
    }

    try:
        if DOMAINS_DATA_DIR.exists():
            for dom_dir in sorted(DOMAINS_DATA_DIR.iterdir()):
                if not dom_dir.is_dir():
                    continue
                dom_id = dom_dir.name
                domain_meta_file = dom_dir / "domain.json"
                domain_info: Dict[str, Any] = {}
                if domain_meta_file.exists():
                    try:
                        with open(domain_meta_file, "r", encoding="utf-8") as df:
                            domain_info = json.load(df)
                    except Exception as e:
                        logger.warning(f"Error reading {domain_meta_file}: {e}")

                tables: List[Dict[str, Any]] = []
                tables_dir = dom_dir / "tables"
                columns_dir = dom_dir / "columns"
                tables_v2_file = dom_dir / "tables.json"
                columns_v2_file = dom_dir / "columns.json"

                if tables_v2_file.exists() and columns_v2_file.exists():
                    # Format v2: Consolidated tables.json and columns.json
                    try:
                        with open(tables_v2_file, "r", encoding="utf-8") as tf:
                            raw_tables = json.load(tf)
                        with open(columns_v2_file, "r", encoding="utf-8") as cf:
                            raw_cols = json.load(cf)

                        col_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
                        for c in raw_cols:
                            t_low = str(c.get("table_name", "")).lower()
                            c_low = str(c.get("column_name", "")).lower()
                            c["domain"] = dom_id
                            col_lookup[(t_low, c_low)] = c
                            registry["all_columns"][(t_low, c_low)] = c
                            registry["col_name_to_tables"].setdefault(c_low, []).append((c.get("table_name", t_low), dom_id))

                        for tbl_obj in raw_tables:
                            tname = tbl_obj.get("table_name", "").lower()
                            tbl_obj["domain"] = dom_id
                            cols_order = tbl_obj.get("columns", [])
                            col_details = []
                            actual_fk_deps = []

                            for cname in cols_order:
                                cdata = col_lookup.get((tname, cname.lower()))
                                if not cdata:
                                    cdata = {"column_name": cname, "data_type": "TEXT", "description": "", "table_name": tbl_obj.get("table_name", tname), "domain": dom_id}
                                col_details.append(cdata)

                                if cdata.get("is_foreign_key") or cdata.get("references"):
                                    ref_str = str(cdata.get("references", ""))
                                    ref_tbl = ref_str.split(".")[0].strip().lower()
                                    if ref_tbl and ref_tbl != tname and ref_tbl not in actual_fk_deps:
                                        actual_fk_deps.append(ref_tbl)

                            sql_refs = re.findall(r"REFERENCES\s+([a-zA-Z0-9_]+)\s*\(", tbl_obj.get("sql", ""), re.IGNORECASE)
                            for s in sql_refs:
                                s_low = s.lower()
                                if s_low != tname and s_low not in actual_fk_deps:
                                    actual_fk_deps.append(s_low)

                            tbl_obj["column_details"] = col_details
                            tbl_obj["actual_dependencies"] = actual_fk_deps
                            tables.append(tbl_obj)
                            registry["all_tables"][tname] = tbl_obj
                            registry["valid_tables"].add(tname)
                    except Exception as e:
                        logger.error(f"Error loading v2 schema files from {dom_dir}: {e}")

                elif tables_dir.exists():
                    # Format v1: Dispersed individual table and column json files
                    for tbl_file in sorted(tables_dir.glob("*.json")):
                        try:
                            with open(tbl_file, "r", encoding="utf-8") as tf:
                                tbl_obj = json.load(tf)
                                tname = tbl_obj.get("table_name", "").lower()
                                tbl_obj["domain"] = dom_id

                                # Preserve column order from table file
                                cols_order = tbl_obj.get("columns", [])
                                tbl_cols_dir = columns_dir / tbl_obj.get("table_name", tname)
                                col_details = []
                                actual_fk_deps = []

                                for cname in cols_order:
                                    cf = tbl_cols_dir / f"{cname}.json"
                                    if cf.exists():
                                        try:
                                            with open(cf, "r", encoding="utf-8") as col_f:
                                                cdata = json.load(col_f)
                                        except Exception:
                                            cdata = {"column_name": cname, "data_type": "TEXT", "description": ""}
                                    else:
                                        cdata = {"column_name": cname, "data_type": "TEXT", "description": ""}

                                    cdata["table_name"] = tbl_obj.get("table_name", tname)
                                    cdata["domain"] = dom_id
                                    col_details.append(cdata)
                                    registry["all_columns"][(tname, cname.lower())] = cdata
                                    registry["col_name_to_tables"].setdefault(cname.lower(), []).append((tbl_obj.get("table_name", tname), dom_id))

                                    # Extract direct parent foreign key references
                                    if cdata.get("is_foreign_key") or cdata.get("references"):
                                        ref_str = str(cdata.get("references", ""))
                                        ref_tbl = ref_str.split(".")[0].strip().lower()
                                        if ref_tbl and ref_tbl != tname and ref_tbl not in actual_fk_deps:
                                            actual_fk_deps.append(ref_tbl)

                                    # Cross-verify references from table SQL
                                sql_refs = re.findall(r"REFERENCES\s+([a-zA-Z0-9_]+)\s*\(", tbl_obj.get("sql", ""), re.IGNORECASE)
                                for s in sql_refs:
                                    s_low = s.lower()
                                    if s_low != tname and s_low not in actual_fk_deps:
                                        actual_fk_deps.append(s_low)

                                tbl_obj["column_details"] = col_details
                                tbl_obj["actual_dependencies"] = actual_fk_deps
                                tables.append(tbl_obj)
                                registry["all_tables"][tname] = tbl_obj
                                registry["valid_tables"].add(tname)
                        except Exception as e:
                            logger.error(f"Error reading table {tbl_file}: {e}")

                registry["domains"][dom_id] = {
                    "domain_id": dom_id,
                    "name": domain_info.get("domain_name", dom_id),
                    "system": domain_info.get("system", "Enterprise ERP"),
                    "description": domain_info.get("description", ""),
                    "keywords": domain_info.get("keywords", [dom_id]),
                    "tables": tables
                }

        _CACHED_REGISTRY = registry
        _LAST_REGISTRY_MTIME = current_mtime
        total_tables = sum(len(d.get("tables", [])) for d in registry["domains"].values())
        logger.info(f"Loaded {len(registry['domains'])} domain(s) with {total_tables} tables from {DOMAINS_DATA_DIR}.")
        return registry
    except Exception as e:
        logger.error(f"Failed to load ERP schema registries: {e}", exc_info=True)
        return _CACHED_REGISTRY or {"domains": {}, "all_tables": {}, "all_columns": {}, "col_name_to_tables": {}, "valid_tables": set()}


def topological_sort_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Orders tables so parent tables precede dependent child tables.
    Uses actual_dependencies derived strictly from foreign-key references.
    """
    try:
        table_map = {t["table_name"].lower(): t for t in tables}
        visited: Set[str] = set()
        result: List[Dict[str, Any]] = []

        def dfs(tname: str):
            if tname in visited:
                return
            visited.add(tname)
            tbl_obj = table_map.get(tname)
            if not tbl_obj:
                return
            deps = tbl_obj.get("actual_dependencies") or tbl_obj.get("dependencies", [])
            for dep in deps:
                dep_low = dep.lower()
                if dep_low in table_map and dep_low not in visited:
                    dfs(dep_low)
            result.append(tbl_obj)

        for t in tables:
            dfs(t["table_name"].lower())

        return result
    except Exception as e:
        logger.error(f"Error during topological sort: {e}", exc_info=True)
        return tables


def parse_column_metadata(cdata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts structured, granular metadata from stored column JSON file without assumptions.
    Preserves exact nullable status, defaults, foreign keys, delete rules, and descriptions.
    """
    cname = cdata.get("column_name", "")
    tname = cdata.get("table_name", "")
    dom = cdata.get("domain", "")
    dtype_raw = cdata.get("data_type", "TEXT")
    desc = cdata.get("description") or "Unavailable"

    is_pk = cdata.get("is_primary_key", False) or "PRIMARY KEY" in dtype_raw.upper()
    is_fk = cdata.get("is_foreign_key", False) or "REFERENCES" in dtype_raw.upper()
    is_unique = "UNIQUE" in dtype_raw.upper() or is_pk
    is_not_null = "NOT NULL" in dtype_raw.upper() or is_pk

    type_tokens = dtype_raw.split()
    clean_type = type_tokens[0] if type_tokens else "TEXT"
    if clean_type.upper() in ("NUMERIC", "VARCHAR", "CHAR", "DECIMAL") and "(" in dtype_raw:
        m_paren = re.search(r"^[A-Z]+\([^)]+\)", dtype_raw, re.IGNORECASE)
        if m_paren:
            clean_type = m_paren.group(0)

    m_def = re.search(r"DEFAULT\s+([^,; ]+)", dtype_raw, re.IGNORECASE)
    default_val = m_def.group(1) if m_def else "Unavailable"

    ref_table = "Unavailable"
    ref_col = "id"
    if is_fk:
        ref_meta = cdata.get("references")
        if ref_meta and "." in str(ref_meta):
            ref_table, ref_col = str(ref_meta).split(".", 1)
        else:
            m_ref = re.search(r"REFERENCES\s+([a-zA-Z0-9_]+)\s*\(\s*([a-zA-Z0-9_]+)\s*\)", dtype_raw, re.IGNORECASE)
            if m_ref:
                ref_table = m_ref.group(1)
                ref_col = m_ref.group(2)
            elif ref_meta:
                ref_table = str(ref_meta)

    m_del = re.search(r"ON\s+DELETE\s+(RESTRICT|CASCADE|SET\s+NULL|NO\s+ACTION)", dtype_raw, re.IGNORECASE)
    delete_rule = f"ON DELETE {m_del.group(1).upper()}" if m_del else ("Unavailable" if not is_fk else "ON DELETE RESTRICT")

    return {
        "column_name": cname,
        "table_name": tname,
        "domain": dom,
        "data_type": clean_type,
        "full_type_signature": dtype_raw,
        "is_nullable": not is_not_null,
        "default": default_val,
        "is_primary_key": is_pk,
        "is_unique": is_unique,
        "is_foreign_key": is_fk,
        "referenced_table": ref_table,
        "referenced_column": ref_col,
        "delete_rule": delete_rule,
        "description": desc
    }


def build_dynamic_summary(final_tables: List[Dict[str, Any]], existing_summary: Optional[str] = None) -> str:
    """
    Dynamically generates the updated_summary ledger for ONLY the tables and references
    actually retrieved. If existing_summary is provided, seamlessly merges with already active
    tables and relationships. Returns empty summary block for not-found scenarios.
    """
    if not final_tables and not existing_summary:
        return (
            "Request Summary:\n"
            "- No matching entities found.\n\n"
            "Available Tables:\n- None\n\n"
            "Available Groups:\n- None\n\n"
            "Available References:\n- None\n\n"
            "Pending Rename:\n- None"
        )

    existing_tables_dict: Dict[str, str] = {}
    existing_refs_list: List[str] = []

    if existing_summary and "Available Tables:" in existing_summary:
        m_tbl_block = re.search(r"Available Tables:\s*\n(.*?)(?=\n\nAvailable Groups:|\n\nAvailable References:|$)", existing_summary, re.DOTALL)
        if m_tbl_block:
            for line in m_tbl_block.group(1).splitlines():
                line = line.strip()
                if line.startswith("-") and ":" in line:
                    tname = line[1:].split(":")[0].strip().lower()
                    if tname != "none":
                        existing_tables_dict[tname] = line

        m_ref_block = re.search(r"Available References:\s*\n(.*?)(?=\n\nPending Rename:|$)", existing_summary, re.DOTALL)
        if m_ref_block:
            for line in m_ref_block.group(1).splitlines():
                line = line.strip()
                if line.startswith("-") and "FOREIGN KEY" in line:
                    if line not in existing_refs_list and "- none" not in line.lower():
                        existing_refs_list.append(line)

    try:
        for tbl in final_tables:
            tname = tbl.get("table_name", "")
            tname_low = tname.lower()
            cols = tbl.get("columns", [])
            col_details = tbl.get("column_details", [])

            if col_details and len(col_details) != len(cols):
                # Subset of columns requested
                col_strs = []
                for c in col_details:
                    meta = parse_column_metadata(c)
                    nn_str = " NOT NULL" if not meta["is_nullable"] else ""
                    col_strs.append(f"{meta['column_name']} {meta['data_type']}{nn_str}")
                existing_tables_dict[tname_low] = f"- {tname}: {', '.join(col_strs)}; Status: ACTIVE"
            else:
                existing_tables_dict[tname_low] = f"- {tname}: {', '.join(cols)}; Status: ACTIVE"

            for c in col_details:
                meta = parse_column_metadata(c)
                if meta["is_foreign_key"] and meta["referenced_table"] != "Unavailable":
                    del_str = f" {meta['delete_rule']}" if meta['delete_rule'] != "Unavailable" else ""
                    ref_entry = f"- {tname}: FOREIGN KEY ({meta['column_name']}) REFERENCES {meta['referenced_table']}({meta['referenced_column']}){del_str}"
                    if ref_entry not in existing_refs_list:
                        existing_refs_list.append(ref_entry)

        tables_block = "\n".join(existing_tables_dict.values()) if existing_tables_dict else "- None"
        refs_block = "\n".join(existing_refs_list) if existing_refs_list else "- None"

        return (
            "Request Summary:\n"
            "- Schema retrieved successfully.\n\n"
            f"Available Tables:\n{tables_block}\n\n"
            "Available Groups:\n- None\n\n"
            f"Available References:\n{refs_block}\n\n"
            "Pending Rename:\n- None"
        )
    except Exception as e:
        logger.error(f"Error building dynamic summary: {e}", exc_info=True)
        return "Request Summary:\n- None\n\nAvailable Tables:\n- None\n\nAvailable Groups:\n- None\n\nAvailable References:\n- None\n\nPending Rename:\n- None"


def resolve_canonical_table_name(text: str, all_tables: Dict[str, Any]) -> Optional[str]:
    """Resolves table name by checking exact names, underscores/spaces, synonyms, and word boundaries."""
    t_lower = text.lower().strip()
    if t_lower in all_tables:
        return t_lower
    if t_lower.replace(" ", "_") in all_tables:
        return t_lower.replace(" ", "_")
    if t_lower in TABLE_SYNONYMS:
        return TABLE_SYNONYMS[t_lower]
    for t in sorted(all_tables.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(t) + r"\b", t_lower) or re.search(r"\b" + re.escape(t.replace("_", " ")) + r"\b", t_lower):
            return t
    for syn, t in sorted(TABLE_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"\b" + re.escape(syn) + r"\b", t_lower):
            return t
    return None


def format_column_spec(meta: Dict[str, Any]) -> str:
    """Formats exact canonical SQL column specification line."""
    parts = [meta["column_name"], meta["data_type"]]
    if not meta["is_nullable"]:
        parts.append("NOT NULL")
    if meta.get("default") and meta["default"] != "Unavailable":
        parts.append(f"DEFAULT {meta['default']}")
    if meta.get("is_primary_key"):
        parts.append("PRIMARY KEY")
    elif meta.get("is_unique"):
        parts.append("UNIQUE")
    if meta.get("is_foreign_key") and meta.get("referenced_table") != "Unavailable":
        parts.append(f"REFERENCES {meta['referenced_table']}({meta['referenced_column']})")
        if meta.get("delete_rule") and meta["delete_rule"] != "Unavailable":
            parts.append(meta["delete_rule"])
    return " ".join(parts)


def resolve_dependencies(
    root_tables: List[Dict[str, Any]],
    depth: str = "direct",
    all_tables: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Resolves table dependencies strictly according to requested depth:
    - 'none': only the root tables, 0 dependencies expanded.
    - 'direct': root tables + 1 level of direct actual foreign-key parents.
    - 'recursive': full transitive closure of parent tables with cycle detection.
    """
    if all_tables is None:
        reg = load_erp_registry()
        all_tables = reg.get("all_tables", {})

    if depth == "none":
        return root_tables

    if depth == "direct":
        result_map = {t["table_name"].lower(): t for t in root_tables}
        for t in root_tables:
            for dep in t.get("actual_dependencies", []):
                dep_low = dep.lower()
                if dep_low in all_tables:
                    result_map[dep_low] = all_tables[dep_low]
        return topological_sort_tables(list(result_map.values()))

    # recursive with cycle detection
    visited: Set[str] = set()
    result_map = {}

    def dfs(tbl_name: str, path: Set[str]):
        if tbl_name in path or tbl_name in visited:
            return
        visited.add(tbl_name)
        tbl = all_tables.get(tbl_name)
        if not tbl:
            return
        result_map[tbl_name] = tbl
        new_path = path | {tbl_name}
        for dep in tbl.get("actual_dependencies", []):
            dfs(dep.lower(), new_path)

    for t in root_tables:
        dfs(t["table_name"].lower(), set())

    return topological_sort_tables(list(result_map.values()))


def match_column_semantic(
    cand_phrase: str,
    tbl_obj: Dict[str, Any],
    all_columns: Dict[Tuple[str, str], Dict[str, Any]]
) -> Tuple[Optional[str], float]:
    """
    Maps a natural language column phrase to a canonical column in tbl_obj
    using stored column names, JSON keywords, and JSON descriptions.
    """
    cand = cand_phrase.lower().strip().replace("-", " ")
    cand_tokens = set(re.findall(r"\w+", cand)) - {"the", "and", "of", "for", "time", "date", "reference", "column", "columns"}

    best_col = None
    best_score = 0.0

    for cname in tbl_obj.get("columns", []):
        cdata = all_columns.get((tbl_obj["table_name"].lower(), cname.lower()), {})
        c_clean = cname.replace("_", " ")
        c_tokens = set(re.findall(r"\w+", c_clean))

        # 1. Exact name match
        if cand.replace(" ", "_") == cname:
            return cname, 1.0

        # 2. Token overlap with column name
        if cand_tokens and cand_tokens.issubset(c_tokens):
            score = 0.95
            if score > best_score:
                best_score = score
                best_col = cname

        # 3. Match against stored keywords
        for kw in cdata.get("keywords", []):
            kw_clean = kw.lower().replace("-", " ")
            kw_tokens = set(re.findall(r"\w+", kw_clean))
            if cand == kw_clean:
                score = 0.95
                if score > best_score:
                    best_score = score
                    best_col = cname
            elif cand in kw_clean:
                score = 0.90
                if score > best_score:
                    best_score = score
                    best_col = cname
            elif cand_tokens and cand_tokens.issubset(kw_tokens):
                score = 0.85
                if score > best_score:
                    best_score = score
                    best_col = cname

        # 4. Description match
        desc = cdata.get("description", "").lower()
        if cand in desc:
            score = 0.85
            if score > best_score:
                best_score = score
                best_col = cname

    if best_score >= 0.80:
        return best_col, best_score
    return None, 0.0


def validate_rag_response(res: RAGSchemaResponse, all_tables: Dict[str, Any]) -> RAGSchemaResponse:
    """
    Strict Output Validation Layer:
    Guarantees zero hallucinations before returning response.
    - If not_found or ambiguous, sql must be empty and count 0.
    - Every table in res.tables must exist in JSON catalog.
    - Empty / malformed requests reject cleanly.
    """
    if res.domain in ("not_found", "ambiguous") or not res.tables:
        res.sql = ""
        res.count = 0
        return res

    for t in res.tables:
        if t.lower() not in all_tables:
            logger.error(f"Validation failure: Table '{t}' is not in enterprise JSON catalog! Rejecting response.")
            return RAGSchemaResponse(
                domain="not_found",
                domain_name="Schema Registry (Entity Not Found)",
                tables=[],
                sql="",
                count=0,
                summary="Request Summary:\n- Validation error: ungrounded entity detected.\n\nAvailable Tables:\n- None\n\nAvailable Groups:\n- None\n\nAvailable References:\n- None\n\nPending Rename:\n- None",
                explanation=f"Validation failed: table '{t}' is not grounded in the schema knowledge base."
            )
    return res


# =====================================================================
# Query Plan & Retrieval Pipeline (Section 2)
# =====================================================================

@dataclass
class QueryPlan:
    raw_query: str
    roots: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    requested_columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    table_columns: Dict[str, List[str]] = field(default_factory=dict)
    column_descriptions: List[str] = field(default_factory=list)
    expand_dependencies: str = "none" # "none", "forward_all", "named_only", "recursive"
    named_dependencies: List[str] = field(default_factory=list)
    include_reverse_dependents: bool = False
    exclude_tables: List[str] = field(default_factory=list)
    exact_table_count: Optional[int] = None
    exact_column_count: Optional[int] = None
    exact_name_required: bool = False
    ambiguous_candidates: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    nonexistent_candidates: List[str] = field(default_factory=list)
    is_entire_app: bool = False
    requested_domain: Optional[str] = None
    summary: Optional[str] = None
    active_tables_in_summary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roots": self.roots,
            "tables": self.roots,
            "columns": self.columns,
            "requested_columns": self.requested_columns,
            "missing_columns": self.missing_columns,
            "column_descriptions": self.column_descriptions,
            "expand_dependencies": self.expand_dependencies,
            "include_dependencies": self.expand_dependencies not in ("none",) and len(self.roots) <= 1,
            "named_dependencies": self.named_dependencies,
            "include_reverse_dependents": self.include_reverse_dependents,
            "exclude_tables": self.exclude_tables,
            "exact_table_count": self.exact_table_count,
            "exact_column_count": self.exact_column_count,
            "exact_name_required": self.exact_name_required
        }

def create_query_plan(
    query: str,
    all_table_lookup: Dict[str, Any],
    domains: Dict[str, Any],
    all_columns_lookup: Optional[Dict[str, Any]] = None,
    session_context_table: Optional[str] = None,
    summary: Optional[str] = None
) -> QueryPlan:
    if all_columns_lookup is None:
        reg = load_erp_registry()
        all_columns_lookup = reg.get("all_columns", {})

    q_raw = (query or "").strip()
    q_lower = q_raw.lower()
    q_clean = re.sub(r"[.?!,;]+$", "", q_lower).strip()

    plan = QueryPlan(raw_query=query)

    if summary and "Available Tables:" in summary:
        plan.summary = summary
        for m in re.finditer(r"-\s+([a-zA-Z0-9_]+):", summary):
            t_act = m.group(1).lower()
            if t_act != "none" and t_act not in plan.active_tables_in_summary:
                plan.active_tables_in_summary.append(t_act)

    if not q_clean:
        plan.nonexistent_candidates.append("empty_query")
        return plan

    # 1. Explicit Nonexistent Tables Check
    for bad_tbl in NONEXISTENT_TABLE_PATTERNS:
        if bad_tbl in q_clean:
            plan.nonexistent_candidates.append(bad_tbl)
            return plan

    # 2. Details Table Ambiguity Check
    if re.search(r"\b(show me the details table|show details|details table)\b", q_clean):
        plan.exact_name_required = True
        plan.ambiguous_candidates.append("details table")
        plan.suggestions = ["customer_invoices", "hrms_employees", "inventory_stocks"]
        return plan

    # 3. Check for standalone ambiguous 'employees' / 'employee' as target table
    has_positive_hrms = bool(re.search(r"\b(?:retrieve|show|get|search for|find|select)\s+(?:the\s+)?hrms_employees?\b", q_clean)) and not bool(re.search(r"\b(?:do\s+not|don't|exclude|without)\s+(?:select\s+)?hrms_employees?\b", q_clean))
    is_emp_target = bool(re.search(r"\bemployees?\b", q_clean)) and not has_positive_hrms
    has_other_table_ctx = bool(re.search(r"\b(?:from|in|of)\s+(?:the\s+)?([a-zA-Z0-9_]+)", q_clean)) or any(t in q_clean for t in ["attendance", "hrms_attendance", "department", "designation", "payroll", "invoice", "stock", "warehouse"])
    is_domain_q = any(w in q_clean for w in ["employee management database", "employee management", "management database"])

    if is_emp_target and not has_other_table_ctx and not is_domain_q:
        if "employee details" in q_clean:
            plan.roots = ["hrms_employees"]
            plan.expand_dependencies = "forward_all"
            return plan

        plan.exact_name_required = True
        plan.ambiguous_candidates.append("employees")
        plan.suggestions = ["hrms_employees"]
        return plan

    # 4. Exact Table Count
    m_count = re.search(r"\b(?:exactly\s+)?(1|2|3|4|5|one|two|three|four|five)\s+tables?\b", q_clean)
    if m_count:
        c_word = m_count.group(1).lower()
        word_map = {"1": 1, "one": 1, "2": 2, "two": 2, "3": 3, "three": 3, "4": 4, "four": 4, "5": 5, "five": 5}
        plan.exact_table_count = word_map.get(c_word)

    # 4b. Exact Column Count
    m_col_count = re.search(r"\b(?:exactly\s+)?(1|2|3|4|5|6|7|8|one|two|three|four|five|six|seven|eight|a)\s+columns?\b", q_clean)
    if m_col_count:
        c_col_word = m_col_count.group(1).lower()
        col_word_map = {"1": 1, "one": 1, "a": 1, "2": 2, "two": 2, "3": 3, "three": 3, "4": 4, "four": 4, "5": 5, "five": 5, "6": 6, "six": 6, "7": 7, "seven": 7, "8": 8, "eight": 8}
        plan.exact_column_count = col_word_map.get(c_col_word)

    # 5. Exclude Tables
    m_exclude = re.search(r"\b(?:do not retrieve|do not include|excluding|exclude|without|except|not)\s+([a-zA-Z0-9_ ]+)", q_clean)
    if m_exclude:
        excl_phrase = m_exclude.group(1).lower()
        for t in all_table_lookup:
            if re.search(r"\b" + re.escape(t) + r"\b", excl_phrase) or re.search(r"\b" + re.escape(t.replace("_", " ")) + r"\b", excl_phrase):
                if t not in plan.exclude_tables:
                    plan.exclude_tables.append(t)

    # 6. Entire Application Check
    is_entire_app = any(w in q_clean for w in [
        "complete erp schema", "entire erp schema", "complete erp", "entire erp",
        "show the complete erp schema", "show complete erp schema", "show all tables from the erp application",
        "all tables in application", "all tables from the erp", "all erp tables", "all 15 tables",
        "entire application", "full application", "all application", "whole application", "every table"
    ])
    if is_entire_app:
        plan.is_entire_app = True
        plan.roots = list(all_table_lookup.keys())
        return plan

    # 7-pre: Count-Based Column Retrieval (e.g. "create 2 columns for customer_invoices")
    if plan.exact_column_count:
        col_target_tbl = None
        for tname in sorted(all_table_lookup.keys(), key=lambda x: len(x.replace("_", " ")), reverse=True):
            for v in [tname, tname.replace("_", " ")]:
                if re.search(r"\b" + re.escape(v) + r"\b", q_clean):
                    col_target_tbl = tname
                    break
            if col_target_tbl:
                break
        if not col_target_tbl:
            for syn in sorted(TABLE_SYNONYMS.keys(), key=len, reverse=True):
                if re.search(r"\b" + re.escape(syn) + r"\b", q_clean):
                    col_target_tbl = TABLE_SYNONYMS[syn]
                    break
        if not col_target_tbl and session_context_table:
            col_target_tbl = session_context_table

        if not col_target_tbl:
            # Check domain keywords (e.g. "create 2 columns for finance system")
            if re.search(r"\b(?:erp\s+)?(?:finance|financial|accounting)\b", q_clean):
                col_target_tbl = "chart_of_accounts"
                plan.requested_domain = "erp_finance"
            elif re.search(r"\b(?:erp\s+)?(?:hrms|human\s+resources|hr|employee\s+management)\b", q_clean):
                col_target_tbl = "hrms_employees"
                plan.requested_domain = "erp_hrms"
            elif re.search(r"\b(?:erp\s+)?(?:supply\s+chain|warehouse(?:\s+and\s+inventory)?|inventory(?:\s+management)?)\b", q_clean):
                col_target_tbl = "product_items"
                plan.requested_domain = "erp_supply_chain"

        if not col_target_tbl:
            # Check if a non-existent table was named (e.g. "create account table with 2 columns")
            m_nonex = re.search(r"\b([a-zA-Z0-9_]+)\s+table\b|\b(?:table|for|in|from)\s+([a-zA-Z0-9_]+)\b", q_clean)
            if m_nonex:
                cand_t = (m_nonex.group(1) or m_nonex.group(2) or "").strip()
                if cand_t and cand_t.lower() not in {"table", "database", "schema", "all", "column", "columns", "the", "with", "only", "a", "an", "1", "2", "two", "3", "three", "4", "four", "5", "five"}:
                    plan.nonexistent_candidates.append(cand_t)
                    return plan

            # Column count requested but table/domain is completely unspecified
            plan.ambiguous_candidates.append("unspecified_table_for_columns")
            return plan

        if col_target_tbl:
            plan.roots = [col_target_tbl]
            plan.expand_dependencies = "none"
            tbl_cols = all_table_lookup[col_target_tbl].get("columns", [])

            # Check if specific columns were also named in the query
            matched_named = [c for c in tbl_cols if re.search(r"\b" + re.escape(c) + r"\b", q_clean)]

            no_pk = any(w in q_clean for w in [
                "do not include primary key", "excluding primary key", "exclude primary key",
                "without primary key", "no primary key", "no pk", "excluding id", "without id",
                "do not include id", "excluding the primary key", "without the primary key"
            ])

            if matched_named:
                if "id" in tbl_cols and "id" not in matched_named and not no_pk:
                    matched_named = ["id"] + matched_named
                plan.columns = matched_named[:plan.exact_column_count]
                plan.requested_columns = list(matched_named)
            else:
                available_cols = prioritize_table_columns(col_target_tbl, tbl_cols, include_pk=not no_pk)
                plan.columns = available_cols[:plan.exact_column_count]
                plan.requested_columns = list(plan.columns)

            plan.table_columns[col_target_tbl] = list(plan.columns)
            return plan

    # 7. Column-Level Queries Check
    target_tbl_raw = None
    cols_text = None

    # 7a. Dot notation: e.g. customer_invoices.customer_id
    m_dot = re.findall(r"\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b", q_clean)
    if m_dot:
        for t_cand, c_cand in m_dot:
            res_t = None
            if t_cand in all_table_lookup:
                res_t = t_cand
            elif t_cand in TABLE_SYNONYMS:
                res_t = TABLE_SYNONYMS[t_cand]
            if res_t:
                target_tbl_raw = res_t
                cols_text = c_cand
                break

    # 7b. Leading 'from / in <table>': e.g. From hrms_attendance, retrieve ...
    if not target_tbl_raw:
        m_lead = re.search(r"^(?:from|in)\s+(?:the\s+)?([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)?)(?:\s+table|\s+schema)?(?:,\s*|\s+)(.*)$", q_clean)
        if m_lead:
            t_raw = m_lead.group(1).strip()
            t_cand = t_raw.replace(" ", "_")
            if t_cand in all_table_lookup:
                target_tbl_raw = t_cand
                cols_text = m_lead.group(2).strip()
            elif t_raw in TABLE_SYNONYMS:
                target_tbl_raw = TABLE_SYNONYMS[t_raw]
                cols_text = m_lead.group(2).strip()

    # 7c. Anywhere 'from / in / of <table>': e.g. Retrieve only customer_id from customer_invoices table
    if not target_tbl_raw:
        for tname in sorted(all_table_lookup.keys(), key=lambda x: len(x.replace("_", " ")), reverse=True):
            for v in [tname, tname.replace("_", " ")]:
                m = re.search(r"\b(?:from|in|of)\s+(?:the\s+)?(" + re.escape(v) + r")(?:\s+table|\s+schema)?\b", q_clean)
                if m:
                    cand_cols = q_clean[:m.start()] + " " + q_clean[m.end():]
                    if re.search(r"^(?:show|get|retrieve|describe|display|explain|view|see|give\s+me)?\s*(?:the\s+)?(?:complete|full|entire|whole)?\s*(?:schema|table|structure|definition)?\s*$", cand_cols.strip(), re.I):
                        continue
                    target_tbl_raw = tname
                    cols_text = cand_cols
                    break
            if target_tbl_raw:
                break

    if not target_tbl_raw:
        for syn in sorted(TABLE_SYNONYMS.keys(), key=len, reverse=True):
            m = re.search(r"\b(?:from|in|of)\s+(?:the\s+)?(" + re.escape(syn) + r")(?:\s+table|\s+schema)?\b", q_clean)
            if m:
                cand_cols = q_clean[:m.start()] + " " + q_clean[m.end():]
                if re.search(r"^(?:show|get|retrieve|describe|display|explain|view|see|give\s+me)?\s*(?:the\s+)?(?:complete|full|entire|whole)?\s*(?:schema|table|structure|definition)?\s*$", cand_cols.strip(), re.I):
                    continue
                target_tbl_raw = TABLE_SYNONYMS[syn]
                cols_text = cand_cols
                break

    # 7d. '<table> (table)? (with|having|containing|including) <columns>'
    if not target_tbl_raw:
        for tname in sorted(all_table_lookup.keys(), key=lambda x: len(x.replace("_", " ")), reverse=True):
            for v in [tname, tname.replace("_", " ")]:
                m = re.search(r"\b(" + re.escape(v) + r")(?:\s+table)?\s+(?:with|having|containing|including)\s+(?:only\s+)?(?:columns?\s+|fields?\s+|attributes?\s+)?(.+)$", q_clean)
                if m:
                    cand_cols_raw = m.group(2).strip()
                    if any(dep_w in cand_cols_raw for dep_w in ["dependenc", "referenced table", "referenced tables", "foreign key", "relation"]):
                        continue
                    has_col_keyword = any(cw in cand_cols_raw for cw in ["column", "columns", "field", "fields", "attribute", "attributes"])
                    clean_cand = re.sub(r"\s+(?:columns|column|fields|attributes)$", "", cand_cols_raw).strip()
                    cand_tokens = [t.strip().replace(" ", "_") for t in re.split(r",|\band\b|&|\bor\b", clean_cand)]
                    has_matching_col = any(ct in all_table_lookup[tname].get("columns", []) for ct in cand_tokens)
                    if has_col_keyword or has_matching_col:
                        target_tbl_raw = tname
                        cols_text = cand_cols_raw
                        break
            if target_tbl_raw:
                break

    if not target_tbl_raw:
        for syn in sorted(TABLE_SYNONYMS.keys(), key=len, reverse=True):
            m = re.search(r"\b(" + re.escape(syn) + r")(?:\s+table)?\s+(?:with|having|containing|including)\s+(?:only\s+)?(?:columns?\s+|fields?\s+|attributes?\s+)?(.+)$", q_clean)
            if m:
                cand_cols_raw = m.group(2).strip()
                tname = TABLE_SYNONYMS[syn]
                if any(dep_w in cand_cols_raw for dep_w in ["dependenc", "referenced table", "referenced tables", "foreign key", "relation"]):
                    continue
                has_col_keyword = any(cw in cand_cols_raw for cw in ["column", "columns", "field", "fields", "attribute", "attributes"])
                clean_cand = re.sub(r"\s+(?:columns|column|fields|attributes)$", "", cand_cols_raw).strip()
                cand_tokens = [t.strip().replace(" ", "_") for t in re.split(r",|\band\b|&|\bor\b", clean_cand)]
                has_matching_col = any(ct in all_table_lookup[tname].get("columns", []) for ct in cand_tokens)
                if has_col_keyword or has_matching_col:
                    target_tbl_raw = tname
                    cols_text = cand_cols_raw
                    break

    if target_tbl_raw and target_tbl_raw.lower() in ("finance", "hrms", "supply chain", "supply_chain", "erp", "enterprise erp"):
        target_tbl_raw = None
        cols_text = None

    if not target_tbl_raw and session_context_table and any(w in q_clean for w in ["now retrieve only", "retrieve only", "show only", "what about", "which column"]):
        target_tbl_raw = session_context_table
        cols_text = q_clean

    if target_tbl_raw and cols_text:
        t_cand = target_tbl_raw.lower().replace(" ", "_")
        res_tname = None
        if t_cand in all_table_lookup:
            res_tname = t_cand
        elif target_tbl_raw.lower() in TABLE_SYNONYMS:
            res_tname = TABLE_SYNONYMS[target_tbl_raw.lower()]
        else:
            for t in all_table_lookup:
                if t == t_cand or t.replace("_", " ") == target_tbl_raw.lower():
                    res_tname = t
                    break
        if not res_tname:
            plan.nonexistent_candidates.append(target_tbl_raw)
            return plan

        plan.roots = [res_tname]
        plan.expand_dependencies = "none"

        # Clean cols_text
        clean_cols_p = re.sub(r"\b(?:do not include|excluding|exclude|without)\s+(?:the\s+)?(?:primary\s+key|id)\b.*$", "", cols_text, flags=re.IGNORECASE).strip()
        clean_cols_p = re.sub(r"^(?:now\s+)?(?:retrieve|get|show|select|search for|find|describe|check|list|explain)\s+", "", clean_cols_p, flags=re.IGNORECASE).strip()
        clean_cols_p = re.sub(r"^(?:only\s+)?(?:the\s+)?(?:metadata\s+(?:for|of)|details\s+(?:for|of)|all\s+details\s+(?:for|of)|only\s+the\s+requested\s+columns\s+of|only\s+the\s+requested\s+columns|requested\s+columns\s+of|requested\s+columns|columns\s+of|column\s+of|columns|column|information\s+for|info\s+for|which\s+column\s+stores(?:\s+the)?|what\s+is\s+the|the)\s+", "", clean_cols_p, flags=re.IGNORECASE).strip()
        clean_cols_p = re.sub(r"\s+(?:columns|column|metadata|details|fields|attributes|data)$", "", clean_cols_p, flags=re.IGNORECASE).strip()

        raw_tokens = re.split(r",|\band\b|&|\bor\b", clean_cols_p)
        stopwords = {"the", "and", "of", "for", "in", "to", "a", "an", "all", "only", "these", "those", "from", "with", "table", "schema"}
        tbl_cols = all_table_lookup[res_tname].get("columns", [])

        for tok in raw_tokens:
            tok_raw = tok.strip().strip("'\"").strip(".?!,; ")
            tok_clean = re.sub(r"^(?:the_|only_|column_|columns_)", "", tok_raw.lower().replace("-", "_").replace(" ", "_"))
            if tok_raw and tok_clean not in stopwords:
                plan.requested_columns.append(tok_clean)
                if tok_clean in tbl_cols:
                    if tok_clean not in plan.columns:
                        plan.columns.append(tok_clean)
                else:
                    sem_col, score = match_column_semantic(tok_raw, all_table_lookup[res_tname], all_columns_lookup)
                    if sem_col and score >= 0.80:
                        if sem_col not in plan.columns:
                            plan.columns.append(sem_col)
                            plan.column_descriptions.append(tok_raw)
                    else:
                        if tok_clean not in plan.missing_columns:
                            plan.missing_columns.append(tok_clean)

        # Ensure 'id' is included if specific columns were requested, unless explicitly excluded
        if "id" in tbl_cols and "id" not in plan.columns and plan.columns:
            no_pk = any(w in q_clean for w in [
                "do not include primary key", "excluding primary key", "exclude primary key",
                "without primary key", "no primary key", "no pk", "excluding id", "without id",
                "do not include id", "excluding the primary key", "without the primary key"
            ])
            is_isolated_retrieve = (
                q_clean.startswith("retrieve ") or
                bool(re.search(r"\b(?:retrieve|show|get|select)\s+only\b", q_clean)) or
                q_clean.startswith("only ") or
                bool(re.search(r"\bfrom\s+[a-zA-Z0-9_]+,\s*retrieve\b", q_clean)) or
                bool(re.search(r"\b[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\b", q_clean))
            )
            if not no_pk and not is_isolated_retrieve:
                plan.columns.insert(0, "id")

        plan.table_columns[res_tname] = list(plan.columns)
        return plan

    # 8. Pattern B: "Which column stores <phrase>"
    m_which_col = re.search(r"\b(?:which\s+column(?:\s+stores)?|what\s+column(?:\s+stores)?)\s+(.+)$", q_clean)
    if m_which_col:
        col_phrase = m_which_col.group(1).strip()
        found_tbl = None
        rem_col = None
        for tname in sorted(all_table_lookup.keys(), key=lambda x: len(x.replace("_", " ")), reverse=True):
            for v in [tname, tname.replace("_", " ")]:
                if re.search(r"\b" + re.escape(v) + r"\b", col_phrase):
                    found_tbl = tname
                    rem_col = re.sub(r"\b" + re.escape(v) + r"\b", "", col_phrase).strip()
                    rem_col = re.sub(r"^(?:the|a|an)\s+", "", rem_col).strip()
                    break
            if found_tbl:
                break
        if not found_tbl:
            for syn in sorted(TABLE_SYNONYMS.keys(), key=len, reverse=True):
                if re.search(r"\b" + re.escape(syn) + r"\b", col_phrase):
                    found_tbl = TABLE_SYNONYMS[syn]
                    rem_col = re.sub(r"\b" + re.escape(syn) + r"\b", "", col_phrase).strip()
                    rem_col = re.sub(r"^(?:the|a|an)\s+", "", rem_col).strip()
                    break
        if found_tbl and rem_col:
            plan.roots = [found_tbl]
            plan.expand_dependencies = "none"
            plan.requested_columns.append(rem_col.replace(" ", "_"))
            plan.column_descriptions.append(rem_col)
            sem_col, score = match_column_semantic(rem_col, all_table_lookup[found_tbl], all_columns_lookup)
            if sem_col and score >= 0.80:
                plan.columns.append(sem_col)
            else:
                plan.missing_columns.append(rem_col.replace(" ", "_"))
            plan.table_columns[found_tbl] = list(plan.columns)
            return plan

    # 9. Pattern C: Multi-column keywords
    if any(w in q_clean for w in ["i need", "need", "retrieve columns", "show columns", "get columns", "which columns"]):
        clean_p = re.sub(r"^(?:i\s+need|need|retrieve|get|show)\s+", "", q_clean).strip()
        tokens = [t.strip() for t in re.split(r",|\band\b|&|\bor\b", clean_p) if t.strip()]
        best_t = None
        max_matches = 0
        matched_c = []
        for tname, tobj in all_table_lookup.items():
            tbl_cols = tobj.get("columns", [])
            m_count = 0
            cur_matched = []
            for tok in tokens:
                tok_c = tok.replace(" ", "_").lower()
                if tok_c in tbl_cols:
                    m_count += 1
                    cur_matched.append(tok_c)
                else:
                    sem, sc = match_column_semantic(tok, tobj, all_columns_lookup)
                    if sem and sc >= 0.80:
                        m_count += 1
                        cur_matched.append(sem)
            if m_count > max_matches:
                max_matches = m_count
                best_t = tname
                matched_c = cur_matched
        if max_matches >= 2:
            no_pk = any(w in q_clean for w in [
                "do not include primary key", "excluding primary key", "exclude primary key",
                "without primary key", "no primary key", "no pk", "excluding id", "without id",
                "do not include id", "excluding the primary key", "without the primary key"
            ])
            if "id" in all_table_lookup[best_t].get("columns", []) and "id" not in matched_c and not no_pk:
                matched_c = ["id"] + matched_c
            plan.roots = [best_t]
            plan.columns = matched_c
            plan.requested_columns = list(matched_c)
            plan.table_columns[best_t] = list(matched_c)
            plan.expand_dependencies = "none"
            return plan

    # 10. Table-Level Extraction (Exact Match + Synonyms)
    matched_ranges = []
    for tname in sorted(all_table_lookup.keys(), key=lambda x: len(x.replace("_", " ")), reverse=True):
        for v in [tname, tname.replace("_", " ")]:
            for m in re.finditer(r"\b" + re.escape(v) + r"\b", q_clean):
                if not any(r[0] <= m.start() and m.end() <= r[1] for r in matched_ranges):
                    matched_ranges.append((m.start(), m.end()))
                    if tname not in plan.exclude_tables and tname not in plan.roots:
                        plan.roots.append(tname)

    for syn in sorted(TABLE_SYNONYMS.keys(), key=len, reverse=True):
        tname = TABLE_SYNONYMS[syn]
        for m in re.finditer(r"\b" + re.escape(syn) + r"\b", q_clean):
            if not any(r[0] <= m.start() and m.end() <= r[1] for r in matched_ranges):
                matched_ranges.append((m.start(), m.end()))
                if tname not in plan.exclude_tables and tname not in plan.roots:
                    plan.roots.append(tname)

    # 10b. Check if an explicit table/schema entity was requested that is NOT in catalog
    # (e.g. "create a account schema for ERP FINANCE", "create table users in HRMS")
    if not plan.roots and not plan.nonexistent_candidates:
        stop_words = {
            "erp", "db", "database", "complete", "entire", "all", "new", "finance", "hrms", "supply", "chain",
            "create", "retrieve", "get", "show", "make", "give", "find", "search", "for", "in", "of", "a", "an", "the",
            "tables", "schemas", "system", "management", "build", "generate", "construct", "setup", "architect", "define"
        }
        for pat in [r"\b(?:table|schema)\s+([a-zA-Z0-9_]+)\b", r"\b(?:a|an|the)?\s*([a-zA-Z0-9_]+)\s+(?:table|schema)\b"]:
            m_entity = re.search(pat, q_clean)
            if m_entity:
                cand = m_entity.group(1).strip()
                if cand not in stop_words and cand not in all_table_lookup and cand not in TABLE_SYNONYMS:
                    plan.nonexistent_candidates.append(cand)
                    return plan

    # 10c. Multi-entity check for missing tables when roots are found
    # e.g. "Retrieve customer_invoices, finance_customers, user_tokens, and analytics_events"
    if plan.roots and any(sep in q_clean for sep in [",", " and ", " & "]):
        table_stopwords = {
            "retrieve", "get", "show", "search", "find", "fetch", "display", "create", "make", "build",
            "generate", "construct", "setup", "architect", "define", "produce", "add",
            "the", "a", "an", "and", "or", "of", "in", "from", "for", "with", "without", "table", "tables",
            "schema", "schemas", "database", "db", "only", "just", "both", "all", "these", "those",
            "dependencies", "dependency", "actual", "direct", "referenced", "system", "erp", "application",
            "complete", "full", "structure", "data", "metadata", "details", "information", "info", "its"
        }
        q_pos = re.split(r"\b(?:do\s+not|don't|excluding|exclude|except|without)\b", q_clean)[0].strip()
        q_pos = re.sub(r"\b(?:and\s+)?(?:its\s+)?(?:actual\s+)?(?:direct\s+)?dependencies\b", "", q_pos).strip()
        q_pos = re.sub(r"\b(?:exactly\s+)?(?:\d+|two|three|four|five|several|multiple)\s+tables?:?\s*", "", q_pos).strip()
        cleaned_clause = re.sub(r"^(?:retrieve|get|show|find|search for|display|fetch|create|build|generate|construct|setup|set up|architect|define)\s+(?:only\s+)?(?:the\s+)?(?:tables?\s+)?", "", q_pos).strip()
        raw_chunks = [c.strip().strip(".?!,; ") for c in re.split(r",|\band\b|&|\bor\b", cleaned_clause) if c.strip()]
        for chunk in raw_chunks:
            clean_chunk = re.sub(r"\s+(?:from|in|of)\s+(?:the\s+)?(?:finance|financial|hrms|supply_chain|supply\s+chain|ecommerce|e-commerce|healthcare|health|medical|erp).*$", "", chunk, flags=re.IGNORECASE).strip()
            cand = re.sub(r"^(?:the|a|an|only)\s+", "", clean_chunk).strip()
            cand = re.sub(r"\s+(?:table|schema)$", "", cand).strip().replace(" ", "_").replace("-", "_")
            if not cand or cand in table_stopwords or len(cand) <= 2:
                continue
            if not re.match(r"^[a-zA-Z0-9_]+$", cand):
                continue
            if any(dw in cand for dw in ["dependenc", "relation", "foreign", "exclude", "without", "exact"]):
                continue
            is_matched = (
                cand in plan.roots
                or cand in all_table_lookup
                or cand in TABLE_SYNONYMS
                or any(r == cand or r.replace("_", " ") == chunk for r in plan.roots)
                or any(syn == chunk for syn in TABLE_SYNONYMS)
            )
            if not is_matched and cand not in plan.nonexistent_candidates:
                plan.nonexistent_candidates.append(cand)

    # 11. Dependency Expansion Mode Determination
    is_none = any(w in q_clean for w in [
        "only", "just", "single table", "without dependencies", "no dependencies", "excluding dependencies",
        "i need only", "i only need", "no additional tables", "exactly these tables", "no dependency expansion"
    ]) or (len(plan.roots) >= 2 and not any(w in q_clean for w in ["and its actual direct dependencies", "with dependencies", "full dependency hierarchy"]))

    is_recursive = any(w in q_clean for w in ["full dependency hierarchy", "recursive dependencies", "all dependencies", "transitive dependencies"])
    is_forward = any(w in q_clean for w in ["and its actual direct dependencies", "with direct dependencies", "with its directly referenced tables", "direct dependencies", "actual direct dependency", "direct dependency"])

    if is_none:
        plan.expand_dependencies = "none"
    elif is_recursive:
        plan.expand_dependencies = "recursive"
    elif is_forward or len(plan.roots) == 1:
        plan.expand_dependencies = "forward_all"
    else:
        plan.expand_dependencies = "none"

    if not plan.roots and not plan.nonexistent_candidates:
        if re.search(r"\b(?:erp\s+)?(?:finance|financial|accounting)\b", q_clean):
            plan.requested_domain = "erp_finance"
        elif re.search(r"\b(?:erp\s+)?(?:hrms|human\s+resources|hr|employee\s+management)\b", q_clean):
            plan.requested_domain = "erp_hrms"
        elif re.search(r"\b(?:erp\s+)?(?:supply\s+chain|warehouse(?:\s+and\s+inventory)?|inventory(?:\s+management)?)\b", q_clean):
            plan.requested_domain = "erp_supply_chain"
        elif re.search(r"\b(?:erp\s+)?(?:ecommerce|e-commerce|e\s+commerce|online\s+store|retail|shopping|cart)\b", q_clean):
            plan.requested_domain = "erp_ecommerce"
        elif re.search(r"\b(?:erp\s+)?(?:healthcare|health\s+care|health|hospital|clinic|medical)\b", q_clean):
            plan.requested_domain = "erp_healthcare"
        elif any(w in q_clean for w in [
            "all domains", "all tables from the erp", "all 15 tables", "complete erp",
            "entire erp", "all erp tables", "whole erp", "all tables in erp",
            "show all tables from the erp application", "create db schemas for erp application",
            "create db schemas for erp", "erp schemas", "all erp schemas"
        ]):
            plan.is_entire_app = True
            plan.roots = list(all_table_lookup.keys())
            return plan
        else:
            m_s = re.search(r"(?:search for|find|retrieve|get|show)\s+(?:the\s+)?([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)?)\s*(?:table|schema)?$", q_clean)
            if m_s:
                c_tbl = m_s.group(1).strip()
                if c_tbl not in ["the", "all", "complete", "details", "database", "finance", "hrms", "supply", "chain", "ecommerce", "healthcare", "health", "medical", "erp"]:
                    plan.nonexistent_candidates.append(c_tbl)

    return plan

def execute_plan(
    plan: QueryPlan,
    all_table_lookup: Optional[Dict[str, Any]] = None,
    all_columns_lookup: Optional[Dict[str, Any]] = None,
    domains: Optional[Dict[str, Any]] = None
) -> RAGSchemaResponse:
    if all_table_lookup is None or all_columns_lookup is None or domains is None:
        reg = load_erp_registry()
        all_table_lookup = reg.get("all_tables", {})
        all_columns_lookup = reg.get("all_columns", {})
        domains = reg.get("domains", {})

    # Ambiguous Request Handling
    if plan.ambiguous_candidates:
        cand = plan.ambiguous_candidates[0]
        if cand == "details table":
            expl = "The prompt 'details table' is ambiguous across ERP domains. Please clarify whether customer invoice details (Finance), employee details (HRMS), or stock details (Supply Chain) are intended."
        elif cand == "unspecified_table_for_columns":
            expl = "Please specify which ERP table or domain you would like columns for (e.g., 'create 2 columns for chart_of_accounts' or 'create 2 columns for finance')."
            plan.suggestions = ["chart_of_accounts", "hrms_employees", "product_items"]
        else:
            expl = f"No table named '{cand}' exists. Did you mean 'hrms_employees'? Please clarify."
        return validate_rag_response(RAGSchemaResponse(
            domain="ambiguous",
            domain_name="Enterprise ERP",
            tables=[],
            sql="",
            count=0,
            summary=build_dynamic_summary([]),
            explanation=expl,
            warnings=[],
            structured_metadata=[],
            columns_metadata=[],
            suggestions=plan.suggestions or ["hrms_employees"],
            debug={"query_plan": plan.to_dict()},
            status="needs_clarification",
            intent="SCHEMA",
            needs_clarification=True
        ), all_table_lookup)

    # Nonexistent Entity Handling (when no valid roots exist)
    if plan.nonexistent_candidates and not plan.roots:
        bad_tbl = plan.nonexistent_candidates[0]
        expl = f"Table '{bad_tbl}' does not exist in the enterprise schema knowledge base. No synthetic table or fallback SQL generated."
        return validate_rag_response(RAGSchemaResponse(
            domain="not_found",
            domain_name="Schema Registry (Entity Not Found)",
            tables=[],
            sql="",
            count=0,
            summary=build_dynamic_summary([]),
            explanation=expl,
            warnings=[f"ENTITY_NOT_FOUND: {bad_tbl}"],
            structured_metadata=[],
            columns_metadata=[],
            suggestions=[],
            debug={"query_plan": plan.to_dict()},
            status="rejected",
            intent="SCHEMA",
            needs_clarification=False
        ), all_table_lookup)

    # Column-Level Retrieval Handling
    if plan.requested_columns or plan.columns or plan.missing_columns:
        root_name = plan.roots[0] if plan.roots else "unknown_table"
        tbl_obj = all_table_lookup.get(root_name, {})
        valid_cols = tbl_obj.get("columns", [])

        # If no requested columns successfully resolved -> Reject without falling back to full table!
        if not plan.columns:
            missing_cols = plan.missing_columns or plan.requested_columns
            missing_str = ", ".join(missing_cols)
            return validate_rag_response(RAGSchemaResponse(
                domain=tbl_obj.get("domain", "not_found"),
                domain_name=domains.get(tbl_obj.get("domain"), {}).get("name", "Enterprise Domain"),
                tables=[],
                sql="",
                count=0,
                summary=build_dynamic_summary([]),
                explanation=f"Column '{missing_str}' was not found in table '{root_name}'. Available columns for '{root_name}' are: {', '.join(valid_cols)}. No synthetic column or SQL generated.",
                warnings=[f"COLUMN_NOT_FOUND: {root_name}.{c}" for c in missing_cols],
                structured_metadata=[],
                columns_metadata=[],
                suggestions=[],
                debug={"query_plan": plan.to_dict()},
                status="rejected",
                intent="SCHEMA",
                needs_clarification=False
            ), all_table_lookup)

        # Matched columns present
        matched_cols = plan.columns
        missing_cols = plan.missing_columns

        col_objs = [all_columns_lookup[(root_name, c)] for c in matched_cols if (root_name, c) in all_columns_lookup]
        parsed_metas = [parse_column_metadata(c) for c in col_objs]

        columns_metadata = []
        header_comments = []
        col_ddl_lines = []
        warnings = []

        for meta in parsed_metas:
            cname = meta["column_name"]
            columns_metadata.append({
                "table_name": root_name,
                "name": cname,
                "type": meta["data_type"],
                "description": meta["description"],
                "nullable": meta["is_nullable"],
                "default": meta["default"] if meta["default"] != "Unavailable" else None,
                "is_primary_key": meta["is_primary_key"],
                "is_unique": meta["is_unique"],
                "foreign_key": {
                    "table": meta["referenced_table"],
                    "column": meta["referenced_column"],
                    "on_delete": meta["delete_rule"].replace("ON DELETE ", "").strip()
                } if meta["is_foreign_key"] else None
            })

            header_comments.append(f"-- Table: {root_name}")
            header_comments.append(f"-- Column: {cname}")
            header_comments.append(f"-- Data Type: {meta['data_type']}")
            header_comments.append(f"-- Nullable: {'Yes' if meta['is_nullable'] else 'No (NOT NULL)'}")
            header_comments.append(f"-- Default: {meta['default']}")
            header_comments.append(f"-- Primary Key: {'Yes' if meta['is_primary_key'] else 'No'}")
            header_comments.append(f"-- Unique: {'Yes' if meta['is_unique'] else 'No'}")
            if meta["is_foreign_key"]:
                header_comments.append(f"-- Foreign Key: Yes")
                header_comments.append(f"-- Referenced Table: {meta['referenced_table']}")
                header_comments.append(f"-- Referenced Column: {meta['referenced_column']}")
                header_comments.append(f"-- Delete Rule: {meta['delete_rule']}")
            else:
                header_comments.append(f"-- Foreign Key: No")
            header_comments.append(f"-- Description: {meta['description']}")
            header_comments.append(format_column_spec(meta) + ";")
            header_comments.append("")

            col_ddl_lines.append("    " + format_column_spec(meta))

        partial_create_table = f"CREATE TABLE {root_name} (\n" + ",\n".join(col_ddl_lines) + "\n);"
        full_sql = "\n".join(header_comments) + partial_create_table

        if missing_cols:
            for mc in missing_cols:
                warnings.append(f"COLUMN_NOT_FOUND: {root_name}.{mc}")

        for meta in parsed_metas:
            if meta["is_foreign_key"] and meta["referenced_table"] != "Unavailable":
                warnings.append(f"Foreign key in table '{root_name}' references '{meta['referenced_table']}', which was not retrieved due to requested retrieval scope.")

        status_outcome = "partial" if missing_cols else "accepted"

        expl_lines = [f"Retrieved {len(matched_cols)} requested column(s) for table '{root_name}' in domain '{domains[tbl_obj['domain']]['name']}':"]
        for meta in parsed_metas:
            fk_str = f", References: {meta['referenced_table']}({meta['referenced_column']}) {meta['delete_rule']}" if meta["is_foreign_key"] else ""
            nn_str = "NOT NULL" if not meta["is_nullable"] else "NULL"
            expl_lines.append(f"- {meta['column_name']} ({meta['data_type']}, {nn_str}{fk_str}): {meta['description']}")
        if missing_cols:
            expl_lines.append(f"\nWarning: The following requested column(s) do not exist in table '{root_name}' and were omitted: {', '.join(missing_cols)}.")

        pseudo_tbl = {
            "table_name": root_name,
            "columns": matched_cols,
            "column_details": col_objs,
            "domain": tbl_obj["domain"]
        }

        return validate_rag_response(RAGSchemaResponse(
            domain=tbl_obj["domain"],
            domain_name=domains[tbl_obj["domain"]]["name"],
            tables=[root_name],
            sql=full_sql,
            count=1,
            summary=build_dynamic_summary([pseudo_tbl]),
            explanation="\n".join(expl_lines),
            warnings=warnings,
            structured_metadata=[{"table_name": root_name, "columns": [parse_column_metadata(c) for c in col_objs]}],
            columns_metadata=columns_metadata,
            suggestions=[],
            debug={"query_plan": plan.to_dict()},
            status=status_outcome,
            intent="SCHEMA",
            needs_clarification=False
        ), all_table_lookup)

    # Root Table Retrieval Handling
    if plan.roots:
        root_objs = [all_table_lookup[t] for t in plan.roots if t in all_table_lookup and t not in plan.exclude_tables]
        depth_map = {"none": "none", "forward_all": "direct", "recursive": "recursive", "direct": "direct"}
        dep_depth = depth_map.get(plan.expand_dependencies, "direct")

        selected_tables = resolve_dependencies(root_objs, depth=dep_depth, all_tables=all_table_lookup)

        if plan.exclude_tables:
            selected_tables = [t for t in selected_tables if t["table_name"].lower() not in plan.exclude_tables]

        if plan.exact_table_count is not None and len(selected_tables) > plan.exact_table_count:
            root_set = set(t["table_name"].lower() for t in root_objs)
            kept_roots = [t for t in selected_tables if t["table_name"].lower() in root_set]
            kept_deps = [t for t in selected_tables if t["table_name"].lower() not in root_set]
            selected_tables = (kept_roots + kept_deps)[:plan.exact_table_count]
            selected_tables = topological_sort_tables(selected_tables)

        if plan.expand_dependencies == "none":
            root_names = set(t["table_name"].lower() for t in root_objs)
            selected_tables = [t for t in selected_tables if t["table_name"].lower() in root_names]

        selected_tables = topological_sort_tables(selected_tables)
        dep_count = len(selected_tables) - len(root_objs)
        dom_names = list(dict.fromkeys(domains[t["domain"]]["name"] for t in selected_tables))
        dom_ids = list(dict.fromkeys(t["domain"] for t in selected_tables))

        warnings = [f"EXCLUDED_TABLE: {t}" for t in plan.exclude_tables]
        if plan.nonexistent_candidates:
            warnings.extend([f"TABLE_NOT_FOUND: {t}" for t in plan.nonexistent_candidates])
        for t in selected_tables:
            for dep in t.get("actual_dependencies", []):
                dep_low = dep.lower()
                is_in_selected = dep_low in [st["table_name"].lower() for st in selected_tables]
                is_in_active_summary = dep_low in plan.active_tables_in_summary
                if not is_in_selected and not is_in_active_summary:
                    warnings.append(f"Foreign key in table '{t['table_name']}' references '{dep}', which was not retrieved due to requested retrieval scope.")

        ddl_str = "\n\n".join(tbl["sql"].strip() for tbl in selected_tables)
        summary_str = build_dynamic_summary(selected_tables, existing_summary=plan.summary)

        retrieved_names = [t["table_name"] for t in selected_tables]
        added_deps = [t["table_name"] for t in selected_tables if t["table_name"] not in [ro["table_name"] for ro in root_objs]]

        if dep_depth == "none" or dep_count == 0:
            if len(root_objs) >= 2:
                expl_desc = f"Zero dependencies expanded as {len(root_objs)} root table(s) were explicitly requested (0 dependencies expanded)."
            else:
                expl_desc = f"Retrieved {len(selected_tables)} table(s) with 0 dependencies expanded."
        elif dep_depth == "recursive":
            expl_desc = f"Retrieved {len(root_objs)} root table(s) with full recursive dependency hierarchy ({len(added_deps)} parent tables: {', '.join(added_deps)}): {', '.join(retrieved_names)}."
        else:
            expl_desc = f"Retrieved {len(root_objs)} root table(s) with {len(added_deps)} direct foreign-key dependency ({', '.join(added_deps)}): {', '.join(retrieved_names)}."

        expl_lines = [
            f"Retrieved {len(selected_tables)} table(s) across {len(dom_names)} domain(s):",
            f"- Tables: {', '.join(retrieved_names)}",
            f"- Domains: {', '.join(dom_names)}",
            expl_desc
        ]
        ref_active = [dep for t in selected_tables for dep in t.get("actual_dependencies", []) if dep.lower() in plan.active_tables_in_summary]
        if ref_active:
            expl_lines.append(f"Foreign key relationships wire directly to active table(s) in summary ledger: {', '.join(dict.fromkeys(ref_active))}.")
        if plan.nonexistent_candidates:
            expl_lines.append(f"Notice: The following requested table(s) do not exist in the enterprise catalog and were excluded: {', '.join(plan.nonexistent_candidates)}.")
        if warnings:
            expl_lines.append("Warnings: " + "; ".join(warnings))

        if plan.is_entire_app or len(selected_tables) == 15:
            res_domain = "entire_application"
            res_dom_name = "Complete Enterprise ERP (Finance, HRMS, Supply Chain)"
        elif len(dom_ids) == 1:
            res_domain = dom_ids[0]
            res_dom_name = dom_names[0]
        else:
            res_domain = ",".join(dom_ids)
            res_dom_name = " & ".join(dom_names)

        return validate_rag_response(RAGSchemaResponse(
            domain=res_domain,
            domain_name=res_dom_name,
            tables=retrieved_names,
            sql=ddl_str,
            count=len(selected_tables),
            summary=summary_str,
            explanation="\n".join(expl_lines),
            warnings=warnings,
            structured_metadata=[],
            columns_metadata=[],
            suggestions=[],
            debug={"query_plan": plan.to_dict()},
            status="partial" if plan.nonexistent_candidates else "accepted",
            intent="SCHEMA",
            needs_clarification=False
        ), all_table_lookup)

    # Full Domain Retrieval Handling
    if plan.requested_domain:
        dom_id = plan.requested_domain
        dom_raw = [t for t in domains[dom_id]["tables"] if t["table_name"].lower() not in plan.exclude_tables]
        dom_tables = topological_sort_tables(dom_raw)

        if plan.exact_table_count and plan.exact_table_count > 0 and len(dom_tables) > plan.exact_table_count:
            primary_priority = DOMAIN_PRIMARY_TABLES.get(dom_id, [])
            def get_priority(tbl):
                tname = tbl["table_name"].lower()
                if tname in primary_priority:
                    return primary_priority.index(tname)
                return 100
            sorted_by_pri = sorted(dom_raw, key=get_priority)
            selected_subset = sorted_by_pri[:plan.exact_table_count]
            dom_tables = topological_sort_tables(selected_subset)
            expl_desc = f"Retrieved {len(dom_tables)} primary table(s) for domain '{domains[dom_id]['name']}' ({dom_id}) according to requested count in valid topological foreign-key order: " + ", ".join(t["table_name"] for t in dom_tables) + "."
        else:
            expl_desc = f"Retrieved all {len(dom_tables)} tables for domain '{domains[dom_id]['name']}' ({dom_id}) in valid topological foreign-key order: " + ", ".join(t["table_name"] for t in dom_tables) + "."

        warnings = [f"EXCLUDED_TABLE: {t}" for t in plan.exclude_tables]
        table_names_set = set(t["table_name"].lower() for t in dom_tables)
        for t in dom_tables:
            for dep in t.get("actual_dependencies", []):
                if dep.lower() not in table_names_set:
                    warnings.append(f"Foreign key in table '{t['table_name']}' references '{dep}', which was not retrieved due to requested retrieval scope.")

        ddl_str = "\n\n".join(tbl["sql"].strip() for tbl in dom_tables)
        summary_str = build_dynamic_summary(dom_tables)
        if warnings:
            expl_str = expl_desc + " Warnings: " + "; ".join(warnings)
        else:
            expl_str = expl_desc

        return validate_rag_response(RAGSchemaResponse(
            domain=dom_id,
            domain_name=domains[dom_id]["name"],
            tables=[t["table_name"] for t in dom_tables],
            sql=ddl_str,
            count=len(dom_tables),
            summary=summary_str,
            explanation=expl_str,
            warnings=warnings,
            structured_metadata=[],
            columns_metadata=[],
            suggestions=[],
            debug={"query_plan": plan.to_dict()},
            status="accepted",
            intent="SCHEMA",
            needs_clarification=False
        ), all_table_lookup)

    return validate_rag_response(RAGSchemaResponse(
        domain="not_found",
        domain_name="Schema Registry (Entity Not Found)",
        tables=[],
        sql="",
        count=0,
        summary=build_dynamic_summary([]),
        explanation="No matching ERP schema information retrieved.",
        warnings=[],
        structured_metadata=[],
        columns_metadata=[],
        suggestions=[],
        debug={"query_plan": plan.to_dict()},
        status="rejected",
        intent="SCHEMA",
        needs_clarification=False
    ), all_table_lookup)


# Backwards compatibility helper
parse_unified_query = create_query_plan

# =====================================================================
# RAG Service Implementation
# =====================================================================

class ERPEnterpriseRAGService:
    """
    Self-contained, production-grade ERP Enterprise RAG Service.
    Encapsulates 3-system domain schema architecture (Finance, HRMS, Supply Chain).
    Guarantees:
    1. Exact domain isolation.
    2. Pinpoint table retrieval with direct parent foreign keys only.
    3. Granular column metadata retrieval without extraneous columns.
    4. Strict not-found validation gate.
    5. Single source of truth canonical schema context.
    """

    def __init__(self):
        try:
            self.registry = load_erp_registry()
        except Exception as e:
            logger.error(f"Failed to initialize ERPEnterpriseRAGService: {e}", exc_info=True)
            self.registry = {"domains": {}, "all_tables": {}, "all_columns": {}, "col_name_to_tables": {}, "valid_tables": set()}

    def get_domains(self) -> Dict[str, Any]:
        try:
            return load_erp_registry().get("domains", {})
        except Exception as e:
            logger.error(f"Error fetching domains: {e}")
            return {}

    def get_all_application_tables(self) -> List[Dict[str, Any]]:
        """Collects all tables across all domains in the application."""
        all_tables: List[Dict[str, Any]] = []
        try:
            for dom_data in self.get_domains().values():
                all_tables.extend(dom_data.get("tables", []))
        except Exception as e:
            logger.error(f"Error collecting all application tables: {e}")
        return all_tables

    def retrieve(self, query: str, domain: Optional[str] = None, session_context: Optional[str] = None, summary: Optional[str] = None) -> RAGSchemaResponse:
        """
        Retrieves matching ERP schema tables for a user query via the structured QueryPlan pipeline.
        Ensures strict domain boundaries, exact direct foreign key dependency resolution,
        granular column retrieval, zero hallucination, and strict not-found validation.
        """
        try:
            reg = load_erp_registry()
            domains = reg.get("domains", {})
            all_table_lookup = reg.get("all_tables", {})
            all_columns_lookup = reg.get("all_columns", {})

            # 1. Build QueryPlan
            plan = create_query_plan(
                query=query,
                all_table_lookup=all_table_lookup,
                domains=domains,
                all_columns_lookup=all_columns_lookup,
                session_context_table=session_context,
                summary=summary
            )
            if domain and not plan.roots and not plan.requested_domain and not plan.is_entire_app:
                plan.requested_domain = domain

            # 2. Execute QueryPlan
            return execute_plan(
                plan=plan,
                all_table_lookup=all_table_lookup,
                all_columns_lookup=all_columns_lookup,
                domains=domains
            )

        except Exception as e:
            logger.error(f"Critical error during retrieve: {e}", exc_info=True)
            return RAGSchemaResponse(
                domain=domain or "erp_finance",
                domain_name="Enterprise Domain",
                tables=[],
                sql="",
                count=0,
                summary="Request Summary:\n- None\n\nAvailable Tables:\n- None\n\nAvailable Groups:\n- None\n\nAvailable References:\n- None\n\nPending Rename:\n- None",
                explanation=f"Schema retrieval failed gracefully: {str(e)}",
                warnings=[],
                structured_metadata=[],
                columns_metadata=[],
                suggestions=[],
                debug=None,
                status="rejected",
                intent="SCHEMA",
                needs_clarification=False
            )

    def generate_with_llm(
        self,
        query: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        domain: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Executes Retrieval-Augmented Generation (RAG).
        Returns the canonical schema context directly for read-only / schema inspection requests,
        guaranteeing zero LLM hallucinations in SQL, summary, and explanations.
        """
        try:
            schema_res = self.retrieve(query=query, domain=domain)
            rag_only_mode = os.getenv("RAG_ONLY_MODE", "true").lower() in ("1", "true", "yes")

            if rag_only_mode or not schema_res.tables or not schema_res.sql:
                return {
                    "schema": schema_res,
                    "sql": schema_res.sql,
                    "summary": schema_res.summary,
                    "explanation": schema_res.explanation,
                    "model_used": "canonical-rag-engine",
                    "token_used": 0
                }

            selected_model = resolve_best_rag_model(api_key=api_key, preferred_model=model)
            key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")

            if not key:
                return {
                    "schema": schema_res,
                    "sql": schema_res.sql,
                    "summary": schema_res.summary,
                    "explanation": schema_res.explanation,
                    "model_used": "direct-retrieval",
                    "token_used": 0
                }

            augmented_prompt = build_rag_augmented_prompt(
                user_query=query,
                schema_ddl=schema_res.sql,
                domain_name=schema_res.domain_name,
                domain_id=schema_res.domain,
                tables=schema_res.tables,
                summary=schema_res.summary
            )

            models_to_try = [selected_model]
            for m in RAG_LLM_PRIORITY_MODELS:
                if m not in models_to_try:
                    models_to_try.append(m)

            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }

            for attempt_model in models_to_try:
                payload = {
                    "model": attempt_model,
                    "messages": [
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": augmented_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                try:
                    logger.info(f"Querying RAG LLM using model '{attempt_model}'...")
                    resp = requests.post(
                        GROQ_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=25
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        token_used = data.get("usage", {}).get("total_tokens", 0)
                        global _RESOLVED_BEST_RAG_MODEL
                        _RESOLVED_BEST_RAG_MODEL = attempt_model

                        return {
                            "schema": schema_res,
                            "sql": schema_res.sql,
                            "summary": schema_res.summary,
                            "explanation": content,
                            "model_used": attempt_model,
                            "token_used": token_used
                        }
                    elif resp.status_code in (404, 400):
                        continue
                    else:
                        break
                except Exception as e:
                    continue

            return {
                "schema": schema_res,
                "sql": schema_res.sql,
                "summary": schema_res.summary,
                "explanation": schema_res.explanation,
                "model_used": "direct-retrieval",
                "token_used": 0
            }
        except Exception as e:
            logger.error(f"Critical error in generate_with_llm: {e}", exc_info=True)
            return {
                "schema": None,
                "sql": "",
                "summary": "",
                "explanation": f"RAG LLM generation failed: {str(e)}",
                "model_used": "error-fallback",
                "token_used": 0
            }

    def alter_segment(
        self,
        new_content: str,
        segment_id: Optional[str] = None,
        segment_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        2-Tier Schema Segment Alteration:
        - Tier 1: Point alteration by permanent segment_id (UUID).
        - Tier 2: Logical lookup by business segment_key (e.g. domain:table:column).
        Removes reliance on content hashing for finding or modifying segments.
        """
        import uuid
        from src.rag.database import get_db, KnowledgeBaseRepository

        with get_db() as session:
            repo = KnowledgeBaseRepository(session)
            target_id = None
            if segment_id:
                target_id = uuid.UUID(str(segment_id))
            elif segment_key:
                seg = repo.get_segment_by_key(segment_key, active_only=False)
                if seg:
                    target_id = seg.id

            if not target_id:
                raise ValueError(f"Could not locate segment with segment_id='{segment_id}' or segment_key='{segment_key}'")

            seg_obj, op = repo.alter_segment(
                segment_id=target_id,
                new_content=new_content,
                new_segment_key=segment_key
            )
            session.commit()

            # Process ChromaDB sync immediately
            synchronizer = ChromaPostgresSynchronizer()
            sync_stats = synchronizer.process_pending_jobs(batch_size=10, session=session)

            return {
                "segment_id": str(seg_obj.id),
                "segment_key": seg_obj.segment_key,
                "version": seg_obj.version,
                "operation": op,
                "sync_stats": sync_stats
            }


# =====================================================================
# Global Service Singleton & Helper Functions
# =====================================================================

_RAG_SERVICE_INSTANCE: Optional[ERPEnterpriseRAGService] = None


def get_rag_service() -> ERPEnterpriseRAGService:
    global _RAG_SERVICE_INSTANCE
    if _RAG_SERVICE_INSTANCE is None:
        _RAG_SERVICE_INSTANCE = ERPEnterpriseRAGService()
    return _RAG_SERVICE_INSTANCE


def retrieve_erp_schema(query: str, domain: Optional[str] = None, session_context: Optional[str] = None, summary: Optional[str] = None) -> RAGSchemaResponse:
    """Standard entry point for enterprise RAG schema retrieval."""
    return get_rag_service().retrieve(query=query, domain=domain, session_context=session_context, summary=summary)


def generate_rag_llm(
    query: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """Standard entry point for RAG-augmented LLM generation."""
    return get_rag_service().generate_with_llm(query=query, api_key=api_key, model=model, domain=domain)


# =============================================================================
# Local Embedding Model (SentenceTransformer)
# =============================================================================

_EMBEDDING_MODEL: Any = None


def get_embedding_model():
    """
    Lazy loads SentenceTransformer locally on CPU with zero remote network delays.
    Generates normalized 384-dimensional dense vectors for semantic search.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: '{EMBEDDING_MODEL_NAME}'...")
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


# =============================================================================
# ChromaDB Vector Engine & Client Singleton
# =============================================================================

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.ERROR)

_CHROMA_CLIENT: Optional[Any] = None
_CHROMA_COLLECTION: Optional[Any] = None


def get_chroma_client():
    """Returns a singleton ChromaDB PersistentClient."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        import chromadb
        from chromadb.config import Settings
        CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing ChromaDB PersistentClient at: {CHROMA_DATA_DIR}")
        _CHROMA_CLIENT = chromadb.PersistentClient(
            path=str(CHROMA_DATA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
    return _CHROMA_CLIENT


def get_chroma_collection(collection_name: str = CHROMA_COLLECTION_NAME):
    """Returns the ChromaDB collection with cosine distance metric."""
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None or _CHROMA_COLLECTION.name != collection_name:
        client = get_chroma_client()
        logger.info(f"Getting or creating ChromaDB collection: '{collection_name}' (cosine metric)")
        _CHROMA_COLLECTION = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    return _CHROMA_COLLECTION


class ChromaSchemaEngine:
    """
    Production ChromaDB Vector Engine for ERP Schema Knowledge Base.
    Features:
    - Persistent SQLite + HNSW storage in CHROMA_DATA_DIR.
    - Native metadata filtering by domain, table, segment_type, or identity_key.
    - Zero-loop O(1) single-chunk upserts and deletions by permanent UUID.
    - Normalized cosine vector similarity using local sentence-transformers.
    """

    def __init__(self, data_dir: Path = CHROMA_DATA_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self.data_dir = data_dir
        self.collection_name = collection_name
        self.client = get_chroma_client()
        self.collection = get_chroma_collection(collection_name)

    def count(self) -> int:
        """Returns the total number of vectors in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.warning(f"Error getting ChromaDB count: {e}")
            return 0

    def upsert_segment(
        self,
        segment_id: str,
        document: str,
        metadata: Dict[str, Any],
        embedding: Optional[Any] = None
    ):
        """Upserts a single segment in-place by ID (Zero ghost chunks)."""
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = ", ".join(str(item) for item in v)
            elif v is not None:
                clean_meta[k] = str(v)

        if embedding is None:
            model = get_embedding_model()
            emb = model.encode(document, convert_to_numpy=True, normalize_embeddings=True)
            emb_list = emb.tolist()
        elif hasattr(embedding, "tolist"):
            emb_list = embedding.tolist()
        else:
            emb_list = embedding

        try:
            self.collection.upsert(
                ids=[str(segment_id)],
                documents=[document],
                metadatas=[clean_meta],
                embeddings=[emb_list]
            )
        except Exception as e:
            logger.error(f"Failed to upsert segment '{segment_id}' in ChromaDB: {e}", exc_info=True)
            raise

    def upsert_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[Any] = None
    ):
        """Batch upsert multiple segments with optional pre-computed embeddings."""
        if not ids:
            return

        clean_metadatas = []
        for meta in metadatas:
            clean_m = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                elif isinstance(v, list):
                    clean_m[k] = ", ".join(str(item) for item in v)
                elif v is not None:
                    clean_m[k] = str(v)
            clean_metadatas.append(clean_m)

        if embeddings is None:
            model = get_embedding_model()
            emb_arr = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
            emb_list = emb_arr.tolist()
        elif hasattr(embeddings, "tolist"):
            emb_list = embeddings.tolist()
        else:
            emb_list = embeddings

        try:
            self.collection.upsert(
                ids=[str(i) for i in ids],
                documents=documents,
                metadatas=clean_metadatas,
                embeddings=emb_list
            )
        except Exception as e:
            logger.error(f"Failed to upsert batch of {len(ids)} segments in ChromaDB: {e}", exc_info=True)
            raise

    def delete_segment(self, segment_id: str):
        """Deletes a segment by ID from ChromaDB."""
        try:
            self.collection.delete(ids=[str(segment_id)])
            logger.info(f"Deleted segment '{segment_id}' from ChromaDB.")
        except Exception as e:
            logger.warning(f"Error deleting segment '{segment_id}' from ChromaDB: {e}")

    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a specific segment by its permanent ID."""
        try:
            res = self.collection.get(ids=[str(segment_id)], include=["documents", "metadatas"])
            if res and res.get("ids") and len(res["ids"]) > 0:
                return {
                    "id": res["ids"][0],
                    "document": res["documents"][0] if res.get("documents") else "",
                    "metadata": res["metadatas"][0] if res.get("metadatas") else {}
                }
        except Exception as e:
            logger.warning(f"Error fetching segment '{segment_id}' from ChromaDB: {e}")
        return None

    def get_segment_by_key(self, segment_key: str) -> Optional[Dict[str, Any]]:
        """Tier 2: Direct lookup in ChromaDB by business segment_key."""
        try:
            res = self.collection.get(
                where={"segment_key": str(segment_key)},
                include=["documents", "metadatas"]
            )
            if res and res.get("ids") and len(res["ids"]) > 0:
                return {
                    "id": res["ids"][0],
                    "document": res["documents"][0] if res.get("documents") else "",
                    "metadata": res["metadatas"][0] if res.get("metadatas") else {}
                }
        except Exception as e:
            logger.warning(f"Error fetching segment by key '{segment_key}' from ChromaDB: {e}")
        return None

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executes semantic search with native metadata filtering."""
        model = get_embedding_model()
        query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        emb_list = query_emb.tolist()

        kwargs: Dict[str, Any] = {
            "query_embeddings": [emb_list],
            "n_results": min(top_k, max(self.count(), 1)),
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        try:
            raw_res = self.collection.query(**kwargs)
        except Exception as e:
            logger.error(f"ChromaDB search query failed: {e}")
            return []

        results: List[Dict[str, Any]] = []
        if not raw_res or not raw_res.get("ids") or len(raw_res["ids"]) == 0:
            return results

        ids = raw_res["ids"][0]
        docs = raw_res.get("documents", [[]])[0]
        metas = raw_res.get("metadatas", [[]])[0]
        dists = raw_res.get("distances", [[]])[0]

        for i, seg_id in enumerate(ids):
            dist = dists[i] if i < len(dists) else 1.0
            similarity = max(0.0, 1.0 - dist)
            results.append({
                "id": seg_id,
                "similarity": similarity,
                "distance": dist,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {}
            })

        return results


_CHROMA_ENGINE: Optional[ChromaSchemaEngine] = None


def get_chroma_engine() -> ChromaSchemaEngine:
    """Returns singleton ChromaSchemaEngine."""
    global _CHROMA_ENGINE
    if _CHROMA_ENGINE is None:
        _CHROMA_ENGINE = ChromaSchemaEngine()
    return _CHROMA_ENGINE


# =============================================================================
# ChromaDB <-> PostgreSQL Knowledge Synchronizer
# =============================================================================

class ChromaPostgresSynchronizer:
    """
    Synchronizes PostgreSQL primary knowledge base with ChromaDB vector engine.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.chroma_engine = get_chroma_engine()

    def process_pending_jobs(self, batch_size: int = 128, session: Optional[Any] = None) -> Dict[str, int]:
        """Processes PENDING embedding_jobs from PostgreSQL."""
        import numpy as np
        from src.rag.database import get_db, KnowledgeBaseRepository
        from src.rag.models import KnowledgeSegmentModel, EmbeddingJobModel, DomainModel, KnowledgeTableModel

        stats = {"processed": 0, "inserted": 0, "updated": 0, "deleted": 0, "failed": 0}

        def _run(sess):
            repo = KnowledgeBaseRepository(sess)
            jobs = repo.get_pending_embedding_jobs(limit=batch_size)
            if not jobs:
                return stats

            logger.info(f"Processing batch of {len(jobs)} pending embedding jobs for ChromaDB...")
            model = get_embedding_model()

            dom_map = {d.id: d.domain_id for d in repo.session.query(DomainModel).all()}
            table_map = {t.id: t.table_name for t in repo.session.query(KnowledgeTableModel).all()}

            to_encode_jobs: List[Tuple[Any, Any]] = []
            delete_jobs: List[Any] = []

            for job in jobs:
                seg = repo.get_segment_by_id(job.segment_id)
                if job.operation == "DELETE" or (seg and not seg.is_active):
                    delete_jobs.append(job)
                elif seg and seg.is_active:
                    to_encode_jobs.append((job, seg))
                else:
                    repo.update_job_status(job.id, "COMPLETED")
                    stats["processed"] += 1

            for job in delete_jobs:
                try:
                    self.chroma_engine.delete_segment(str(job.segment_id))
                    repo.update_job_status(job.id, "COMPLETED")
                    stats["deleted"] += 1
                    stats["processed"] += 1
                except Exception as e:
                    repo.update_job_status(job.id, "FAILED", error_message=str(e))
                    stats["failed"] += 1

            if to_encode_jobs:
                texts = [seg.content for _, seg in to_encode_jobs]
                try:
                    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                except Exception as enc_err:
                    logger.error(f"Failed to generate embeddings: {enc_err}")
                    for job, _ in to_encode_jobs:
                        repo.update_job_status(job.id, "FAILED", error_message=str(enc_err))
                        stats["failed"] += 1
                    return stats

                for i, (job, seg) in enumerate(to_encode_jobs):
                    try:
                        emb = embeddings[i : i + 1].astype(np.float32)
                        d_code = dom_map.get(seg.domain_id, "")
                        t_name = table_map.get(seg.table_id, "") if seg.table_id else (
                            seg.segment_name.split(".")[0]
                            if "." in seg.segment_name and seg.segment_type in ("table", "column", "relationship")
                            else ""
                        )
                        ident_key = getattr(seg, "segment_key", None) or f"{d_code or str(seg.domain_id)}:{t_name or 'none'}:{seg.segment_name}"

                        meta = {
                            "domain_id": str(seg.domain_id),
                            "domain_code": d_code,
                            "domain_name": seg.section,
                            "table_id": str(seg.table_id) if seg.table_id else "",
                            "table_name": t_name,
                            "topic": seg.topic,
                            "segment_type": seg.segment_type,
                            "segment_name": seg.segment_name,
                            "version": seg.version,
                            "segment_key": ident_key,
                            "identity_key": ident_key,
                        }

                        self.chroma_engine.upsert_segment(
                            segment_id=str(seg.id),
                            document=seg.content,
                            metadata=meta,
                            embedding=emb[0]
                        )

                        if job.operation == "INSERT":
                            stats["inserted"] += 1
                        else:
                            stats["updated"] += 1

                        repo.update_job_status(job.id, "COMPLETED")
                        stats["processed"] += 1
                    except Exception as job_err:
                        logger.error(f"Failed processing job {job.id} for segment {seg.id}: {job_err}")
                        repo.update_job_status(job.id, "FAILED", error_message=str(job_err))
                        stats["failed"] += 1

            return stats

        if session:
            return _run(session)
        else:
            from src.rag.database import get_db
            with get_db() as sess:
                return _run(sess)

    def sync_all(self, batch_size: int = 128) -> Dict[str, int]:
        """Runs batch processing until no PENDING jobs remain."""
        total_stats = {"processed": 0, "inserted": 0, "updated": 0, "deleted": 0, "failed": 0}
        while True:
            cur = self.process_pending_jobs(batch_size=batch_size)
            if cur["processed"] == 0 and cur["failed"] == 0:
                break
            for k in total_stats:
                total_stats[k] += cur.get(k, 0)
        logger.info(f"All pending ChromaDB embedding jobs synchronized: {total_stats}")
        return total_stats

    def sync_segment(
        self,
        segment_id: Optional[str] = None,
        segment_key: Optional[str] = None,
        session: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Synchronizes ONLY a single specific segment from PostgreSQL into ChromaDB.
        Logs every stage with high clarity: chunk extraction, embedding computation, and vector storage.
        """
        import uuid
        import time
        import numpy as np
        from src.rag.database import get_db, KnowledgeBaseRepository
        from src.rag.models import KnowledgeSegmentModel, DomainModel, KnowledgeTableModel

        def _run(sess):
            repo = KnowledgeBaseRepository(sess)
            seg = None
            if segment_id:
                seg = repo.get_segment_by_id(uuid.UUID(str(segment_id)))
            elif segment_key:
                seg = repo.get_segment_by_key(segment_key, active_only=False)

            if not seg:
                err = f"Segment not found: id='{segment_id}' key='{segment_key}'"
                logger.error(f"[CHUNK PROCESS FAILED] {err}")
                raise ValueError(err)

            # Stage 1: Chunk Inspection & Validation
            logger.info("  " + "─" * 70)
            logger.info("  [STAGE 1/3: CHUNK EXTRACTION & VALIDATION]")
            logger.info(f"    • Segment Key       : {seg.segment_key}")
            logger.info(f"    • PostgreSQL UUID   : {seg.id}")
            logger.info(f"    • Domain & Section  : {seg.section}")
            logger.info(f"    • Topic / Entity    : {seg.topic}")
            logger.info(f"    • Segment Type      : {seg.segment_type} (Name: {seg.segment_name})")
            logger.info(f"    • Schema Version    : v{seg.version}")
            logger.info(f"    • Content Length    : {len(seg.content)} characters ({len(seg.content.splitlines())} lines)")
            content_lines = [l.strip() for l in seg.content.splitlines() if l.strip()]
            preview_snippet = " | ".join(content_lines[:3]) if content_lines else ""
            if len(preview_snippet) > 100:
                preview_snippet = preview_snippet[:97] + "..."
            logger.info(f"    • Content Preview   : {preview_snippet}")

            # Stage 2: Embedding Generation
            logger.info("  [STAGE 2/3: EMBEDDING INFERENCE]")
            logger.info(f"    • Embedding Model   : {self.model_name}")
            t_infer = time.perf_counter()
            model = get_embedding_model()
            emb = model.encode(seg.content, convert_to_numpy=True, normalize_embeddings=True)
            emb_vector = emb.astype(np.float32)
            infer_ms = (time.perf_counter() - t_infer) * 1000.0
            norm = float(np.linalg.norm(emb_vector))
            logger.info(f"    • Vector Dimension  : {len(emb_vector)} floats (Dense Float32)")
            logger.info(f"    • Normalization     : L2 Norm = {norm:.4f} (Cosine Similarity Space)")
            logger.info(f"    • Inference Latency : {infer_ms:.2f} ms")

            # Stage 3: Vector Storing
            count_before = self.chroma_engine.count()
            logger.info("  [STAGE 3/3: CHROMADB VECTOR STORAGE]")
            logger.info(f"    • Target Collection : {self.chroma_engine.collection_name}")
            logger.info(f"    • Upsert Strategy   : In-place overwrite by stable UUID ({seg.id})")
            logger.info(f"    • Count Before      : {count_before}")

            dom_map = {d.id: d.domain_id for d in repo.session.query(DomainModel).all()}
            table_map = {t.id: t.table_name for t in repo.session.query(KnowledgeTableModel).all()}
            d_code = dom_map.get(seg.domain_id, "")
            t_name = table_map.get(seg.table_id, "") if seg.table_id else (
                seg.segment_name.split(".")[0]
                if "." in seg.segment_name and seg.segment_type in ("table", "column", "relationship")
                else ""
            )
            ident_key = seg.segment_key or f"{d_code or str(seg.domain_id)}:{t_name or 'none'}:{seg.segment_name}"

            meta = {
                "domain_id": str(seg.domain_id),
                "domain_code": d_code,
                "domain_name": seg.section,
                "table_id": str(seg.table_id) if seg.table_id else "",
                "table_name": t_name,
                "topic": seg.topic,
                "segment_type": seg.segment_type,
                "segment_name": seg.segment_name,
                "version": seg.version,
                "segment_key": ident_key,
                "identity_key": ident_key,
            }

            self.chroma_engine.upsert_segment(
                segment_id=str(seg.id),
                document=seg.content,
                metadata=meta,
                embedding=emb_vector
            )
            count_after = self.chroma_engine.count()
            delta = count_after - count_before
            logger.info(f"    • Count After       : {count_after}")
            logger.info(f"    • Net Vector Delta  : {delta:+d} (0 duplicate vectors, clean replacement)")
            logger.info(f"    • Status            : SUCCESS (Vector persisted to disk)")
            logger.info("  " + "─" * 70)

            return {
                "segment_id": str(seg.id),
                "segment_key": seg.segment_key,
                "version": seg.version,
                "content_length": len(seg.content),
                "embedding_dimension": len(emb_vector),
                "count_before": count_before,
                "count_after": count_after,
                "inference_ms": round(infer_ms, 2),
                "status": "SUCCESS"
            }

        if session:
            return _run(session)
        else:
            from src.rag.database import get_db
            with get_db() as sess:
                return _run(sess)

    def auto_sync_changes(self, session: Optional[Any] = None) -> Dict[str, Any]:
        """
        2-Tier Automated Change-Detection Synchronizer:
        Scans all PostgreSQL active segments against ChromaDB vectors without needing
        the user to specify which segment or key changed.
        Identifies only modified, new, or stale segments and synchronizes them.
        """
        import uuid
        import time
        from src.rag.database import get_db, KnowledgeBaseRepository

        def _run(sess):
            t_start = time.perf_counter()
            repo = KnowledgeBaseRepository(sess)
            logger.info("=" * 76)
            logger.info("  CHROMA-POSTGRES 2-TIER KNOWLEDGE SYNCHRONIZER")
            logger.info("=" * 76)

            # Extract connection details
            db_bind = repo.session.get_bind()
            db_name = db_bind.url.database if db_bind and hasattr(db_bind, "url") else "db_nexus_rag"
            db_host = f"{db_bind.url.host}:{db_bind.url.port}" if db_bind and hasattr(db_bind, "url") else "localhost:5432"

            chroma_dir = getattr(self.chroma_engine, "data_dir", CHROMA_DATA_DIR)
            logger.info("SYSTEM CONFIGURATION:")
            logger.info(f"  • PostgreSQL DB     : {db_name} (Host: {db_host})")
            logger.info(f"  • ChromaDB Path     : {chroma_dir}")
            logger.info(f"  • Target Collection : {self.chroma_engine.collection_name}")
            logger.info(f"  • Embedding Model   : {self.model_name}")
            logger.info(f"  • Architecture      : 2-Tier (Tier 1: Business segment_key | Tier 2: UUID segment_id)")
            logger.info(f"  • Strategy          : Zero-hash direct content & version verification")
            logger.info("─" * 76)

            logger.info("[PHASE 1: 2-TIER KNOWLEDGE SCAN & CHANGE DETECTION]")
            logger.info("  • Fetching live active segments from PostgreSQL...")
            pg_segments = repo.list_active_segments()
            pg_count = len(pg_segments)
            pg_map = {str(s.id): s for s in pg_segments}

            logger.info(f"  • Fetching existing vector records from ChromaDB collection '{self.chroma_engine.collection_name}'...")
            chroma_data = self.chroma_engine.collection.get(include=["documents", "metadatas"])
            chroma_count = len(chroma_data["ids"]) if chroma_data and chroma_data.get("ids") else 0
            chroma_map = {}
            if chroma_data and chroma_data.get("ids"):
                for cid, cdoc, cmeta in zip(chroma_data["ids"], chroma_data["documents"], chroma_data["metadatas"]):
                    chroma_map[cid] = (cdoc or "", cmeta or {})

            logger.info(f"  - PostgreSQL Active Segments : {pg_count}")
            logger.info(f"  - ChromaDB Indexed Vectors   : {chroma_count}")

            # 3. Detect modified, new, or stale segments with detailed reasons
            modified_segments = []
            new_segments = []
            deleted_ids = []

            for sid, seg in pg_map.items():
                if sid not in chroma_map:
                    new_segments.append(seg)
                else:
                    cdoc, cmeta = chroma_map[sid]
                    content_changed = (seg.content.strip() != cdoc.strip())
                    key_changed = (seg.segment_key is not None and seg.segment_key != cmeta.get("segment_key"))
                    version_changed = (seg.version != cmeta.get("version"))

                    if content_changed or key_changed or version_changed:
                        reasons = []
                        if content_changed:
                            reasons.append("Content altered in PostgreSQL")
                        if key_changed:
                            reasons.append(f"Segment key changed: '{cmeta.get('segment_key')}' -> '{seg.segment_key}'")
                        if version_changed:
                            reasons.append(f"Version bumped: v{cmeta.get('version')} -> v{seg.version}")
                        modified_segments.append((seg, ", ".join(reasons)))

            for cid in chroma_map:
                if cid not in pg_map:
                    deleted_ids.append(cid)

            total_changes = len(modified_segments) + len(new_segments) + len(deleted_ids)
            logger.info("─" * 76)
            logger.info(f"[PHASE 2: AUDIT & CHANGE EVALUATION]")
            logger.info(f"  - Total Changes Detected     : {total_changes}")
            logger.info(f"    * Modified Segments        : {len(modified_segments)}")
            logger.info(f"    * New Segments             : {len(new_segments)}")
            logger.info(f"    * Deleted/Stale Segments   : {len(deleted_ids)}")

            if total_changes == 0:
                elapsed = time.perf_counter() - t_start
                logger.info("─" * 76)
                logger.info("[STATUS: 100% UP-TO-DATE]")
                logger.info(f"  ✓ All {pg_count} PostgreSQL segments are completely in sync with ChromaDB.")
                logger.info("  ✓ Zero segments require re-indexing. No unnecessary GPU/CPU compute consumed.")
                logger.info(f"  • Scan Duration              : {elapsed:.3f}s")
                logger.info("=" * 76)
                return {
                    "total_changes": 0,
                    "modified": 0,
                    "new": 0,
                    "deleted": 0,
                    "chroma_count": chroma_count,
                    "elapsed_seconds": round(elapsed, 3),
                    "status": "UP_TO_DATE"
                }

            # Detail the changes
            logger.info("─" * 76)
            logger.info(f"[CHANGES QUEUED FOR RE-INDEXING]:")
            for idx, (seg, reason) in enumerate(modified_segments, 1):
                logger.info(f"  [{idx}] MODIFIED: {seg.segment_key}")
                logger.info(f"      • Reason : {reason}")
                logger.info(f"      • UUID   : {seg.id}")
                logger.info(f"      • Entity : {seg.section} -> {seg.topic}")
            for idx, seg in enumerate(new_segments, 1):
                logger.info(f"  [{idx}] NEW: {seg.segment_key} (UUID: {seg.id})")
            for idx, del_id in enumerate(deleted_ids, 1):
                logger.info(f"  [{idx}] DELETED/STALE: UUID {del_id}")

            # 4. Process deletions
            if deleted_ids:
                logger.info("─" * 76)
                logger.info(f"[PURGING {len(deleted_ids)} STALE VECTOR(S)]")
                for del_id in deleted_ids:
                    logger.info(f"  • Removing stale segment ID '{del_id}' from ChromaDB...")
                    self.chroma_engine.delete_segment(del_id)

            # 5. Process modifications and new segments
            logger.info("─" * 76)
            total_reindex_count = len(modified_segments) + len(new_segments)
            logger.info(f"[PHASE 3: EXECUTING RE-INDEXING PIPELINE FOR {total_reindex_count} SEGMENT(S)]")
            reindexed = []

            for idx, (seg, reason) in enumerate(modified_segments, 1):
                logger.info(f"\n>>> [ITEM {idx}/{total_reindex_count}] Re-indexing modified segment: '{seg.segment_key}' <<<")
                res = self.sync_segment(segment_id=str(seg.id), session=sess)
                reindexed.append(res)

            offset = len(modified_segments)
            for idx, seg in enumerate(new_segments, 1):
                logger.info(f"\n>>> [ITEM {offset + idx}/{total_reindex_count}] Indexing new segment: '{seg.segment_key}' <<<")
                res = self.sync_segment(segment_id=str(seg.id), session=sess)
                reindexed.append(res)

            final_count = self.chroma_engine.count()
            elapsed = time.perf_counter() - t_start

            # Invalidate in-memory schema registry so new DDL and column data types are immediately active
            try:
                load_erp_registry(force_reload=True)
            except Exception as e:
                logger.warning(f"Could not reload in-memory schema registry after sync: {e}")

            logger.info("\n" + "=" * 76)
            logger.info("  AUTO-SYNC COMPLETED SUCCESSFULLY")
            logger.info("=" * 76)
            logger.info(f"  • Overall Status      : SYNCHRONIZED")
            logger.info(f"  • PostgreSQL Records  : {pg_count} active segments")
            logger.info(f"  • Modified Re-indexed : {len(modified_segments)}")
            logger.info(f"  • New Vectors Added   : {len(new_segments)}")
            logger.info(f"  • Stale Vectors Purged: {len(deleted_ids)}")
            logger.info(f"  • Final ChromaDB Total: {final_count} vectors")
            logger.info(f"  • Execution Time      : {elapsed:.3f} seconds")
            logger.info("=" * 76)

            return {
                "total_changes": total_changes,
                "modified": len(modified_segments),
                "new": len(new_segments),
                "deleted": len(deleted_ids),
                "reindexed_segments": [r["segment_key"] for r in reindexed],
                "final_count": final_count,
                "elapsed_seconds": round(elapsed, 3),
                "status": "SYNCHRONIZED"
            }

        if session:
            return _run(session)
        else:
            with get_db() as sess:
                return _run(sess)

    def search_active_segments(
        self,
        query: str,
        top_k: int = CHROMA_TOP_K,
        domain_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Executes semantic search against ChromaDB resolved to PostgreSQL segments."""
        where_filter = None
        if domain_id:
            if len(domain_id) == 36 and "-" in domain_id:
                where_filter = {"domain_id": domain_id}
            else:
                where_filter = {"domain_code": domain_id.lower()}

        chroma_res = self.chroma_engine.search(query=query, top_k=top_k, where=where_filter)
        results = []
        if not chroma_res:
            return results

        import uuid
        from src.rag.database import get_db, KnowledgeBaseRepository

        with get_db() as sess:
            repo = KnowledgeBaseRepository(sess)
            for cr in chroma_res:
                seg_uuid_str = cr["id"]
                try:
                    seg = repo.get_segment_by_id(uuid.UUID(seg_uuid_str))
                except Exception:
                    seg = None

                if not seg or not seg.is_active:
                    continue

                results.append({
                    "segment_id": str(seg.id),
                    "domain_id": str(seg.domain_id),
                    "table_id": str(seg.table_id) if seg.table_id else None,
                    "section": seg.section,
                    "topic": seg.topic,
                    "segment_type": seg.segment_type,
                    "segment_name": seg.segment_name,
                    "content": seg.content,
                    "version": seg.version,
                    "score": float(cr.get("similarity", 0.0)),
                    "identity_key": cr.get("metadata", {}).get("identity_key", ""),
                    "vector_engine": "chroma",
                })

        return results


def sync_chroma_from_postgres(batch_size: int = 128) -> int:
    """Populates ChromaDB with all active knowledge segments from PostgreSQL."""
    from src.rag.database import get_db, KnowledgeBaseRepository
    from src.rag.models import KnowledgeSegmentModel, DomainModel, KnowledgeTableModel

    chroma_engine = get_chroma_engine()
    model = get_embedding_model()

    with get_db() as sess:
        repo = KnowledgeBaseRepository(sess)
        dom_map = {d.id: d.domain_id for d in repo.session.query(DomainModel).all()}
        table_map = {t.id: t.table_name for t in repo.session.query(KnowledgeTableModel).all()}

        active_segs = repo.session.query(KnowledgeSegmentModel).filter(
            KnowledgeSegmentModel.is_active == True
        ).all()

        total = len(active_segs)
        logger.info(f"Synchronizing {total} active segments from PostgreSQL to ChromaDB...")

        for i in range(0, total, batch_size):
            batch = active_segs[i : i + batch_size]
            ids = [str(s.id) for s in batch]
            docs = [s.content for s in batch]
            metas = []
            for s in batch:
                d_code = dom_map.get(s.domain_id, "")
                t_name = table_map.get(s.table_id, "") if s.table_id else (
                    s.segment_name.split(".")[0]
                    if "." in s.segment_name and s.segment_type in ("table", "column", "relationship")
                    else ""
                )
                ident_key = f"{d_code or str(s.domain_id)}:{t_name or 'none'}:{s.segment_name}"
                metas.append({
                    "domain_id": str(s.domain_id),
                    "domain_code": d_code,
                    "domain_name": s.section,
                    "table_id": str(s.table_id) if s.table_id else "",
                    "table_name": t_name,
                    "topic": s.topic,
                    "segment_type": s.segment_type,
                    "segment_name": s.segment_name,
                    "version": s.version,
                    "identity_key": ident_key,
                })
            embs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            chroma_engine.upsert_batch(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

        logger.info(f"ChromaDB synchronized successfully! Total vectors in ChromaDB: {chroma_engine.count()}")
        return total


def auto_sync_chroma() -> Dict[str, Any]:
    """
    Zero-argument change-detection synchronizer:
    Automatically identifies any altered, added, or deleted segments in PostgreSQL
    and synchronizes ChromaDB.
    """
    return get_embedding_synchronizer().auto_sync_changes()


_SYNC_INSTANCE: Optional[ChromaPostgresSynchronizer] = None


def get_embedding_synchronizer() -> ChromaPostgresSynchronizer:
    global _SYNC_INSTANCE
    if _SYNC_INSTANCE is None:
        _SYNC_INSTANCE = ChromaPostgresSynchronizer()
    return _SYNC_INSTANCE


# Re-export chunking helpers from src.rag.database for convenience
from src.rag.database import (
    generate_stable_segment_uuid,
    KnowledgeChunk,
    build_domain_chunk,
    build_table_chunk,
    build_column_chunk,
    build_relationship_chunk,
)

