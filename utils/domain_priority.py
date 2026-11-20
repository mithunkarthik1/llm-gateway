"""
Dynamic Domain and Table Entity Priority Engine.
Derives core master entities, topological dependency closures, and primary business column rankings
dynamically from the PostgreSQL knowledge base, domain metadata, and relational dependency graphs.
Zero hardcoded domain tables or column lists.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

DOMAINS_DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "rag" / "data" / "domains"


def compute_dynamic_topological_priority(tables: List[Dict[str, Any]]) -> List[str]:
    """
    Computes a strictly closed topological priority sequence for any ERP domain:
    1. Analyzes foreign-key graph dependencies and transitive downstream impact.
    2. Identifies foundational master/hub entities (categories, customers, departments, etc.).
    3. Guarantees that for ANY prefix of K tables, all parent dependencies are satisfied
       (zero dangling foreign keys in generated SQL DDL).
    4. Automatically adapts to any newly added ERP domain with zero hardcoding.
    """
    if not tables:
        return []

    t_map = {t["table_name"].lower(): t for t in tables if "table_name" in t}
    all_tnames = list(t_map.keys())

    # Build direct children and dependencies within this domain
    children: Dict[str, Set[str]] = {t: set() for t in all_tnames}
    deps: Dict[str, Set[str]] = {}
    for tname, tobj in t_map.items():
        raw_deps = tobj.get("dependencies") or tobj.get("actual_dependencies") or []
        tdeps = set(
            d.lower() for d in raw_deps
            if d.lower() in t_map and d.lower() != tname
        )
        deps[tname] = tdeps
        for d in tdeps:
            children[d].add(tname)

    # Compute transitive downstream reach (how many tables directly or indirectly rely on this table)
    def get_downstream(t: str) -> Set[str]:
        visited: Set[str] = set()
        stack = list(children[t])
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                stack.extend(children[curr])
        return visited

    downstream = {t: len(get_downstream(t)) for t in all_tnames}

    base_scores: Dict[str, float] = {}
    for t in all_tnames:
        base_scores[t] = 100 * downstream[t] + 15 * len(children[t]) - 5 * len(deps[t])

    ordered: List[str] = []
    remaining: Set[str] = set(all_tnames)

    while remaining:
        # Candidates whose dependencies are already satisfied in ordered
        ready = [t for t in remaining if deps[t].issubset(set(ordered))]
        if not ready:
            # Cycle breaker: pick candidate with smallest unsatisfied dependency count
            ready = sorted(remaining, key=lambda t: len(deps[t] - set(ordered)))
            best = ready[0]
        else:
            def dyn_score(t: str) -> tuple:
                parents_in = len(deps[t].intersection(set(ordered)))
                is_line_item = 1 if any(t.endswith(sfx) for sfx in ["_items", "_lines", "_repayments", "_records"]) else 0
                is_isolated_leaf = 1 if (downstream[t] == 0 and parents_in == 0 and len(children[t]) <= 1) else 0
                score = base_scores[t] + (150 * parents_in) + (200 * is_line_item if parents_in > 0 else 0) - (300 * is_isolated_leaf)
                return (
                    score,
                    len(t_map[t].get("columns", [])),
                    -len(t),
                    t
                )
            best = max(ready, key=dyn_score)

        ordered.append(best)
        remaining.remove(best)

    return ordered


def ensure_closed_dependency_ordering(seed_order: List[str], domain_tables: List[Dict[str, Any]]) -> List[str]:
    """
    Given a domain table ordering, ensures that every parent table precedes its dependent child
    tables so that ANY prefix of K tables forms a closed relational subgraph with 0 dangling FKs.
    """
    if not seed_order or not domain_tables:
        return list(seed_order)

    t_map = {t["table_name"].lower(): t for t in domain_tables if "table_name" in t}
    all_valid = [t for t in seed_order if t.lower() in t_map]

    deps: Dict[str, List[str]] = {}
    for tname in all_valid:
        tobj = t_map[tname.lower()]
        raw_deps = tobj.get("dependencies") or tobj.get("actual_dependencies") or []
        deps[tname.lower()] = [d.lower() for d in raw_deps if d.lower() in t_map and d.lower() != tname.lower()]

    result: List[str] = []
    added: Set[str] = set()

    def add_table(t: str):
        if t in added:
            return
        # Add all dependencies first to guarantee closed prefix
        for dep in deps.get(t, []):
            if dep not in added and dep in t_map:
                add_table(dep)
        added.add(t)
        result.append(t)

    for t in all_valid:
        add_table(t.lower())

    # Add any remaining tables not in seed_order
    for t in t_map:
        if t not in added:
            add_table(t)

    return result


class DynamicDomainPrimaryTables(dict):
    """
    Dynamic dictionary proxy for DOMAIN_PRIMARY_TABLES.
    Derives table priorities dynamically from domain metadata / PostgreSQL knowledge base
    and ensures closed topological dependencies.
    Zero hardcoded domain tables.
    """
    def __init__(self):
        super().__init__()
        self._cache: Dict[str, List[str]] = {}

    def _load_domain_priority(self, domain_id: str) -> List[str]:
        dom_id = domain_id.lower().strip()
        if dom_id in self._cache:
            return self._cache[dom_id]

        # 1. Try reading author seed order from domain.json
        seed_order: List[str] = []
        domain_json_path = DOMAINS_DATA_DIR / dom_id / "domain.json"
        if domain_json_path.exists():
            try:
                with open(domain_json_path, "r", encoding="utf-8") as f:
                    d_meta = json.load(f)
                    seed_order = d_meta.get("table_names", [])
            except Exception:
                pass

        # 2. Load tables from active registry
        tables: List[Dict[str, Any]] = []
        try:
            from src.rag.service import load_erp_registry
            reg = load_erp_registry()
            if reg and "domains" in reg and dom_id in reg["domains"]:
                tables = reg["domains"][dom_id].get("tables", [])
        except Exception:
            pass

        if seed_order and tables:
            priority = ensure_closed_dependency_ordering(seed_order, tables)
            self._cache[dom_id] = priority
            return priority

        if tables:
            priority = compute_dynamic_topological_priority(tables)
            self._cache[dom_id] = priority
            return priority

        if seed_order:
            self._cache[dom_id] = seed_order
            return seed_order

        return []

    def __getitem__(self, domain_id: str) -> List[str]:
        prio = self._load_domain_priority(domain_id)
        if prio:
            return prio
        return super().get(domain_id, [])

    def get(self, domain_id: str, default=None):
        prio = self._load_domain_priority(domain_id)
        if prio:
            return prio
        return default if default is not None else []

    def __contains__(self, domain_id: str) -> bool:
        prio = self._load_domain_priority(domain_id)
        return bool(prio)

    def keys(self):
        try:
            from src.rag.service import load_erp_registry
            reg = load_erp_registry()
            if reg and "domains" in reg:
                return list(reg["domains"].keys())
        except Exception:
            pass
        return list(super().keys())

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def clear_cache(self):
        self._cache.clear()


DOMAIN_PRIMARY_TABLES = DynamicDomainPrimaryTables()


def get_domain_primary_tables(domain_id: str, domain_tables: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    Dynamically derives primary/core master entities for any domain.
    If domain_tables are provided, computes closed topological dependency sequence.
    Otherwise, loads tables dynamically from the PostgreSQL knowledge base.
    Guarantees a complete, full-domain ranking for any table count K without hardcoding.
    """
    dom_low = domain_id.lower().strip()
    prio = DOMAIN_PRIMARY_TABLES.get(dom_low, [])
    if prio:
        return prio

    if domain_tables:
        return compute_dynamic_topological_priority(domain_tables)

    return []


def prioritize_table_columns(
    table_name: str,
    all_cols: List[str],
    include_pk: bool = True,
    column_details: Optional[List[Dict[str, Any]]] = None
) -> List[str]:
    """
    Ranks columns dynamically by business importance for ANY table without static configurations:
    1. Primary Key ('id' / is_primary_key) -> rank 0
    2. Natural business keys & identifiers (code, number, sku, name, title) -> rank 1
    3. Monetary amount / financial balance (total_amount, amount, total, price, salary, balance) -> rank 2
    4. Categorical / temporal attributes (date, status, type, rate, email, phone) -> rank 3
    5. Foreign keys (endswith('_id') / is_foreign_key) -> rank 4
    6. Audit timestamps & security hashes (created_at, updated_at, deleted_at, password_hash) -> rank 6
    7. General attributes -> rank 5
    """
    if not all_cols:
        return []

    detail_map = {}
    if column_details:
        for cd in column_details:
            cname = cd.get("column_name", "").lower()
            if cname:
                detail_map[cname] = cd
    else:
        try:
            from src.rag.service import load_erp_registry
            reg = load_erp_registry()
            if reg and "all_tables" in reg and table_name.lower() in reg["all_tables"]:
                for cd in reg["all_tables"][table_name.lower()].get("column_details", []):
                    cname = cd.get("column_name", "").lower()
                    if cname:
                        detail_map[cname] = cd
        except Exception:
            pass

    col_idx = {c.lower(): i for i, c in enumerate(all_cols)}

    def col_rank(c: str) -> tuple:
        cl = c.lower()
        cd = detail_map.get(cl, {})
        # 1. Primary keys
        if cl in ("id", "primary_key") or cd.get("is_primary_key"):
            return (0, col_idx.get(cl, 99))
        # 2. Natural business keys & identifiers
        if any(k in cl for k in ["code", "number", "sku", "name", "title", "identifier"]):
            return (1, col_idx.get(cl, 99))
        # 3. Monetary amounts and financial values (primary aggregate first)
        if cl in ("total_amount", "amount", "total", "grand_total", "net_amount", "salary", "balance"):
            return (2, 0, col_idx.get(cl, 99))
        if any(k in cl for k in ["amount", "total", "price", "salary", "balance"]):
            return (2, 1, col_idx.get(cl, 99))
        # 4. Dates and statuses
        if any(k in cl for k in ["rate", "date", "status", "type", "email", "phone"]):
            return (3, col_idx.get(cl, 99))
        # 5. Foreign keys
        if cl.endswith("_id") or cd.get("is_foreign_key"):
            return (4, col_idx.get(cl, 99))
        # 6. Audit timestamps & security hashes
        if any(k in cl for k in ["created_at", "updated_at", "deleted_at", "password_hash", "token"]):
            return (6, col_idx.get(cl, 99))
        # 7. General attributes
        return (5, col_idx.get(cl, 99))

    ranked = sorted(all_cols, key=col_rank)

    if not include_pk:
        ranked = [c for c in ranked if c.lower() not in ("id", "primary_key")]
    elif "id" in all_cols and (not ranked or ranked[0].lower() != "id"):
        ranked = ["id"] + [c for c in ranked if c.lower() != "id"]

    return ranked


class DynamicTablePrimaryColumnsMap(dict):
    """
    Dynamic dictionary proxy for TABLE_PRIMARY_COLUMNS_MAP.
    Derives primary columns dynamically for any table from DB metadata.
    Zero hardcoded column lists.
    """
    def __getitem__(self, table_name: str) -> List[str]:
        return self.get(table_name, [])

    def get(self, table_name: str, default=None):
        try:
            from src.rag.service import load_erp_registry
            reg = load_erp_registry()
            t_low = table_name.lower().strip()
            if reg and "all_tables" in reg and t_low in reg["all_tables"]:
                tobj = reg["all_tables"][t_low]
                cols = tobj.get("columns", [])
                cdetails = tobj.get("column_details", [])
                return prioritize_table_columns(t_low, cols, include_pk=True, column_details=cdetails)[:5]
        except Exception:
            pass
        return default if default is not None else []

    def __contains__(self, table_name: str) -> bool:
        try:
            from src.rag.service import load_erp_registry
            reg = load_erp_registry()
            return bool(reg and "all_tables" in reg and table_name.lower().strip() in reg["all_tables"])
        except Exception:
            return False


TABLE_PRIMARY_COLUMNS_MAP = DynamicTablePrimaryColumnsMap()

__all__ = [
    "DOMAIN_PRIMARY_TABLES",
    "TABLE_PRIMARY_COLUMNS_MAP",
    "compute_dynamic_topological_priority",
    "ensure_closed_dependency_ordering",
    "get_domain_primary_tables",
    "prioritize_table_columns",
]
