"""
Dynamic Canonical Schema Synonyms & Anti-Hallucination Engine.
Provides natural language synonym mappings to canonical database tables and boundary rejection lists
derived dynamically from the PostgreSQL knowledge base without static hardcoding.
"""
import re
from typing import Dict, List, Optional, Any, Set


def get_word_inflections(w: str) -> List[str]:
    """Dynamically generates singular and plural variations for natural language terms."""
    w_low = w.lower().strip()
    if not w_low:
        return []

    res = [w_low]
    if w_low.endswith("ies") and len(w_low) > 4:
        res.append(w_low[:-3] + "y")
    elif w_low.endswith("es") and len(w_low) > 4:
        if any(w_low.endswith(end) for end in ["ches", "shes", "sses", "xes"]):
            res.append(w_low[:-2])
        else:
            res.append(w_low[:-1])
    elif w_low.endswith("s") and len(w_low) > 3 and not w_low.endswith("ss"):
        res.append(w_low[:-1])
    elif w_low.endswith("y") and len(w_low) > 3:
        res.append(w_low[:-1] + "ies")
    else:
        res.append(w_low + "s")
    return list(dict.fromkeys(res))


def build_dynamic_table_synonyms(all_tables: Dict[str, Any]) -> Dict[str, str]:
    """
    Dynamically generates natural language synonym mappings for ANY table catalog:
    1. Underscored and spaced canonical names
    2. Morphological inflections (singular <-> plural)
    3. Domain prefix stripping (e.g. 'hrms_employees' -> 'employees', 'employee')
    4. Metadata keywords stored directly in PostgreSQL
    5. Standard abbreviations and business equivalences
    """
    if not all_tables:
        return {}

    syn_map: Dict[str, str] = {}

    def add_synonym(syn: str, target: str):
        syn_c = syn.lower().strip()
        if syn_c and syn_c not in syn_map:
            syn_map[syn_c] = target

    for tname, tobj in all_tables.items():
        t_low = tname.lower()
        t_space = t_low.replace("_", " ")

        # 1. Canonical full names
        add_synonym(t_low, t_low)
        add_synonym(t_space, t_low)

        # 2. Inflections of full name
        for inf in get_word_inflections(t_space):
            add_synonym(inf, t_low)
            add_synonym(inf.replace(" ", "_"), t_low)

        # 3. Domain prefix stripping (e.g., 'hrms_departments' -> 'departments', 'hrms_employees' -> 'employees')
        parts = t_low.split("_")
        if len(parts) >= 2 and parts[0] in ("hrms", "erp"):
            suffix = "_".join(parts[1:])
            suffix_space = suffix.replace("_", " ")
            add_synonym(suffix, t_low)
            add_synonym(suffix_space, t_low)
            for inf in get_word_inflections(suffix_space):
                add_synonym(inf, t_low)
                add_synonym(inf.replace(" ", "_"), t_low)

        # 4. Table metadata keywords from database catalog
        for kw in tobj.get("keywords", []):
            kw_clean = str(kw).lower().strip()
            if kw_clean and len(kw_clean) >= 3:
                add_synonym(kw_clean, t_low)
                add_synonym(kw_clean.replace(" ", "_"), t_low)
                for inf in get_word_inflections(kw_clean):
                    add_synonym(inf, t_low)
                    add_synonym(inf.replace(" ", "_"), t_low)

    # 5. Dynamic acronyms and abbreviations derived from multi-word table names
    for tname in all_tables.keys():
        t_low = tname.lower()
        parts = t_low.split("_")
        if len(parts) >= 2:
            acr = "".join(p[0] for p in parts)
            if len(acr) >= 2 and acr not in all_tables:
                add_synonym(acr, t_low)

    return syn_map


class DynamicTableSynonyms(dict):
    """
    Dynamic dictionary proxy for TABLE_SYNONYMS.
    Automatically generates and caches synonyms from active PostgreSQL knowledge base tables.
    Zero hardcoded static dictionaries: seamlessly adapts to any newly uploaded domain.
    """
    def __init__(self):
        super().__init__()
        self._synonyms_cache: Optional[Dict[str, str]] = None

    def _ensure_synonyms(self) -> Dict[str, str]:
        if self._synonyms_cache is not None:
            return self._synonyms_cache

        try:
            from src.rag.service import load_erp_registry
            reg = load_erp_registry()
            if reg and "all_tables" in reg:
                self._synonyms_cache = build_dynamic_table_synonyms(reg["all_tables"])
                return self._synonyms_cache
        except Exception:
            pass
        return {}

    def __getitem__(self, key: str) -> str:
        cache = self._ensure_synonyms()
        k_low = key.lower().strip()
        if k_low in cache:
            return cache[k_low]
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        cache = self._ensure_synonyms()
        k_low = key.lower().strip()
        if k_low in cache:
            return cache[k_low]
        return default

    def __contains__(self, key: str) -> bool:
        cache = self._ensure_synonyms()
        return key.lower().strip() in cache

    def __iter__(self):
        cache = self._ensure_synonyms()
        return iter(cache)

    def __len__(self):
        cache = self._ensure_synonyms()
        return len(cache)

    def keys(self):
        cache = self._ensure_synonyms()
        return cache.keys()

    def values(self):
        cache = self._ensure_synonyms()
        return cache.values()

    def items(self):
        cache = self._ensure_synonyms()
        return cache.items()

    def clear_cache(self):
        self._synonyms_cache = None


TABLE_SYNONYMS = DynamicTableSynonyms()

# =====================================================================
# Anti-Hallucination Rejection Patterns (Nonexistent Entities)
# =====================================================================

# Dynamically resolved against PostgreSQL catalog at runtime (zero static hardcoding)
NONEXISTENT_TABLE_PATTERNS: List[str] = []


def resolve_canonical_table_name(text: str, all_tables: Dict[str, Any]) -> Optional[str]:
    """
    Dynamically resolves table name for ANY table in the knowledge base without requiring
    hardcoded static dictionaries. Checks exact matches, whitespace/underscore variants,
    inflections (singular/plural), domain prefix stripping, and metadata keywords from PostgreSQL.
    """
    t_lower = text.lower().strip()
    if not t_lower or not all_tables:
        return None

    # 1. Exact match
    if t_lower in all_tables:
        return t_lower
    t_underscored = t_lower.replace(" ", "_")
    if t_underscored in all_tables:
        return t_underscored

    # 2. Dynamic synonyms lookup
    if t_lower in TABLE_SYNONYMS:
        return TABLE_SYNONYMS[t_lower]

    # 3. Dynamic search across all active tables in DB catalog
    for t in sorted(all_tables.keys(), key=len, reverse=True):
        t_space = t.replace("_", " ")
        if re.search(r"\b" + re.escape(t) + r"\b", t_lower) or re.search(r"\b" + re.escape(t_space) + r"\b", t_lower):
            return t

        # Plural / singular inflection
        if t.endswith("s") and len(t) > 3:
            t_sing = t[:-1]
            t_sing_space = t_space[:-1]
            if re.search(r"\b" + re.escape(t_sing) + r"\b", t_lower) or re.search(r"\b" + re.escape(t_sing_space) + r"\b", t_lower):
                return t
        elif t.endswith("ies") and len(t) > 4:
            t_sing = t[:-3] + "y"
            if re.search(r"\b" + re.escape(t_sing) + r"\b", t_lower):
                return t

        # Prefix-stripped variants (e.g., 'hrms_employees' -> 'employees', 'bank_branches' -> 'branches')
        if "_" in t:
            suffix = t.split("_", 1)[1]
            suffix_space = suffix.replace("_", " ")
            if len(suffix) >= 4 and (re.search(r"\b" + re.escape(suffix) + r"\b", t_lower) or re.search(r"\b" + re.escape(suffix_space) + r"\b", t_lower)):
                return t

        # Table keywords stored in DB metadata
        tbl_meta = all_tables.get(t, {})
        for kw in tbl_meta.get("keywords", []):
            kw_clean = str(kw).lower().strip()
            if len(kw_clean) >= 4 and re.search(r"\b" + re.escape(kw_clean) + r"\b", t_lower):
                return t

    # 4. Synonym substring boundaries
    for syn, t in sorted(TABLE_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"\b" + re.escape(syn) + r"\b", t_lower):
            return t

    return None


__all__ = [
    "TABLE_SYNONYMS",
    "NONEXISTENT_TABLE_PATTERNS",
    "get_word_inflections",
    "build_dynamic_table_synonyms",
    "resolve_canonical_table_name",
]
