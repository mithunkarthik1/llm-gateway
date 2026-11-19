"""
Utilities package for domain entity priorities and schema synonyms.
"""
from utils.domain_priority import (
    DOMAIN_PRIMARY_TABLES,
    TABLE_PRIMARY_COLUMNS_MAP,
    prioritize_table_columns,
)
from utils.schema_synonyms import (
    TABLE_SYNONYMS,
    NONEXISTENT_TABLE_PATTERNS,
    resolve_canonical_table_name,
)

__all__ = [
    "DOMAIN_PRIMARY_TABLES",
    "TABLE_PRIMARY_COLUMNS_MAP",
    "prioritize_table_columns",
    "TABLE_SYNONYMS",
    "NONEXISTENT_TABLE_PATTERNS",
    "resolve_canonical_table_name",
]
