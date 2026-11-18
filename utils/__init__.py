"""
Utilities package for domain entity priorities and schema synonyms.
"""
from utils.domain_priority import (
    DOMAIN_PRIMARY_TABLES,
    TABLE_PRIMARY_COLUMNS_MAP,
)
from utils.schema_synonyms import (
    TABLE_SYNONYMS,
    NONEXISTENT_TABLE_PATTERNS,
)

__all__ = [
    "DOMAIN_PRIMARY_TABLES",
    "TABLE_PRIMARY_COLUMNS_MAP",
    "TABLE_SYNONYMS",
    "NONEXISTENT_TABLE_PATTERNS",
]
