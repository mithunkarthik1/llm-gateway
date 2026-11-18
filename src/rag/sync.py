"""
Executable Auto-Detection Synchronizer for ChromaDB:
Scans PostgreSQL knowledge base against ChromaDB vectors, automatically identifies
any modified, new, or deleted segments using the 2-tier key architecture,
and re-indexes only the changed segments.

Usage:
    poetry run python -m src.rag.sync
"""
import sys
import json
from src.rag.service import auto_sync_chroma

if __name__ == "__main__":
    result = auto_sync_chroma()
    sys.exit(0)
