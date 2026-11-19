"""
Executable Auto-Detection Synchronizer for ChromaDB:
Scans PostgreSQL knowledge base against ChromaDB vectors, automatically identifies
any modified, new, or deleted segments using the 2-tier key architecture,
and re-indexes only the changed segments.

Usage:
    poetry run python utils/sync_chroma.py
"""
import sys
from src.rag.sync import run_sync

if __name__ == "__main__":
    sys.exit(run_sync())
