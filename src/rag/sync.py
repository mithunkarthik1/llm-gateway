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
from src.config import get_logger
from src.rag.service import auto_sync_chroma

logger = get_logger("src.rag.sync")


def run_sync() -> int:
    """Executes full automated 2-tier synchronization with comprehensive logging."""
    try:
        result = auto_sync_chroma()
        return 0
    except Exception as e:
        logger.error(f"[AUTO-SYNC FATAL ERROR] Failed to complete sync: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(run_sync())
