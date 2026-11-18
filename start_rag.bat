@echo off
cd /d "%~dp0"
echo ========================================================
echo [1/3] Activating Virtual Environment...
echo ========================================================
call .venv\Scripts\activate.bat 2>nul

echo.
echo ========================================================
echo [2/3] Ingesting ERP Schemas into PostgreSQL (Primary)...
echo ========================================================
python -c "from src.rag.db.migration_service import run_full_migration; stats = run_full_migration(); print('Migration stats:', stats)"
if errorlevel 1 (
    echo.
    echo [ERROR] PostgreSQL Ingestion failed! Please verify PostgreSQL service and .env settings.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo [3/3] Synchronizing Vector Stores (ChromaDB + FAISS)...
echo ========================================================
python -c "from src.rag.db.embedding_sync import get_faiss_synchronizer, sync_chroma_from_postgres; syncer = get_faiss_synchronizer(); stats = syncer.sync_all(); n = sync_chroma_from_postgres(); print('FAISS Sync stats:', stats); print('ChromaDB Vectors:', n)"

echo.
echo ========================================================
echo [SUCCESS] PostgreSQL Knowledge Base, ChromaDB, and FAISS Ready!
echo ========================================================
pause
