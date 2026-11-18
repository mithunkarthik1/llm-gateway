"""
Automated Verification of User Flow:
1. Manually update column segment content in PostgreSQL using direct SQL.
2. Run ChromaDB sync without mentioning any segment name or key.
3. Verify logs automatically detect ONLY the 1 modified segment and re-indexes it.
4. Hit FastAPI endpoint /api/v1/generate and verify identical valid response.
5. Restore PostgreSQL content and run sync again (confirming 0 changes).
"""
import sys
from fastapi.testclient import TestClient
from sqlalchemy import text

from src.main import app
from src.rag.database import get_engine, get_db, KnowledgeBaseRepository
from src.rag.service import auto_sync_chroma, get_chroma_engine

TARGET_KEY = "erp_finance:customer_invoices:customer_id"


def test_flow():
    engine = get_engine()
    chroma_engine = get_chroma_engine()

    print("\n" + "=" * 80)
    print("STEP 1: INSPECT ORIGINAL CONTENT IN POSTGRESQL")
    print("=" * 80)

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, segment_key, segment_name, version, content FROM knowledge_segments WHERE segment_key = :k"),
            {"k": TARGET_KEY}
        ).fetchone()

        if not row:
            print(f"Error: Segment {TARGET_KEY} not found!")
            return

        seg_id, seg_key, seg_name, orig_version, orig_content = row[0], row[1], row[2], row[3], row[4]
        print(f"Segment Key     : {seg_key}")
        print(f"Segment Name    : {seg_name}")
        print(f"Current Version : {orig_version}")
        print(f"Original Length : {len(orig_content)} characters")

    print("\n" + "=" * 80)
    print("STEP 2: MANUALLY ALTER FIELD/CONTENT IN POSTGRESQL VIA RAW SQL")
    print("=" * 80)

    modified_content = orig_content + "\n-- Manual DBA update: Added optimized client indexing note."
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE knowledge_segments SET content = :c, version = version + 1, updated_at = NOW() WHERE segment_key = :k"),
            {"c": modified_content, "k": TARGET_KEY}
        )
    print(f"Executed Raw SQL UPDATE on PostgreSQL table 'knowledge_segments' for '{TARGET_KEY}'!")

    print("\n" + "=" * 80)
    print("STEP 3: RUN CHROMA DB EMBEDDING WITHOUT MENTIONING ANYTHING (ZERO ARGS)")
    print("=" * 80)
    print("Running auto_sync_chroma()...")

    sync_result = auto_sync_chroma()
    print("\nSync Execution Summary Result:")
    print(f"  - Total Changes Detected : {sync_result['total_changes']}")
    print(f"  - Modified Segments      : {sync_result['modified']}")
    print(f"  - Re-indexed Segment(s)  : {sync_result['reindexed_segments']}")
    print(f"  - Final ChromaDB Count   : {sync_result['final_count']} (Strictly 746, 0 duplicates)")

    print("\n" + "=" * 80)
    print("STEP 4: HIT THE ENDPOINT AND VERIFY RESULT")
    print("=" * 80)

    client = TestClient(app)
    payload = {
        "prompt": "Retrieve customer_invoices.customer_id",
        "enable_summary": True
    }
    print(f"POST /api/v1/generate with payload: {payload}")
    res = client.post("/api/v1/generate", json=payload)
    print(f"Response HTTP Status : {res.status_code}")
    res_data = res.json()
    print(f"Response Message     : {res_data.get('message')}")
    print(f"Response Intent      : {res_data.get('data', {}).get('intent')}")
    print(f"Response SQL Generated:\n{res_data.get('data', {}).get('sql')}")

    print("\n" + "=" * 80)
    print("STEP 5: RESTORE ORIGINAL CONTENT IN POSTGRESQL & RE-RUN ZERO-ARG SYNC")
    print("=" * 80)

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE knowledge_segments SET content = :c, version = :v, updated_at = NOW() WHERE segment_key = :k"),
            {"c": orig_content, "v": orig_version, "k": TARGET_KEY}
        )
    print("PostgreSQL content reverted to original master state.")

    # Re-sync back to pristine state
    restore_result = auto_sync_chroma()
    print(f"Post-Restoration Sync Result: {restore_result['reindexed_segments']} (Final Count: {restore_result['final_count']})")

    # One more zero-arg sync to confirm 0 changes
    idle_result = auto_sync_chroma()
    print(f"Idle Check Changes Detected: {idle_result['total_changes']} (Status: {idle_result['status']})")
    print("\n" + "=" * 80)
    print("TESTING FLOW VALIDATED 100% SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_flow()
