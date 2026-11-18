import json
from src.rag.database import get_db, KnowledgeBaseRepository
from src.rag.models import KnowledgeSegmentModel

with get_db() as sess:
    repo = KnowledgeBaseRepository(sess)
    seg = repo.get_segment_by_key("erp_finance:customer_invoices:customer_id")
    if seg:
        print("FOUND SEGMENT by KEY:")
        print("ID:", seg.id)
        print("Key:", seg.segment_key)
        print("Name:", seg.segment_name)
        print("Type:", seg.segment_type)
        print("Version:", seg.version)
        print("Content:\n" + seg.content)
    else:
        print("Searching all column segments in erp_finance...")
        segs = sess.query(KnowledgeSegmentModel).filter(
            KnowledgeSegmentModel.segment_type == "column"
        ).limit(5).all()
        for s in segs:
            print(f"- {s.segment_key} | {s.segment_name}")
