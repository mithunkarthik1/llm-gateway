import time
from src.rag.service import get_chroma_engine

engine = get_chroma_engine()
t0 = time.time()
data = engine.collection.get(include=["documents", "metadatas"])
t1 = time.time()
print(f"Fetched {len(data['ids'])} vectors in {round((t1 - t0) * 1000, 2)} ms")
