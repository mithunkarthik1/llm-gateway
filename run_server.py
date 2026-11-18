import os
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=True)

