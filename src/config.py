import os
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Automatically load .env if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODELS_URL = os.getenv("GROQ_MODELS_URL", "https://api.groq.com/openai/v1/models")

def normalize_groq_model(model_name: str) -> str:
    m = (model_name or "").strip()
    if not m:
        return "openai/gpt-oss-120b"
    if m.startswith("gpt-oss-"):
        return f"openai/{m}"
    return m

DEFAULT_GROQ_MODEL = normalize_groq_model(os.getenv("DEFAULT_GROQ_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")))

SRC_DIR = Path(__file__).resolve().parent
MODEL_DIR = SRC_DIR / "model"
if not MODEL_DIR.exists():
    MODEL_DIR = BASE_DIR / "model"
MODEL_SUBFOLDER = "onnx"
MODEL_FILE = "model_q4.onnx"

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

