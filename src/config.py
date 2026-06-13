import os
from pathlib import Path

# Fix for StanfordOpenIE KeyError: 'CORENLP_HOME' on script exit
if 'CORENLP_HOME' not in os.environ:
    os.environ['CORENLP_HOME'] = ""

# Monkey-patch StanfordOpenIE to ignore KeyError on exit
try:
    from openie import StanfordOpenIE
    _original_del = getattr(StanfordOpenIE, '__del__', None)
    if _original_del:
        def safe_del(self):
            try:
                _original_del(self)
            except Exception:
                pass
        StanfordOpenIE.__del__ = safe_del
except Exception:
    pass

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys (defaults to empty list if not found)
raw_keys = os.environ.get("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

# If no env vars are set, fallback to your hardcoded list (for transition)
if not GROQ_API_KEYS:
    GROQ_API_KEYS = [
        "put your api keys here"
    ]

# Models
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama-3.3-70b-versatile")
GPT_MODEL = os.environ.get("GPT_MODEL", "openai/gpt-oss-120b")

# Paths
CACHE_DIR = BASE_DIR / "Cache"
LOG_DIR = BASE_DIR / "Logging"