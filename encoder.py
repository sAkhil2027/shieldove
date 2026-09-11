import os
from threading import Lock
from sentence_transformers import SentenceTransformer

_lock = Lock()
_encoder = None

def get_encoder():
    """Return a singleton SentenceTransformer instance.
    The model name can be overridden via the ``EMBEDDING_MODEL_NAME`` environment variable.
    """
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
                _encoder = SentenceTransformer(model_name)
    return _encoder
