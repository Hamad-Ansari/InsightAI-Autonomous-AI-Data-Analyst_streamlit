"""
InsightAI - Embedding provider.

Serves document & query embeddings. Prefers Sentence-Transformers (local, free)
and falls back to a TF-IDF + SVD vectoriser when that dependency is unavailable,
so RAG never blocks the application.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

_HAS_ST = False
_st_model = None
_tfidf_model = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def _get_st(model_name: str):
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(model_name)
    return _st_model


def _get_tfidf():
    global _tfidf_model
    if _tfidf_model is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        _tfidf_model = {"vec": TfidfVectorizer(stop_words="english", max_features=5000),
                        "svd": TruncatedSVD(n_components=200, random_state=42)}
    return _tfidf_model


def embed_documents(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a batch of documents into a dense matrix."""
    if not texts:
        return np.zeros((0, 1), dtype=float)
    if _HAS_ST:
        try:
            model = _get_st(model_name)
            return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
        except Exception:
            pass
    tfidf = _get_tfidf()
    vec = tfidf["vec"]
    X = vec.fit_transform(texts) if not hasattr(vec, "vocabulary_") else vec.transform(texts)
    svd = tfidf["svd"]
    X = svd.fit_transform(X)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def embed_query(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query string."""
    if _HAS_ST:
        try:
            model = _get_st(model_name)
            return np.asarray(model.encode([text], normalize_embeddings=True))[0]
        except Exception:
            pass
    tfidf = _get_tfidf()
    X = tfidf["vec"].transform([text])
    X = tfidf["svd"].transform(X)
    v = X[0]
    return v / (np.linalg.norm(v) + 1e-9)


def backend_name(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> str:
    if _HAS_ST:
        try:
            _get_st(model_name)
            return "sentence-transformers"
        except Exception:
            pass
    return "tfidf"
