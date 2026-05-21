from typing import List, Optional
import numpy as np

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: List[str]) -> np.ndarray:
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def top_k_indices(query_emb: np.ndarray, corpus_embs: np.ndarray, k: int) -> List[int]:
    if len(corpus_embs) == 0:
        return []
    sims = corpus_embs @ query_emb
    k = min(k, len(sims))
    return list(np.argsort(sims)[::-1][:k])
