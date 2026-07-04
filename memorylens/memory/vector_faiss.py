"""
FAISSMemory — production-grade vector retrieval backed by a FAISS index.

Same retrieval semantics as RAGMemory (top-K cosine similarity + recency
window) but the search runs inside faiss.IndexFlatIP instead of a NumPy
matmul, which is what a production deployment would use at scale.

Requires the optional dependency:  pip install "memorylens-bench[faiss]"
"""

from typing import Dict, List

import numpy as np

from memorylens.memory.base import BaseMemory
from memorylens.utils.embeddings import embed


def _require_faiss():
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "FAISSMemory requires faiss-cpu. Install it with: "
            'pip install "memorylens-bench[faiss]"'
        ) from e
    return faiss


class FAISSMemory(BaseMemory):
    """Top-K inner-product search over normalised embeddings in a FAISS index."""

    name = "faiss"

    def __init__(self, top_k: int = 5, recency_window: int = 4):
        self._faiss = _require_faiss()
        self.top_k = top_k
        self.recency_window = recency_window
        self.messages: List[Dict] = []
        self.index = None  # created lazily once embedding dimension is known

    def add_message(self, role: str, content: str, turn: int) -> None:
        emb = embed([content]).astype(np.float32)
        if self.index is None:
            self.index = self._faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.messages.append({"role": role, "content": content, "turn": turn})

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        if not self.messages:
            return []

        q_emb = embed([query]).astype(np.float32)
        k = min(self.top_k, len(self.messages))
        _, indices = self.index.search(q_emb, k)

        semantic = {int(i) for i in indices[0] if i >= 0}
        recency = set(range(max(0, len(self.messages) - self.recency_window),
                            len(self.messages)))
        selected = sorted(semantic | recency)

        return [{"role": self.messages[i]["role"], "content": self.messages[i]["content"]}
                for i in selected]

    def reset(self) -> None:
        self.messages = []
        self.index = None
