from typing import List, Dict
import numpy as np
from .base import BaseMemory
from memorylens.utils.embeddings import embed, top_k_indices


class RAGMemory(BaseMemory):
    """Embed every message; retrieve top-K semantically relevant ones per query."""

    name = "rag"

    def __init__(self, top_k: int = 5, recency_window: int = 4):
        self.messages: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.top_k = top_k
        self.recency_window = recency_window

    def add_message(self, role: str, content: str, turn: int) -> None:
        self.messages.append({"role": role, "content": content, "turn": turn})
        self.embeddings.append(embed([content])[0])

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        if not self.messages:
            return []

        q_emb = embed([query])[0]
        corpus = np.stack(self.embeddings)

        semantic_indices = set(top_k_indices(q_emb, corpus, self.top_k))
        recency_indices = set(range(max(0, len(self.messages) - self.recency_window), len(self.messages)))
        selected = sorted(semantic_indices | recency_indices)

        return [{"role": self.messages[i]["role"], "content": self.messages[i]["content"]}
                for i in selected]

    def reset(self) -> None:
        self.messages = []
        self.embeddings = []
