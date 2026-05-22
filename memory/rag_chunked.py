"""
memory/rag_chunked.py — Bounded Chunked RAG Memory.

Models a production RAG system with two critical realism constraints:

1. **Chunking**: messages are split into overlapping token-window chunks before
   indexing, just as documents are chunked in real RAG pipelines.  A long
   message that mentions two different facts may produce chunks that each
   only contain one — increasing retrieval difficulty.

2. **Bounded index**: the vector index has a hard capacity (`max_chunks`).
   When capacity is reached, the *oldest* chunks are evicted (FIFO),
   simulating a production system that cannot store unlimited embeddings.

These two constraints together produce realistic recall decay: early-injected
facts gradually fall out of the index or get buried by noise chunks, causing
recall to degrade in a way the ideal RAGMemory never shows.

Contrast with RAGMemory (memory/rag.py):
  RAGMemory  — whole messages, unbounded, perfect recall (upper bound)
  ChunkedRAGMemory — chunked + evicting, bounded index (realistic lower bound)
"""

from typing import List, Dict, Tuple
import numpy as np
from .base import BaseMemory
from utils.embeddings import embed, top_k_indices


def _chunk_text(text: str, chunk_chars: int = 120, overlap_chars: int = 30) -> List[str]:
    """
    Split `text` into overlapping character-window chunks.

    chunk_chars  ~= 30 tokens  (GPT/Claude tokenise at ~4 chars/token)
    overlap_chars ~= 7 tokens  (~25% overlap, standard in production RAG)

    Short texts that fit in one chunk are returned as-is.
    """
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunks.append(text[start:end].strip())
        start += chunk_chars - overlap_chars
    return [c for c in chunks if c]


class ChunkedRAGMemory(BaseMemory):
    """
    Production-realistic RAG with chunking and a bounded FIFO index.

    Parameters
    ----------
    top_k        : chunks to retrieve per query (semantic)
    recency_k    : most-recent chunks always included (recency bias)
    chunk_chars  : characters per chunk (~30 tokens)
    overlap_chars: overlap between consecutive chunks
    max_chunks   : hard index capacity — oldest evicted when exceeded
    """

    name = "rag_chunked"

    def __init__(
        self,
        top_k:         int = 5,
        recency_k:     int = 4,
        chunk_chars:   int = 120,
        overlap_chars: int = 30,
        max_chunks:    int = 200,
    ):
        self.top_k         = top_k
        self.recency_k     = recency_k
        self.chunk_chars   = chunk_chars
        self.overlap_chars = overlap_chars
        self.max_chunks    = max_chunks

        # Each entry: {"role", "content"(chunk text), "source_turn", "chunk_idx"}
        self.chunks: List[Dict]        = []
        self.embeddings: List[np.ndarray] = []

    def add_message(self, role: str, content: str, turn: int) -> None:
        pieces = _chunk_text(content, self.chunk_chars, self.overlap_chars)
        new_chunks = [
            {"role": role, "content": piece, "source_turn": turn, "chunk_idx": i}
            for i, piece in enumerate(pieces)
        ]
        new_embs = embed([c["content"] for c in new_chunks])

        for chunk, emb in zip(new_chunks, new_embs):
            self.chunks.append(chunk)
            self.embeddings.append(emb)

        # Evict oldest chunks when over capacity (FIFO)
        if len(self.chunks) > self.max_chunks:
            excess = len(self.chunks) - self.max_chunks
            self.chunks     = self.chunks[excess:]
            self.embeddings = self.embeddings[excess:]

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        if not self.chunks:
            return []

        q_emb  = embed([query])[0]
        corpus = np.stack(self.embeddings)

        semantic_indices = set(top_k_indices(q_emb, corpus, self.top_k))
        recency_indices  = set(range(max(0, len(self.chunks) - self.recency_k), len(self.chunks)))
        selected = sorted(semantic_indices | recency_indices)

        # Deduplicate by source_turn to avoid flooding context with chunk variants
        seen_turns = set()
        result: List[Dict] = []
        for i in selected:
            c = self.chunks[i]
            key = (c["source_turn"], c["chunk_idx"])
            if key not in seen_turns:
                seen_turns.add(key)
                result.append({"role": c["role"], "content": c["content"]})
        return result

    def reset(self) -> None:
        self.chunks     = []
        self.embeddings = []
