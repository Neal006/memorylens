from typing import List, Dict, Optional
import numpy as np
from .base import BaseMemory
from utils.embeddings import embed, top_k_indices


def _extractive_summary(messages: List[Dict], max_chars: int = 400) -> str:
    """
    Lightweight extractive summary: keep sentences that contain key=value patterns.
    No LLM call needed — fast and cost-free.
    """
    lines = []
    for m in messages:
        content = m.get("content", "")
        # Keep lines that look like personal facts
        if any(kw in content.lower() for kw in ["my ", "is ", "are ", "changed to", "name", "city", "age"]):
            lines.append(f"{m['role']}: {content}")
    summary = " | ".join(lines)
    return summary[:max_chars] if summary else "No key facts."


class CascadingTemporalMemory(BaseMemory):
    """
    Three-tier cascading memory with temporal decay:

    Hot  — last `hot_size` messages, verbatim, full fidelity
    Warm — older messages, full text but semantically filtered on retrieval
    Cold — ancient context, compressed to extractive summaries
    """

    name = "cascading"

    def __init__(self, hot_size: int = 12, warm_size: int = 30, cold_max: int = 4):
        self.hot_size = hot_size
        self.warm_size = warm_size
        self.cold_max = cold_max

        self.hot: List[Dict] = []
        self.warm: List[Dict] = []
        self.warm_embs: List[np.ndarray] = []
        self.cold: List[str] = []

        self.turn_count = 0

    def add_message(self, role: str, content: str, turn: int) -> None:
        msg = {"role": role, "content": content, "turn": turn}
        self.hot.append(msg)
        self.turn_count += 1

        if len(self.hot) > self.hot_size:
            self._cascade_hot()

    def _cascade_hot(self) -> None:
        overflow = self.hot[: len(self.hot) - self.hot_size]
        self.hot = self.hot[-self.hot_size :]

        for msg in overflow:
            self.warm.append(msg)
            self.warm_embs.append(embed([msg["content"]])[0])

        if len(self.warm) > self.warm_size:
            self._cascade_warm()

    def _cascade_warm(self) -> None:
        overflow = self.warm[: len(self.warm) - self.warm_size]
        self.warm = self.warm[-self.warm_size :]
        self.warm_embs = self.warm_embs[-self.warm_size :]

        summary = _extractive_summary(overflow)
        self.cold.append(summary)

        if len(self.cold) > self.cold_max:
            # Merge two oldest summaries
            merged = self.cold[0] + " | " + self.cold[1]
            self.cold = [merged[:600]] + self.cold[2:]

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        context: List[Dict] = []

        # Cold tier: inject as a system-level summary
        if self.cold:
            combined = " | ".join(self.cold[-2:])
            context.append({"role": "system", "content": f"[Historical context] {combined}"})

        # Warm tier: semantic retrieval with age-based decay
        if self.warm:
            q_emb = embed([query])[0]
            corpus = np.stack(self.warm_embs)
            raw_sims = (corpus @ q_emb).tolist()

            scored = []
            for i, sim in enumerate(raw_sims):
                age = current_turn - self.warm[i].get("turn", 0)
                decay = max(0.2, 1.0 - age / max(1, current_turn) * 0.6)
                scored.append((i, sim * decay))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_warm = sorted(idx for idx, _ in scored[:3])
            for i in top_warm:
                context.append({"role": self.warm[i]["role"], "content": self.warm[i]["content"]})

        # Hot tier: always fully included
        for msg in self.hot:
            context.append({"role": msg["role"], "content": msg["content"]})

        return context

    def reset(self) -> None:
        self.hot = []
        self.warm = []
        self.warm_embs = []
        self.cold = []
        self.turn_count = 0
