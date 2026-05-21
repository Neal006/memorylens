from typing import List, Dict
from .base import BaseMemory


class NaiveMemory(BaseMemory):
    """Keep the entire conversation history; truncate oldest when over token budget."""

    name = "naive"

    def __init__(self, max_context_tokens: int = 6000):
        self.messages: List[Dict] = []
        self.max_context_tokens = max_context_tokens

    def add_message(self, role: str, content: str, turn: int) -> None:
        self.messages.append({"role": role, "content": content, "turn": turn})

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        ctx = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        # Trim oldest pairs when over budget
        while self._tokens(ctx) > self.max_context_tokens and len(ctx) >= 2:
            ctx = ctx[2:]
        return ctx

    def _tokens(self, messages: List[Dict]) -> int:
        return sum(len(m.get("content", "")) for m in messages) // 4

    def reset(self) -> None:
        self.messages = []
