from abc import ABC, abstractmethod
from typing import List, Dict


class BaseMemory(ABC):
    name: str = "base"

    @abstractmethod
    def add_message(self, role: str, content: str, turn: int) -> None:
        pass

    @abstractmethod
    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        """Return list of {role, content} dicts relevant to query."""
        pass

    def token_count(self, query: str, current_turn: int) -> int:
        ctx = self.get_context(query, current_turn)
        return sum(len(m.get("content", "")) for m in ctx) // 4 + len(query) // 4

    @abstractmethod
    def reset(self) -> None:
        pass
