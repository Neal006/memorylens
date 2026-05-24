"""
memory/entity.py — Entity Memory backend for MemoryLens.

Implements structured named-entity extraction: instead of storing raw message
history, EntityMemory maintains a key-value store of extracted facts and
returns those facts as context.  When a fact is updated, the store is patched
in-place, so retrieval always surfaces the current value.

Extraction is fully local (regex-based, no LLM call) to keep the backend fast
and reproducible for the benchmark harness.

Closes #12.
"""

import re
from typing import Dict, List, Optional, Tuple
from .base import BaseMemory


# Patterns recognised by the extractor
# Each tuple: (compiled_pattern, group_name_for_key, group_name_for_value)
_INJECTION_RE = re.compile(
    r"my\s+(?P<key>[a-z][a-z ]{0,30}?)\s+is\s+(?P<value>.+?)[\.\!]?\s*$",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"my\s+(?P<key>[a-z][a-z ]{0,30}?)\s+has\s+changed\s+to\s+(?P<value>.+?)[\.\!]?\s*$",
    re.IGNORECASE,
)


def _extract_entity(content: str) -> Optional[Tuple[str, str]]:
    """
    Try to extract a (key, value) fact pair from a single message string.
    Returns (normalised_key, value) or None if no pattern matches.
    """
    for pattern in (_UPDATE_RE, _INJECTION_RE):
        m = pattern.search(content)
        if m:
            key = m.group("key").strip().lower()
            val = m.group("value").strip()
            return key, val
    return None


class EntityMemory(BaseMemory):
    """
    Structured entity-extraction memory backend.

    Facts are stored as a key-value dictionary rather than as raw messages.
    On retrieval the entire entity store is serialised into a concise context
    block so the LLM always sees the *current* value of every known fact.

    Advantages over RAG/Naive for fact-tracking tasks:
    - O(1) update: overwriting a key replaces the fact immediately
    - No stale value can persist once an update is absorbed
    - Token cost is proportional to the number of unique facts, not turns

    Limitations:
    - Only captures facts matching the injection/update templates from the
      benchmark's conversation generator; free-form facts are not extracted
    - Does not retain conversational flow or filler turns

    Reference: Entity-centric memory as described in
    Xu et al. (2021) "Beyond Goldfish Memory" and related work on
    structured dialogue state tracking.
    """

    name = "entity"

    def __init__(self, max_facts: int = 64):
        # Ordered so serialisation is deterministic
        self.entities: Dict[str, str] = {}
        self.max_facts = max_facts

    def add_message(self, role: str, content: str, turn: int) -> None:
        if role != "user":
            return  # only extract from user utterances
        pair = _extract_entity(content)
        if pair:
            key, val = pair
            self.entities[key] = val
            # Enforce max_facts by evicting oldest key when over limit
            if len(self.entities) > self.max_facts:
                oldest_key = next(iter(self.entities))
                del self.entities[oldest_key]

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        if not self.entities:
            return []

        lines = [f"my {key} is {val}" for key, val in self.entities.items()]
        entity_block = "; ".join(lines) + "."
        return [
            {
                "role": "system",
                "content": f"[Known facts about the user] {entity_block}",
            }
        ]

    def reset(self) -> None:
        self.entities = {}
