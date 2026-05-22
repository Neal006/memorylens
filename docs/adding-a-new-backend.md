# How to Add a New Memory Backend to MemoryLens

This guide walks through implementing a custom LLM memory backend and benchmarking it against the existing baselines. The full interface is **3 methods** — it takes under an hour to add a working backend.

---

## The BaseMemory Interface

Every memory backend in MemoryLens inherits from `BaseMemory`:

```python
# memory/base.py
class BaseMemory(ABC):
    name: str = "base"   # used in --backends flag and results tables

    @abstractmethod
    def add_message(self, role: str, content: str, turn: int) -> None:
        """Store one message. Called for every user and assistant turn."""
        pass

    @abstractmethod
    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        """Return a list of {"role": ..., "content": ...} dicts.
        These are what get measured by the evaluator.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored state. Called between benchmark runs."""
        pass

    def token_count(self, query: str, current_turn: int) -> int:
        """Default: count characters / 4. Override for a more accurate estimate."""
        ctx = self.get_context(query, current_turn)
        return sum(len(m.get("content", "")) for m in ctx) // 4 + len(query) // 4
```

---

## Step-by-Step: Implementing EntityMemory

As a concrete example, let's implement an entity-extraction memory backend that stores named entities separately from conversation flow.

### Step 1 — Create `memory/entity.py`

```python
from typing import List, Dict
from .base import BaseMemory
import re

class EntityMemory(BaseMemory):
    """
    Stores named entities extracted from conversation messages.
    Retrieves entity facts relevant to the query.
    
    This models systems like LangChain's EntityMemory that maintain
    a separate knowledge store for named entities rather than raw messages.
    """

    name = "entity"

    def __init__(self, max_entities: int = 50):
        self.entities: Dict[str, str] = {}    # key -> latest value
        self.max_entities = max_entities
        self.recent: List[Dict] = []          # last 4 messages verbatim

    _PATTERN = re.compile(
        r"my (\w[\w\s]+?) (?:is|has changed to) ([^.]+)\.",
        re.IGNORECASE,
    )

    def add_message(self, role: str, content: str, turn: int) -> None:
        # Extract entities
        for match in self._PATTERN.finditer(content):
            key = match.group(1).strip().lower()
            val = match.group(2).strip()
            self.entities[key] = val

        # Keep last 4 messages verbatim for recency
        self.recent.append({"role": role, "content": content})
        if len(self.recent) > 4:
            self.recent.pop(0)

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        context = []
        
        # Inject entity store as system context
        if self.entities:
            facts = " | ".join(f"{k}: {v}" for k, v in self.entities.items())
            context.append({
                "role": "system",
                "content": f"[Known facts about user] {facts}",
            })
        
        # Append recent messages
        context.extend({"role": m["role"], "content": m["content"]} for m in self.recent)
        return context

    def reset(self) -> None:
        self.entities = {}
        self.recent = []
```

### Step 2 — Register in `evaluation/benchmark.py`

```python
from memory.entity import EntityMemory

def _make_memory(name: str, decay: str = "ebbinghaus") -> BaseMemory:
    # ... existing cases ...
    if name == "entity":
        return EntityMemory()
    # ...
```

Also add `"entity"` to `VALID_BACKENDS`.

### Step 3 — Add tests

```python
# tests/test_pipeline.py
def test_entity_recall_early():
    from memory.entity import EntityMemory
    mem = EntityMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.75, f"Expected >=75% recall, got {rate:.0%}"
    print(f"PASS: EntityMemory recall early ({rate:.0%})")
```

### Step 4 — Run the benchmark

```bash
python main.py --backends naive rag entity cascading
python main.py --seeds 5 --backends naive rag entity cascading
```

### Step 5 — Open a PR

Include:
- `memory/entity.py` — the backend implementation
- Updated `evaluation/benchmark.py` — registration
- Test in `tests/test_pipeline.py`
- Entry in `CHANGELOG.md` under `[Unreleased]`

---

## Ideas for New Backends

| Backend | Strategy | Hypothesis to test |
|---------|----------|-------------------|
| `entity` | Named-entity extraction into a key-value store | Does structured storage beat unstructured retrieval? |
| `qdrant` | Production vector DB (Qdrant) | Does a real vector DB beat NumPy cosine at scale? |
| `redis` | Persistent Redis-backed storage | Does persistence across sessions affect recall? |
| `memgpt_style` | Virtual paging between in-context and external | Does OS-style memory management beat Cascading? |
| `graph` | Knowledge graph (entities + relationships) | Does structured relationships help with multi-hop facts? |
| `sliding_window` | Fixed K-message window | What's the optimal window size? |
| `importance_weighted` | Keep messages by semantic importance score | Does importance sampling beat recency? |

Each of these is a potential research contribution. Open an issue to claim one before starting.

---

*Questions? Open a [Discussion](https://github.com/Neal006/memorylens/discussions) or check [CONTRIBUTING.md](../CONTRIBUTING.md).*
