---
name: New Memory Backend
about: Propose or claim a new LLM memory architecture implementation for the MemoryLens benchmark
title: "[BACKEND] "
labels: enhancement, new-backend
assignees: ''
---

## Backend name

<!-- e.g. EntityMemory, GraphMemory, QdrantMemory, RedisMemory, SlidingWindowMemory -->

## Memory strategy

<!-- Describe in 2-3 sentences how this backend stores and retrieves messages.
     Examples:
     - "Extract named entities into a key-value store; retrieve by entity name matching"
     - "Use Qdrant vector DB with HNSW index; evict chunks older than N days"
     - "Knowledge graph: entities as nodes, relationships as edges; multi-hop retrieval"
-->

## Research hypothesis

<!-- What does this backend test? What trade-off does it make?
     The MemoryLens benchmark exists to measure LLM memory decay — what specific decay 
     behavior do you expect this backend to show vs Naive / RAG / Cascading?
-->

## Expected Recall@T curve

<!-- e.g. "Should match Cascading at T=50 but outperform at T=100 due to structured storage" -->

## Implementation sketch

```python
class YourMemory(BaseMemory):
    name = "your_backend"   # used in --backends flag

    def add_message(self, role: str, content: str, turn: int) -> None: ...
    def get_context(self, query: str, current_turn: int) -> List[Dict]: ...
    def reset(self) -> None: ...
```

## Dependencies required

<!-- Any new pip packages? Keep them optional (add to pyproject.toml extras). -->

## Related work

<!-- Papers, existing implementations, or LLM memory systems this is based on:
     e.g. MemGPT (Packer 2023), A-MEM (Xu 2024), LangChain EntityMemory
-->

## Are you claiming this to implement?

- [ ] Yes — I'll open a PR within 2 weeks
- [ ] No — leaving it open for the community

<!-- See docs/adding-a-new-backend.md for the full implementation guide -->
