---
name: New Memory Backend
about: Propose or claim a new memory backend implementation
title: "[BACKEND] "
labels: enhancement, new-backend
assignees: ''
---

## Backend name

<!-- e.g. SummaryMemory, EntityMemory, RedisMemory -->

## What strategy does it use?

<!-- Describe in 2-3 sentences how this backend stores and retrieves messages -->

## Why is it interesting to benchmark?

<!-- What hypothesis does this backend test? What trade-off does it make? -->

## Implementation sketch

```python
class YourMemory(BaseMemory):
    name = "your_backend"

    def add_message(self, role, content, turn): ...
    def get_context(self, query, current_turn): ...
    def reset(self): ...
```

## Dependencies required

<!-- Any new pip packages? Keep them minimal. -->

## Are you claiming this to implement?

- [ ] Yes — I'll open a PR within 2 weeks
- [ ] No — leaving it open for the community

<!-- See CONTRIBUTING.md for the full backend implementation guide -->
