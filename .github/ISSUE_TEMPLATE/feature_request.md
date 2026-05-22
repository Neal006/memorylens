---
name: Feature Request
about: Propose a new feature, LLM memory backend, evaluation metric, or benchmark scenario for MemoryLens
title: "[FEAT] "
labels: enhancement
assignees: ''
---

## What problem does this solve?

<!-- Describe the gap in LLM memory evaluation. What can't MemoryLens measure right now?
     Examples:
     - "There's no way to benchmark memory across sessions (persistent memory)"
     - "Chunked RAG doesn't model importance-weighted eviction"
     - "The EdTech scenario is missing — I want to benchmark student/teacher memory decay"
-->

## Proposed solution

<!-- How would you implement this? New memory backend? New metric? New benchmark scenario? Dashboard change? -->

## Which layer does this touch?

- [ ] `simulator/` — conversation generation, fact injection, or new domain scenario
- [ ] `memory/` — new or improved memory backend (LLM memory architecture)
- [ ] `evaluation/` — new metric or multi-seed benchmark change
- [ ] `utils/providers.py` — new LLM provider
- [ ] `dashboard.py` — visualisation (Streamlit + Plotly)
- [ ] `main.py` / CLI
- [ ] Documentation / research paper

## Expected impact on recall or efficiency

<!-- If this is a new memory backend, what recall@T behavior do you expect?
     If this is a new metric, what does it capture that Recall@T doesn't?
-->

## Alternatives considered

<!-- Have you tried working around this? Which existing backends or metrics come closest? -->

## Are you willing to implement this?

- [ ] Yes, I'd like to open a PR for this
- [ ] I can help review a PR
- [ ] I'm just proposing — happy for someone else to pick it up

## Additional context

<!-- Links to papers (MemGPT, A-MEM, RAGAS, etc.), related LLM memory work, mockups, etc. -->
