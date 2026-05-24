---
name: New Domain Scenario
about: Propose a new benchmark domain (medical, legal, HR, finance, etc.)
title: "feat: <DomainName>Scenario — X-fact domain benchmark"
labels: "feature, area: simulator, status: open-for-contribution, good first issue, difficulty: beginner"
assignees: ""
---

## Domain name

<!-- e.g. Medical, Legal, HR, Finance, Customer Support, Gaming, Education -->

## Why this domain?

<!-- Why is this domain a useful benchmark for LLM memory decay?
     What kinds of fact updates are common and important in this domain?
     e.g. "Medical: diagnosis changes and medication updates are high-stakes" -->

## Proposed facts (8 facts)

<!-- List 8 trackable facts. At least 2 should have natural update events.
     Follow the Fact dataclass signature: Fact(key, value, injected_at, updated_at?, updated_value?)
     Example:
     - name: "Alice Chen" (injected_at=0)
     - diagnosis: "acid reflux" → "costochondritis" (injected_at=5, updated_at=30)
-->

1.
2.
3.
4.
5.
6.
7.
8.

## Sample filler questions (5 examples)

<!-- What does the human side of this conversation sound like between fact injections? -->

1.
2.
3.
4.
5.

## Implementation notes

<!-- Any special considerations for this domain?
     e.g. "Medical terms may need exact-match checking" or "Legal scenarios have multiple parties" -->

## Reference

<!-- Existing scenario to copy as a template: simulator/scenarios/edtech.py
     Full guide: https://github.com/Neal006/memorylens/blob/main/CONTRIBUTING.md#how-to-add-a-new-domain-scenario -->
