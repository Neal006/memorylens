# Why LLM Memory Evaluation Matters

> This document explains the **LLM memory decay problem** — why it exists, why no one is measuring it, and what MemoryLens does about it.

---

## The Invisible Bug in Every LLM Application

Every production chatbot, coding assistant, or AI agent that handles multi-turn conversations faces the same silent failure: **memory decay**.

Here is a concrete example. A user tells their AI assistant:

```
Turn 1:  "My name is Sofia and I live in Mexico City."
Turn 35: "What's a good weekend activity in my city?"
```

If the AI's memory system evicted turn 1 to make room for turns 2–34, it has no idea what city Sofia lives in. It will either hallucinate an answer or ask again — both are bad outcomes.

This is not a hypothetical. It is the default behavior of naive context-window management, and most production applications use it.

---

## Why LLM Memory Decay Is Hard to Measure

The LLM memory decay problem has been ignored for three reasons:

### 1. Single-point evaluation is easier
Most evaluation frameworks (RAGAS, DeepEval, TruLens) measure quality at a *single point in time*. They ask: "Given this context, did the LLM answer correctly?" They don't ask: "After 100 conversation turns, does the memory system still have the right context to pass to the LLM?"

### 2. Retrieval quality ≠ answer quality
A memory system can successfully retrieve a chunk containing a fact value — but the chunk may lack surrounding context, making the LLM unable to extract the answer correctly. This gap between **content recall** (does the token exist in context?) and **LLM recall** (can the model answer the question?) is the key insight that MemoryLens's LLM evaluation pipeline measures.

The same gap applies to the Temporal Drift metric: content-level drift (does the old value appear in retrieved chunks?) is a worst-case proxy that *overestimates* actual stale-answer rates. The LLM-based drift measurement tells you whether the model actually responds with the stale value.

### 3. Results depend on the conversation
Memory decay is not a property of the model — it's a property of the *conversation*. A RAG system that retrieves perfectly for 50 turns may fail at turn 200 when index capacity is exceeded and the most important early facts are evicted. This temporal dimension requires benchmarking over time, not at a snapshot.

---

## What MemoryLens Measures

### Recall@T — Does the right fact survive in memory?

$$\text{Recall@T}(f, T) = \mathbf{1}\left[\text{current\_value}(f) \in \text{context retrieved at turn } T\right]$$

This tells you: of the facts the user explicitly shared, what fraction can the system still surface at turn T?

### Temporal Drift — Does the system still surface the old value?

When a user says "I moved to Mumbai", does the memory system still retrieve the old "Bangalore" message? Temporal Drift measures the proportion of stale-to-fresh fact appearances in retrieved context:

$$\text{Drift} = \frac{\text{stale hits}}{\text{stale hits} + \text{fresh hits}}$$

A Drift of 1.0 means retrieved context contains only the stale value. A Drift of 0.0 means it surfaces only the updated value.

**Important: this is a retrieval-layer proxy, not a behavioral measurement.** If a memory system retrieves both `"My city is Bangalore"` and `"My city has changed to Mumbai"`, Drift = 0.5 — but the LLM may still answer "Mumbai" correctly, because the explicit update message is a strong signal. The content drift metric therefore *overestimates* real-world stale-answer rates.

For the behavioral measurement (does the LLM actually answer with the old or new value?), MemoryLens provides a second metric, `llm_temporal_drift`, which is run in LLM mode:

```bash
python main.py --llm   # enables the answer+judge pipeline for all drift measurements
```

The two drift metrics together give you both the worst-case bound (content) and the real-world answer quality (LLM). On most well-structured backends, the LLM drift is 15–30% lower than content drift because the model correctly resolves contradictions when both values are present.

### Cascade Efficiency — How much recall does each token buy?

The most practical metric for production systems:

$$\text{CascEff}(T) = \frac{\text{Recall}_\text{cascading}(T) / \text{Tokens}_\text{cascading}(T)}{\text{Recall}_\text{naive}(T) / \text{Tokens}_\text{naive}(T)}$$

A value of 5.67× means the Cascading architecture delivers 5.67 times more recall per token than the naive baseline — the same information accessed at 1/5.67 the inference cost.

---

## The Ebbinghaus Connection

Hermann Ebbinghaus (1885) established empirically that human memory follows a forgetting curve:

$$R(t) = e^{-t/S}$$

where *S* is memory stability and *t* is time elapsed. MemoryLens's Cascading Temporal Memory uses this curve directly to weight warm-tier retrieval:

```python
decay = exp(-age / (stability * sqrt(1 + age)))
```

This is not decorative. Ablation experiments show that the Ebbinghaus curve outperforms ad-hoc linear decay by 4% in cascade efficiency — because it correctly models the steep initial forgetting followed by a flattening retention curve for consolidated memories.

---

## Related Work

- **MemGPT (Packer et al., 2023)** — virtual context management for LLMs. MemoryLens benchmarks the *problem MemGPT solves*, making it a natural evaluation harness for MemGPT-style systems.
- **RAGAS (Es et al., 2023)** — evaluates RAG quality at a point in time. MemoryLens extends this to measure how RAG quality changes over 100 turns.
- **A-MEM (Xu et al., 2024)** — agentic memory with dynamic note structures. MemoryLens can be used to evaluate A-MEM-style systems.

---

*Back to main repository: [github.com/Neal006/memorylens](https://github.com/Neal006/memorylens)*
