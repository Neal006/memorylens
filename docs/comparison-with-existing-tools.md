# MemoryLens vs Other LLM Evaluation Frameworks

> A detailed comparison of MemoryLens against RAGAS, TruLens, DeepEval, MemGPT, and LangChain memory modules — focusing on what each tool actually measures.

---

## Summary Table

| | **MemoryLens** | RAGAS | TruLens | DeepEval | MemGPT | LangChain Memory |
|---|---|---|---|---|---|---|
| **Primary focus** | Memory decay over time | RAG quality | LLM app quality | LLM answer quality | Memory system | Memory implementation |
| **Temporal evaluation** | ✅ Core feature | ❌ | ❌ | ❌ | N/A | N/A |
| **Multi-architecture comparison** | ✅ 5 backends | ❌ | ❌ | ❌ | N/A | N/A |
| **No API key mode** | ✅ Full benchmark | ❌ | ❌ | Partial | ❌ | ❌ |
| **Decay formula** | ✅ Ebbinghaus (1885) | N/A | N/A | N/A | N/A | N/A |
| **Statistical validation** | ✅ n=5, mean ± std | ❌ | ❌ | ❌ | N/A | N/A |
| **Temporal drift metric** | ✅ | ❌ | ❌ | ❌ | N/A | N/A |
| **Token efficiency metric** | ✅ Cascade Efficiency | ❌ | Partial | ❌ | N/A | N/A |
| **Open source** | ✅ MIT | ✅ MIT | ✅ MIT | ✅ MIT | ✅ MIT | ✅ MIT |

---

## RAGAS vs MemoryLens

[RAGAS](https://github.com/explodinggradients/ragas) evaluates Retrieval-Augmented Generation pipelines across four dimensions: faithfulness, answer relevance, context precision, and context recall. It is the most widely used RAG evaluation framework.

**What RAGAS does well:**
- Measures whether a RAG pipeline's answers are faithful to retrieved context
- Works with any RAG pipeline via a clean Python API
- Has strong LLM-as-judge implementations for each metric

**What RAGAS does not measure:**
- How RAG recall *changes over time* as conversations grow
- Whether a memory system still surfaces a fact injected 100 turns ago
- The trade-off between token cost and recall fidelity
- How fact updates propagate through memory (temporal drift)

**Use both:** RAGAS is the right tool for evaluating the quality of a RAG pipeline at a point in time. MemoryLens is the right tool for evaluating how that quality degrades as conversation history accumulates.

---

## TruLens vs MemoryLens

[TruLens](https://github.com/truera/trulens) provides continuous evaluation for LLM applications, with a UI for tracking metrics across runs.

**What TruLens does well:**
- Continuous monitoring of LLM apps in production
- Rich UI for exploring evaluation results
- Integration with LangChain, LlamaIndex, and other frameworks

**What TruLens does not measure:**
- Temporal decay of memory — there is no concept of "turn T" or "checkpoint"
- Cross-architecture comparison of memory strategies
- Token efficiency or recall-per-token

---

## DeepEval vs MemoryLens

[DeepEval](https://github.com/confident-ai/deepeval) is a comprehensive LLM testing framework with 14+ metrics including G-Eval, hallucination detection, and contextual recall.

**What DeepEval does well:**
- Wide range of LLM quality metrics
- pytest-compatible test runner
- Synthetic data generation for evaluation

**What DeepEval does not measure:**
- Memory decay over conversation turns
- The "how much does my AI forget?" question
- Temporal drift after fact updates

---

## MemGPT vs MemoryLens

[MemGPT](https://github.com/cpacker/MemGPT) (now Letta) is a memory *architecture* — it implements virtual context management with paging between in-context and external storage, analogous to OS virtual memory.

MemoryLens and MemGPT are **complementary, not competing**:
- MemGPT is a memory *system* you can deploy
- MemoryLens is an evaluation *framework* that can benchmark MemGPT-style systems

If you are building a MemGPT-inspired memory system, MemoryLens provides the benchmark to measure whether your system actually outperforms simpler alternatives.

---

## LangChain Memory Modules vs MemoryLens

LangChain provides several memory implementations: `ConversationBufferMemory`, `ConversationSummaryMemory`, `ConversationBufferWindowMemory`, `VectorStoreRetrieverMemory`.

These are **implementations**, not benchmarks. They provide the memory storage but do not measure how well they perform over time. MemoryLens can benchmark LangChain memory modules — they implement the 3-method `BaseMemory` interface and can be wrapped as backends.

---

## When to Use MemoryLens

Use MemoryLens when you need to answer any of these questions:

1. **"How many conversation turns can my memory system handle before recall degrades?"**
2. **"Is my RAG memory actually better than a simple sliding window at the 90-turn mark?"**
3. **"When a user updates a fact mid-conversation, does my system propagate the update or keep returning stale answers?"**
4. **"How much does chunking cost me in recall compared to whole-message indexing?"**
5. **"Which temporal decay formula is most appropriate for my use case?"**

---

*Full benchmark and source code: [github.com/Neal006/memorylens](https://github.com/Neal006/memorylens)*
