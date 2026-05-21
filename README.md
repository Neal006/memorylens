# 🔭 MemoryLens

**An Evaluation Framework for LLM Memory Decay**

> *You can't improve what you can't measure. Nobody is measuring memory.*

MemoryLens is an open-source benchmark harness that measures, compares, and visualises how different LLM memory architectures degrade over long conversations. It answers questions no existing framework does:

- How much does an AI "remember" after 10 conversations? After 100?
- When does memory become noise instead of signal?
- Which architecture retains the most useful context at the lowest token cost?

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourname/memorylens.git
cd memorylens
pip install -r requirements.txt

# 2. Set your Groq API key (free at console.groq.com)
cp .env.example .env
# edit .env and set GROQ_API_KEY=...

# 3. Launch the dashboard
streamlit run dashboard.py
# → click "📊 Demo" for instant results, or "▶ Run Live" for real evaluation

# 4. Or run the CLI benchmark
python main.py --turns 100 --backends naive rag cascading
```

---

## Architecture

```
memorylens/
├── simulator/          # Synthetic conversation + fact injection engine
│   ├── facts.py        # Benchmark facts with optional update events
│   └── conversation.py # Multi-turn conversation generator
├── memory/             # Three pluggable memory backends
│   ├── naive.py        # Full history — baseline
│   ├── rag.py          # Semantic retrieval (sentence-transformers)
│   └── cascading.py    # Cascading Temporal Memory (hot/warm/cold tiers)
├── evaluation/
│   ├── metrics.py      # Content-based, reproducible metric functions
│   └── benchmark.py    # Orchestrator — runs all backends + checkpoints
├── dashboard.py        # Streamlit visualisation layer
├── main.py             # CLI entry point
└── demo_results.json   # Pre-computed results for zero-API-key demo
```

---

## Evaluation Metrics

| Metric | Definition | How Measured |
|--------|-----------|--------------|
| **Recall@T** | Can the memory surface fact X after T turns? | Context contains expected value — no LLM call needed |
| **Precision@K** | Of K retrieved chunks, what fraction is relevant? | Chunk content vs known fact values |
| **Temporal Drift** | After a fact update, does old stale data leak through? | Old-value hits vs new-value hits in context |
| **Memory Noise Ratio** | Off-topic retrieval: irrelevant chunks / total chunks | Off-topic query against fact-laden memory |
| **Token Cost** | Avg tokens per query for this memory strategy | Character count of `get_context()` output |

All primary metrics are **content-based** — they do not require LLM inference calls, making benchmarks fast, deterministic, and reproducible.

---

## Memory Backends

### Naive Memory
Keeps the entire conversation history. Truncates oldest messages when over the token budget. Simple but expensive — context grows linearly with conversation length.

### RAG Memory
Embeds every message with `all-MiniLM-L6-v2`. On retrieval, computes cosine similarity and returns top-K semantic matches plus a recency window. Constant token cost regardless of conversation length.

### Cascading Temporal Memory *(novel)*
Three-tier architecture with temporal decay:

```
HOT  (last 12 msgs)  → verbatim, always in context
WARM (last 30 msgs)  → full text, retrieved semantically with age decay
COLD (summaries)     → extractive compression of ancient context
```

Cascade flow: `HOT overflow → WARM → extractive summary → COLD`

Age-decay formula: `effective_score = similarity × max(0.2, 1 - age/total × 0.6)`

---

## Benchmark Results (Demo)

| Backend | T=10 | T=25 | T=50 | T=75 | T=100 | Tokens/Query | Monthly Cost* |
|---------|------|------|------|------|-------|-------------|---------------|
| Naive | 87.5% | 75.0% | 62.5% | 50.0% | 43.8% | 6,400 | ₹5,31,200 |
| RAG | 90.0% | 87.5% | 81.3% | 78.8% | 76.3% | 330 | ₹27,390 |
| **Cascading** | **92.5%** | **91.3%** | **90.0%** | **88.8%** | **87.5%** | **250** | **₹20,750** |

\*At 100K queries/month, ₹83/1M tokens

**Result: Cascading Temporal Memory delivers 96% cost reduction vs Naive while achieving 43.7pp higher recall at T=100.**

---

## Research Export

The dashboard's **⬇ LaTeX table** button exports benchmark tables ready for arXiv submission.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (llama-3.1-8b-instant) — free tier |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) — local |
| Visualisation | Streamlit + Plotly |
| Data | NumPy, Pandas |
| No vector DB needed | Pure NumPy cosine similarity |

---

## Roadmap

- [ ] LLM-as-judge evaluation mode (answer quality, not just content match)
- [ ] Qdrant integration for scale testing
- [ ] RAGAS integration for RAG-specific metrics
- [ ] LangGraph orchestration layer
- [ ] HuggingFace dataset card + arXiv preprint

---

## License

MIT
