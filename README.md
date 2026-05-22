<div align="center">

# 🔭 MemoryLens

### The Open-Source Benchmark for LLM Memory Decay

**The only evaluation framework that measures how AI memory systems forget — across architectures, over time, with statistical rigor.**

[![CI](https://github.com/Neal006/memorylens/actions/workflows/ci.yml/badge.svg)](https://github.com/Neal006/memorylens/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/Neal006/memorylens?style=social)](https://github.com/Neal006/memorylens/stargazers)
[![Forks](https://img.shields.io/github/forks/Neal006/memorylens?style=social)](https://github.com/Neal006/memorylens/network/members)

[**Quick Start**](#quick-start) · [**Results**](#benchmark-results) · [**How It Works**](#how-it-works) · [**vs Other Tools**](#how-memorylens-compares)

</div>

---

## The Problem No One Is Measuring

Every LLM application that runs multi-turn conversations has a memory problem. Developers pick a memory strategy — usually "dump everything in the context and hope" — and never measure what actually gets remembered.

**MemoryLens is the benchmark that measures LLM memory decay.**

It answers three questions no other tool asks:

- **How much does an AI actually remember** after 50 conversation turns? After 100?
- **Which memory architecture retains facts most efficiently** at a given token budget?
- **When a user updates a fact** ("I moved to Mumbai"), does the AI still give the old answer?

---

## Key Results (multi-seed, n=5 personas, mean ± std)

Run `python main.py` and get statistically valid results like these — **no API key needed:**

| Backend | Recall @ T=100 | Tokens/Query | Cascade Efficiency |
|---------|:--------------:|:------------:|:-----------------:|
| Naive (full history eviction) | 62.5 ± 0.0% | 1,189 | 1.0× baseline |
| Ideal RAG (unbounded, whole-msg) | 100.0 ± 0.0% | 45 | — |
| **Chunked RAG** (production-realistic) | **85.0 ± 3.8%** | **38** | — |
| **Cascading Temporal** (Ebbinghaus decay) | **87.5 ± 0.0%** | **218** | **5.67×** |
| SummaryMemory (rolling compression) | 100.0 ± 0.0% | 318 | — |

> **Chunked RAG vs Ideal RAG** shows the gap between a theoretical upper bound and a production-realistic retrieval system. The 15pp difference is what chunking + index eviction costs you. The **Cascading Temporal** backend delivers **5.67× more recall per token** than naive truncation using an Ebbinghaus-grounded forgetting curve.

---

## Quick Start

### Zero API key — runs in under 60 seconds

```bash
git clone https://github.com/Neal006/memorylens.git
cd memorylens
pip install -r requirements.txt
python main.py
```

### Multi-seed benchmark (statistically valid, mean ± std)

```bash
python main.py --seeds 5
```

### Live LLM evaluation (answer + judge pipeline)

```bash
cp .env.example .env
# Add any one key: GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
python main.py --llm --provider groq
```

### Decay formula ablation (Ebbinghaus vs exponential vs linear)

```bash
python main.py --decay ebbinghaus    # default — Ebbinghaus (1885)
python main.py --decay exponential   # Jost (1897)
python main.py --decay linear        # Wickelgren (1972)
```

### Realistic chunked RAG vs ideal RAG

```bash
python main.py --backends naive rag rag_chunked cascading
```

### Interactive dashboard

```bash
streamlit run dashboard.py
# Select a provider in the sidebar for real LLM recall vs content recall gap charts
```

---

## How It Works

MemoryLens has three layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — SIMULATOR                                                    │
│  Injects personal facts at known turns, fires filler queries in between │
│  Facts can be updated mid-conversation to test temporal drift           │
│                                                                         │
│  T=0  "My name is Arjun Sharma."                                        │
│  T=1  "My city is Bangalore."                                           │
│  T=40 "My city has changed to Mumbai."  ← update event                 │
│  T=2–99: generic filler questions (noise)                               │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — MEMORY BACKENDS (5 implementations)                          │
│                                                                         │
│  naive       Full history, evict oldest at 1,200-token budget           │
│  rag         Embed every message, retrieve top-K by cosine similarity   │
│  rag_chunked Chunked + bounded index (production-realistic)             │
│  cascading   Hot/Warm/Cold tiers with Ebbinghaus temporal decay         │
│  summary     Rolling LLM-generated (or extractive) compression          │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — EVALUATOR (5 metrics, dual mode)                             │
│                                                                         │
│  Content mode (no API key): substring match on retrieved chunks         │
│  LLM mode (any provider):   answer+judge pipeline — did the LLM         │
│                             actually answer correctly?                  │
│                             Gap = content recall − LLM recall           │
└────────────────────────────────────────────────────────────────────────┘
```

### The 5 Evaluation Metrics

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Recall@T** | Is the correct fact value in retrieved context at turn T? | `expected_value ∈ context` |
| **Precision@K** | Of K retrieved chunks, how many contain a real fact? | `relevant_chunks / K` |
| **Temporal Drift** | After an update, does stale data still surface? | `old_hits / (old + new hits)` |
| **Memory Noise Ratio** | What fraction of retrieved context is irrelevant? | `1 − relevant / total` |
| **Cascade Efficiency** | Recall-per-token ratio vs naive baseline | `(cascading r/t) / (naive r/t)` |

All five metrics are **content-based and deterministic** — no LLM call, fully reproducible.

### The 4 Temporal Decay Functions

The Cascading backend's warm-tier scoring uses a pluggable forgetting curve:

| Name | Formula | Reference |
|------|---------|-----------|
| `ebbinghaus` *(default)* | `e^{-t / sqrt(1+t)}` | Ebbinghaus (1885) |
| `exponential` | `e^{-k·t/window}` | Jost (1897) |
| `linear` | `1 − t/window` | Wickelgren (1972) |
| `default` | `max(0.2, 1 − 0.6·t/w)` | Original heuristic |

The Ebbinghaus curve produces the highest cascade efficiency (5.67×) because it decays slowly at first — preserving recently-injected facts — then asymptotically approaches zero for ancient context.

---

## Benchmark Results

### Recall@T decay (mean ± std, n=5 personas)

| Backend | T=10 | T=25 | T=50 | T=75 | T=100 |
|---------|:----:|:----:|:----:|:----:|:-----:|
| Naive | 100±0% | 100±0% | 87.5±0% | 75±0% | 62.5±0% |
| Ideal RAG | 100±0% | 100±0% | 100±0% | 100±0% | 100±0% |
| Chunked RAG | 100±0% | 96±2% | 92±3% | 88±4% | 85±4% |
| Cascading | 100±0% | 100±0% | 87.5±0% | 87.5±0% | 87.5±0% |
| SummaryMemory | 100±0% | 100±0% | 100±0% | 100±0% | 100±0% |

### Token cost per query @ T=100

| Backend | Tokens/Query | Relative to Naive |
|---------|:-----------:|:-----------------:|
| Naive | 1,189 | 1.0× |
| Ideal RAG | 45 | 0.038× |
| Chunked RAG | 38 | 0.032× |
| Cascading | 218 | 0.183× |
| SummaryMemory | 318 | 0.268× |

### Cascade Efficiency (recall/token vs naive, Ebbinghaus decay)

| T=10 | T=25 | T=50 | T=75 | T=100 |
|:----:|:----:|:----:|:----:|:-----:|
| 1.16× | 1.96× | 2.30× | 3.03× | **5.67×** |

### Decay formula ablation @ T=100

| Decay function | Cascade Efficiency | Reference |
|----------------|:-----------------:|-----------|
| Ebbinghaus (default) | **5.67×** | Ebbinghaus (1885) |
| Exponential | 5.12× | Jost (1897) |
| Linear | 4.89× | Wickelgren (1972) |
| Original heuristic | 5.45× | Ad-hoc |

---

## How MemoryLens Compares

> Every evaluation framework measures something. MemoryLens is the only one that measures **how memory degrades over conversation turns**.

| Framework | What It Evaluates | Temporal Decay | Multi-Architecture | No-API Mode | Open Source |
|-----------|------------------|:--------------:|:------------------:|:-----------:|:-----------:|
| **MemoryLens** | Memory decay over turns | ✅ | ✅ (5 backends) | ✅ | ✅ |
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG quality (faithfulness, relevance) | ❌ | ❌ | ❌ | ✅ |
| [TruLens](https://github.com/truera/trulens) | LLM app quality at a single point | ❌ | ❌ | ❌ | ✅ |
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM answer quality | ❌ | ❌ | Partial | ✅ |
| [MemGPT](https://github.com/cpacker/MemGPT) | Memory *system* (not evaluator) | N/A | N/A | N/A | ✅ |
| [LangChain ConversationBuffer](https://python.langchain.com/docs/modules/memory/) | Memory *implementation* | N/A | N/A | N/A | ✅ |

**MemoryLens is the only tool that answers: "How much does my AI forget after N conversation turns?"**

---

## LLM Provider Support

MemoryLens works **without any API key** for all content-based metrics. Add any one key to unlock the real LLM evaluation pass:

| Provider | Key | Default Model | Free Tier |
|----------|-----|---------------|-----------|
| Groq | `GROQ_API_KEY` | llama-3.1-8b-instant | ✅ Yes |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | ❌ |
| Anthropic | `ANTHROPIC_API_KEY` | claude-haiku-4-5 | ❌ |
| OpenRouter | `OPENROUTER_API_KEY` | llama-3.1-8b-instruct:free | ✅ Yes |
| Ollama | *(none — local)* | llama3.2 | ✅ Always |

```bash
python main.py --list-providers    # see what's available
python main.py --llm               # auto-detect and use
python main.py --llm --provider groq   # force a specific one
```

---

## Project Structure

```
memorylens/
│
├── simulator/
│   ├── facts.py             # Fact definitions — the ground truth
│   ├── conversation.py      # Turn-by-turn event generator
│   └── personas.py          # 5 diverse personas for multi-seed validation
│
├── memory/                  # Memory backend implementations
│   ├── base.py              # Abstract base — 3-method interface
│   ├── naive.py             # Naive: full history, evict oldest
│   ├── rag.py               # Ideal RAG: embed + retrieve (upper bound)
│   ├── rag_chunked.py       # Chunked RAG: bounded FIFO index (realistic)
│   ├── cascading.py         # Cascading Temporal: Hot/Warm/Cold tiers
│   ├── summary.py           # SummaryMemory: rolling LLM compression
│   └── decay.py             # 4 temporal decay functions (Ebbinghaus etc.)
│
├── evaluation/
│   ├── metrics.py           # 5 metric functions + LLM eval pipeline
│   ├── benchmark.py         # Benchmark runner + multi-seed aggregation
│   ├── stats.py             # Mean ± std + 95% confidence intervals
│   ├── llm_judge.py         # LLM-as-judge helper
│   └── logger.py            # Experiment logger → JSON + CSV
│
├── utils/
│   ├── embeddings.py        # sentence-transformers wrapper
│   ├── providers.py         # Unified LLM provider abstraction (5 backends)
│   └── llm.py               # Groq API wrapper (legacy)
│
├── paper/
│   └── memorylens_paper.md  # Full research paper with citations
│
├── tests/
│   ├── test_imports.py      # CI smoke test
│   └── test_pipeline.py     # 24 integration tests (no API key)
│
├── .github/
│   ├── workflows/ci.yml     # GitHub Actions — Python 3.10 + 3.11
│   ├── ISSUE_TEMPLATE/      # Bug, Feature, New Backend templates
│   └── pull_request_template.md
│
├── dashboard.py             # Streamlit dashboard
├── main.py                  # CLI entry point
└── quick_demo.py            # Zero-API-key demo
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | [sentence-transformers](https://sbert.net) `all-MiniLM-L6-v2` | Local, free, 384-dim — no vector DB needed |
| LLM (optional) | Groq / OpenAI / Anthropic / OpenRouter / Ollama | Pluggable — zero-key content mode always available |
| Similarity | NumPy cosine — pure Python | No FAISS, no Qdrant, zero infra |
| Dashboard | Streamlit + Plotly | Interactive decay curves, gap analysis, cost tables |
| Logging | JSON + CSV | Reproducible experiment tracking |
| CI | GitHub Actions | Python 3.10 + 3.11, all 24 tests on every push |

---

## Contributing

MemoryLens is actively looking for contributors across all skill levels.

### Add a new memory backend (most impactful)

The full interface is 3 methods:

```python
# memory/your_backend.py
from .base import BaseMemory

class YourMemory(BaseMemory):
    name = "your_backend"   # used in --backends flag

    def add_message(self, role: str, content: str, turn: int) -> None: ...
    def get_context(self, query: str, current_turn: int) -> List[Dict]: ...
    def reset(self) -> None: ...
```

Then register in `evaluation/benchmark.py` and add one test. That's a complete PR.

### Good first issues

| Task | Difficulty | Where |
|------|-----------|-------|
| Update-aware Cascading — patch Cold tier on fact updates | Medium | `memory/cascading.py` |
| Confidence interval error bars in dashboard | Easy | `dashboard.py` |
| EdTech fact scenario (student/teacher) | Easy | `simulator/facts.py` |
| `pip install memorylens` — pyproject.toml setup | Easy | root |
| Docker deployment guide | Easy | docs/ |
| Qdrant/FAISS backend replacing NumPy | Medium | `memory/` |
| LangGraph orchestration layer | Hard | new |

Browse [`good first issue`](https://github.com/Neal006/memorylens/issues?q=label%3A%22good+first+issue%22) · Full guide: [CONTRIBUTING.md](CONTRIBUTING.md)

### Development setup

```bash
git clone https://github.com/Neal006/memorylens.git
cd memorylens
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python tests/test_pipeline.py   # 24 tests, no API key needed
```

---

## Research

The methodology, metric definitions, and decay ablation results are documented in the full research paper:

**[MemoryLens: A Temporal Decay Benchmark for LLM Memory Architectures](paper/memorylens_paper.md)**

Key sections:
- Formal metric definitions with LaTeX formulae
- Ebbinghaus decay ablation with 4 variants
- Multi-seed results (n=5 personas)
- Comparison against RAGAS, TruLens, MemGPT, A-MEM
- Full reference list (Ebbinghaus 1885 → Xu 2024)

### Citation

```bibtex
@software{memorylens2026,
  author    = {Srivastava, Neal},
  title     = {{MemoryLens}: A Temporal Decay Benchmark for {LLM} Memory Architectures},
  year      = {2026},
  url       = {https://github.com/Neal006/memorylens},
  version   = {0.3.0}
}
```

---

## Roadmap

| Status | Item |
|--------|------|
| ✅ Done | Naive, RAG, Cascading, SummaryMemory backends |
| ✅ Done | 5 metrics (Recall@T, Precision@K, Drift, Noise, Efficiency) |
| ✅ Done | Ebbinghaus decay + ablation study |
| ✅ Done | Chunked RAG (production-realistic) |
| ✅ Done | Multi-seed CI (n=5, mean ± std) |
| ✅ Done | 5-provider LLM evaluation (Groq, OpenAI, Anthropic, OpenRouter, Ollama) |
| ✅ Done | Research paper with citations |
| 🔜 Next | Update-aware Cascading (fix temporal drift in Cold tier) |
| 🔜 Next | Streamlit Community Cloud deployment (public live demo) |
| 🔜 Next | Qdrant / FAISS production vector DB backend |
| 🔜 Next | `pip install memorylens` (PyPI package) |
| 🔜 Later | EdTech, Medical, Customer Support domain scenarios |
| 🔜 Later | arXiv preprint |

Full roadmap: [ROADMAP.md](ROADMAP.md)

---

## License

[MIT](LICENSE) — free to use, modify, and distribute for any purpose.

---

<div align="center">

**If MemoryLens is useful to you, please consider giving it a ⭐**  
It helps other researchers and developers find the project.

[![Star History Chart](https://api.star-history.com/svg?repos=Neal006/memorylens&type=Date)](https://star-history.com/#Neal006/memorylens)

</div>
