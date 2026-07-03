<div align="center">

# 🔭 MemoryLens

### Measure how your AI's memory forgets

**An open-source benchmark for LLM memory decay — 8 memory architectures, 6 metrics, 4 domain scenarios, statistical rigor, zero API keys required.**

[![CI](https://github.com/Neal006/memorylens/actions/workflows/ci.yml/badge.svg)](https://github.com/Neal006/memorylens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/memorylens)](https://pypi.org/project/memorylens/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

[**Install**](#install) · [**Quick Start**](#quick-start) · [**Results**](#benchmark-results) · [**How It Works**](#how-it-works) · [**Contributing**](#contributing)

</div>

---

## The Problem

Every LLM application that runs multi-turn conversations has a memory strategy — usually "keep everything in context and hope." Almost nobody measures what that strategy actually remembers after 50, 100, or 200 turns.

MemoryLens answers three questions:

- **How much does an AI still recall** after N conversation turns — and at what token cost?
- **Which memory architecture** (full history, RAG, tiered compression, entity store, knowledge graph…) retains facts most efficiently?
- **When a user updates a fact** ("I moved to Mumbai"), does the memory surface the new value, the stale one — or contradict itself with both?

Every core metric is content-based and deterministic: no API key, fully reproducible. Add any LLM key and a two-stage answer+judge pipeline measures what the model *actually* answers.

---

## Install

```bash
pip install memorylens
```

| Extra | Installs | For |
|-------|----------|-----|
| `memorylens[dashboard]` | streamlit, plotly, pandas | Interactive dashboard |
| `memorylens[server]` | fastapi, uvicorn | REST API |
| `memorylens[faiss]` | faiss-cpu | FAISS vector backend |
| `memorylens[groq]` / `[openai]` / `[anthropic]` | provider SDK | LLM evaluation mode |
| `memorylens[all]` | everything above | — |

Python 3.10–3.13 · Linux, macOS, Windows (all tested in CI).

---

## Quick Start

### CLI — results in under a minute, no API key

```bash
memorylens                        # 100-turn benchmark, 3 backends
memorylens --seeds 5              # multi-seed: mean ± std across 5 personas
memorylens --scenario medical     # domain scenarios: default | edtech | support | medical
memorylens --backends naive rag_chunked cascading graph
memorylens --seeds 5 --fit-curves # fit Ebbinghaus + exponential forgetting curves
memorylens --llm --provider groq  # real answer+judge LLM evaluation
```

### Python API

```python
from memorylens import run_benchmark, results_to_display_dict

raw = run_benchmark(total_turns=100, backends=["naive", "rag", "cascading"])
results = results_to_display_dict(raw)
print(results["cascading"]["recall"])   # recall at each checkpoint
```

### Dashboard and REST API

```bash
pip install "memorylens[dashboard]"
streamlit run dashboard.py            # decay curves, run history, cost projection

pip install "memorylens[server]"
uvicorn memorylens.api:app            # POST /v1/benchmarks → job id → poll results
```

---

## Benchmark Results

All numbers below are reproduced from this exact code (v0.4.0) with:
`memorylens --seeds 5 --backends naive rag rag_chunked cascading summary entity graph faiss`

### Recall@T — 100 turns (mean ± std, n=5 personas)

| Backend | T=10 | T=50 | T=100 | Tokens/query @ T=100 |
|---------|:----:|:----:|:-----:|:--------------------:|
| naive (1,200-token budget) | 100% | 100% | **35.0 ± 5.6%** | 1,193 |
| rag | 100% | 100% | 100% | 67 |
| rag_chunked | 100% | 100% | 100% | 56 |
| cascading | 100% | 100% | 100% | 326 |
| summary *(extractive)* | 100% | 100% | 100% | 356 |
| entity | 100% | 100% | 100% | 63 |
| graph | 100% | 100% | 100% | 62 |
| faiss | 100% | 100% | 100% | 67 |

**Read this before quoting numbers.** At 100 turns with 8 templated facts, every retrieval- and extraction-based backend saturates — the differentiation is *token cost at equal recall* (rag_chunked delivers the same recall as naive at ~1/21 the tokens) and naive's collapse once its context budget evicts early turns. Two structural caveats:

1. **entity / graph exploit the fact templates.** Facts are injected as `"My X is Y"`, which the regex extractors match by construction. Their 100% shows structured stores never lose what they capture — it says nothing about free-form conversation. Harder paraphrased injections are on the [roadmap](ROADMAP.md).
2. **summary is extractive in zero-key mode** — it keeps fact-bearing lines verbatim, so substring recall is guaranteed. With an LLM key, compression becomes abstractive and honest decay appears.

### Stress test — 200 turns

At 200 turns the bounded systems saturate and real decay separates the field
(`memorylens --turns 200 --checkpoints 10 50 100 150 200 --seeds 5 --backends ...`):

| Backend | T=100 | T=150 | T=200 | Tokens/query @ T=200 |
|---------|:-----:|:-----:|:-----:|:--------------------:|
| naive | 35.0 ± 5.6% | 7.5 ± 6.9% | **7.5 ± 6.9%** | 1,189 |
| rag_chunked (bounded index) | 100% | 17.5 ± 6.9% | **5.0 ± 6.9%** | 70 |
| cascading | 100% | 100% | **100%** | 326 |
| rag · summary · entity · graph · faiss | 100% | 100% | 100% | 62–382 |

The two production-realistic constraints fail in opposite ways: naive's fixed token budget evicts old turns wholesale, while rag_chunked's bounded FIFO vector index (200 chunks) silently drops the earliest facts even though each query costs only 70 tokens. Cascading survives because its cold tier compresses fact-bearing lines instead of discarding them.

### Temporal drift and contradiction after fact updates

Two facts update mid-conversation (city at T=40, age at T=60). At T=100:

| Backend | Drift (stale-only retrieval) | Contradiction (old + new surfaced) |
|---------|:---------------------------:|:----------------------------------:|
| naive | 0.0 | 1.0 — full history keeps both values |
| cascading | 0.0 | 0.0 — cold summaries patched in place |
| entity / graph | 0.0 | 0.0 — updates overwrite in place |

> Drift is a retrieval-layer proxy (worst-case bound). For behavioral measurement — does the model *answer* with the stale value — run `memorylens --llm`.

---

## How It Works

```
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — SIMULATOR                                                    │
│  Injects facts at known turns, fires domain filler queries in between   │
│  Facts can be updated mid-conversation to test drift + contradiction    │
│                                                                         │
│  T=0  "My name is Arjun Sharma."                                        │
│  T=1  "My city is Bangalore."                                           │
│  T=40 "My city has changed to Mumbai."   ← update event                │
│  Scenarios: default (tech Q&A) · edtech · support · medical             │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — MEMORY BACKENDS (8 implementations)                          │
│                                                                         │
│  naive        Full history, evict oldest at a token budget              │
│  rag          Embed every message, retrieve top-K (upper bound)         │
│  rag_chunked  Chunked + bounded FIFO index (production-realistic)       │
│  cascading    Hot/Warm/Cold tiers with Ebbinghaus temporal decay        │
│  summary      Rolling compression (LLM or extractive)                   │
│  entity       Structured key-value fact store                          │
│  graph        NetworkX knowledge graph, in-place fact updates           │
│  faiss        FAISS vector index (optional dependency)                  │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — EVALUATOR (6 metrics, dual mode)                             │
│                                                                         │
│  Content mode (no API key): deterministic substring checks              │
│  LLM mode (any provider):   answer+judge pipeline — did the LLM         │
│                             actually answer correctly?                  │
└────────────────────────────────────────────────────────────────────────┘
```

### The 6 Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Recall@T** | Is the correct current fact value in retrieved context at turn T? |
| **Precision@K** | Of K retrieved chunks, how many contain a real fact? |
| **Temporal Drift** | After an update, what fraction of retrieval still carries the stale value? *(worst-case proxy)* |
| **Contradiction** | Does the context surface both old *and* new values at once, forcing the LLM to arbitrate? |
| **Memory Noise Ratio** | What fraction of retrieved context is irrelevant to any known fact? |
| **Cascade Efficiency** | Recall-per-token vs the naive baseline |

### The 4 Temporal Decay Functions

The cascading backend's warm-tier scoring uses a pluggable forgetting curve — compare them with `--decay`:

| Name | Formula | Reference |
|------|---------|-----------|
| `ebbinghaus` *(default)* | `e^{-t / (S·√(1+t))}` | Ebbinghaus (1885) |
| `exponential` | `e^{-k·t/window}` | Jost (1897) |
| `linear` | `1 − t/window` | Wickelgren (1972) |
| `default` | `max(0.2, 1 − 0.6·t/w)` | Original heuristic |

`--fit-curves` fits both Ebbinghaus and exponential models to your measured Recall@T series and reports half-life, stability, and R².

---

## LLM Provider Support

All content metrics work with **no API key**. Add any one key for the LLM answer+judge pass:

| Provider | Key | Default Model | Free Tier |
|----------|-----|---------------|-----------|
| Groq | `GROQ_API_KEY` | llama-3.1-8b-instant | ✅ |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | ❌ |
| Anthropic | `ANTHROPIC_API_KEY` | claude-haiku-4-5 | ❌ |
| OpenRouter | `OPENROUTER_API_KEY` | llama-3.1-8b-instruct:free | ✅ |
| Ollama | *(none — local)* | llama3.2 | ✅ |

```bash
memorylens --list-providers
memorylens --llm                  # auto-detect
memorylens --llm --provider groq  # force one
```

---

## How MemoryLens Compares

Most evaluation frameworks measure quality at a single point in time. MemoryLens measures how memory quality **changes over conversation turns** — a dimension the tools below don't cover (they solve different problems, and compose well with this one):

| Framework | Focus | Decay over turns | Multi-architecture | No-API mode |
|-----------|-------|:----------------:|:------------------:|:-----------:|
| **MemoryLens** | Memory decay | ✅ | ✅ 8 backends | ✅ |
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG answer quality | ❌ | ❌ | ❌ |
| [TruLens](https://github.com/truera/trulens) | LLM app monitoring | ❌ | ❌ | ❌ |
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM answer quality | ❌ | ❌ | Partial |
| [MemGPT / Letta](https://github.com/cpacker/MemGPT) | A memory *system*, not a benchmark | — | — | — |

Details: [docs/comparison-with-existing-tools.md](docs/comparison-with-existing-tools.md)

---

## Project Structure

```
memorylens/              installable package
├── memory/              8 backends + decay functions (add yours here)
├── simulator/           facts, conversation generator, personas, scenario registry
├── evaluation/          metrics, benchmark runner, stats, experiment logger
├── utils/               local embeddings, LLM providers, SQLite run storage
├── api.py               FastAPI REST server
└── cli.py               `memorylens` command

tests/                   39 integration tests, no API key needed
dashboard.py             Streamlit dashboard (decay curves, run history, export)
docs/                    guides: adding a backend, comparisons, methodology
```

---

## Contributing

A new memory backend is 3 methods and one registry line — the full guide with a worked example is in [docs/adding-a-new-backend.md](docs/adding-a-new-backend.md):

```python
from memorylens.memory.base import BaseMemory

class YourMemory(BaseMemory):
    name = "your_backend"
    def add_message(self, role, content, turn): ...
    def get_context(self, query, current_turn): ...
    def reset(self): ...
```

New scenarios are a data file plus one registry entry. New metrics are plain functions.

```bash
git clone https://github.com/Neal006/memorylens && cd memorylens
pip install -e ".[server,dev]"
pytest tests -q          # all green before you start
```

Start here: [`good first issue`](https://github.com/Neal006/memorylens/issues?q=label%3A%22good+first+issue%22) · Guide: [CONTRIBUTING.md](CONTRIBUTING.md) · Plans: [ROADMAP.md](ROADMAP.md)

---

## Citation

```bibtex
@software{memorylens2026,
  author  = {Daftary, Neal},
  title   = {{MemoryLens}: A Temporal Decay Benchmark for {LLM} Memory Architectures},
  year    = {2026},
  url     = {https://github.com/Neal006/memorylens},
  version = {0.4.0}
}
```

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

<div align="center">

**If MemoryLens is useful to you, a ⭐ helps other researchers and developers find it.**

</div>
