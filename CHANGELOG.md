# Changelog

All notable changes to MemoryLens are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added — Multi-Provider Real LLM Evaluation (`feat/multi-provider-llm-eval`)
- `utils/providers.py` — unified LLM abstraction layer supporting **5 providers**:
  - **Groq** (`GROQ_API_KEY`) — free tier, llama-3.1-8b-instant
  - **OpenAI** (`OPENAI_API_KEY`) — gpt-4o-mini
  - **Anthropic** (`ANTHROPIC_API_KEY`) — claude-haiku-4-5
  - **OpenRouter** (`OPENROUTER_API_KEY`) — 200+ models via one key
  - **Ollama** (local, no key) — any locally running model
  - Auto-detection priority: Groq → OpenAI → Anthropic → OpenRouter → Ollama
- **Two-stage LLM evaluation pipeline** in `evaluation/metrics.py`:
  - `llm_recall_at_t()` — LLM answers the query; a judge LLM call verifies correctness
  - `llm_temporal_drift()` — checks if LLM returns old vs new value after a fact update
- **CLI flags** in `main.py`:
  - `--llm` — enable real LLM evaluation pass
  - `--provider <name>` — force a specific provider
  - `--list-providers` — print availability of all providers and exit
- **Three-table CLI output**: Content Recall, LLM Recall, and Gap (Content − LLM)
- **Dashboard updates**:
  - Provider selector in sidebar (auto-detects available providers)
  - Tabbed recall chart: Content Recall / LLM Recall / Gap
  - KPI cards show LLM Recall with gap delta when available
  - `summary` backend added to backend multiselect
- Updated `.env.example` with all five provider keys and inline documentation

### Added — SummaryMemory Backend
- `memory/summary.py`: SummaryMemory backend — rolling compression memory with dual-mode support:
  - **LLM mode** (when `GROQ_API_KEY` is set): Groq-powered abstractive summarisation
  - **Extractive fallback** (zero API cost): regex-based fact-pattern extraction
- 6 new tests in `tests/test_pipeline.py` covering SummaryMemory: recall, compression, context structure, reset, token cost, and benchmark registration
- `SummaryMemory` registered as `"summary"` in `evaluation/benchmark.py`

### Results (extractive mode, 100 turns)
| Backend | Recall@100 | Tokens/Query |
|---------|:----------:|:------------:|
| SummaryMemory | 100% | 318 |

---

## [0.2.0] — 2026-05-22

### Added
- `cascade_efficiency` metric: recall-per-token ratio of Cascading vs Naive (5.45× at T=100)
- `quick_demo.py`: full benchmark run requiring zero API keys — uses local embeddings only
- `evaluation/llm_judge.py`: optional Groq-powered LLM-as-judge for answer quality scoring
- `evaluation/logger.py`: persist every benchmark run to `experiment_logs/` as JSON + CSV
- GitHub Actions CI workflow — tests on Python 3.10 and 3.11

### Fixed
- Cascading cold-tier recall: now surfaces **all** cold summaries instead of only the last two
- CI `IndentationError`: replaced inline `python -c` with proper test scripts

### Changed
- Naive memory context budget tightened to 1,200 tokens to expose realistic decay
- Demo results updated with empirically validated numbers from live benchmark runs

---

## [0.1.0] — 2026-05-21

### Added
- Initial framework with three memory backends: Naive, RAG, Cascading Temporal
- Five evaluation metrics: Recall@T, Precision@K, Temporal Drift, Memory Noise Ratio, Cascade Efficiency
- Streamlit dashboard with demo data and live benchmark mode
- CLI runner (`main.py`) with JSON and LaTeX export
- Synthetic conversation simulator with 8 tracked facts and update events
- `sentence-transformers` (all-MiniLM-L6-v2) for local, free embeddings
- Groq integration (llama-3.1-8b-instant) for LLM evaluation mode
