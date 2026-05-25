# Changelog

All notable changes to MemoryLens are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Documentation — Metric Accuracy Clarifications

Three research-quality gaps identified and documented across README, docs/, and ROADMAP:

- **±0.0% std for structure-deterministic backends** — Naive and Cascading produce std=0% across 5 personas because they make eviction/decay decisions based on *position and elapsed time*, not content values. All personas share identical injection timing, so results are structurally identical. Chunked RAG shows real variance (±3.8%) because cosine similarity scoring is content-dependent. Documented with explanation in README results tables. Fix tracked in ROADMAP as "varied persona injection timing."

- **SummaryMemory 100% recall is extractive-mode behavior** — In zero-API-key mode, SummaryMemory keeps fact-bearing messages verbatim (selective full history, not true compression). 100% recall is therefore expected and not a performance claim. With real LLM compression (any API key), abstractive paraphrasing can cause substring-match failures. Tables and callouts updated to label these results as "(extractive mode)."

- **Temporal Drift is a retrieval-layer proxy** — `temporal_drift_score` counts stale value appearances in retrieved context, which overestimates actual stale-answer rates (an LLM given both old and new values often resolves to the correct one). Clarified as a worst-case upper bound. `llm_temporal_drift` (LLM mode) is the behavioral measurement. Both the README metric table and `docs/why-memory-evaluation-matters.md` updated accordingly.

### Added — Research-Grade Fixes (`feat/research-grade-fixes`)

**Fix 1 — Multi-seed statistical validation**
- `simulator/personas.py`: 5 demographically diverse personas (Arjun Sharma, Sofia Reyes, Wei Zhang, Amara Osei, Lars Eriksson) for cross-population result validation
- `evaluation/stats.py`: `aggregate_metric()` + `aggregate_checkpoint_series()` with mean, std, and 95% confidence intervals (t-distribution)
- `evaluation/benchmark.py`: `run_benchmark_multi_seed()` — runs N personas and returns aggregated stats
- CLI: `--seeds N` flag reports `mean ± std` across N personas instead of single-run results

**Fix 2 — Scientifically-grounded decay formula with ablation**
- `memory/decay.py`: 4 pluggable temporal decay functions with academic references:
  - `ebbinghaus` (default) — Ebbinghaus (1885) forgetting curve: `e^{-t/sqrt(1+t)}`
  - `exponential` — Jost (1897): `e^{-k*t/window}`
  - `linear` — Wickelgren (1972) baseline: `1 - t/window`
  - `default` — original heuristic preserved for backwards compat
- `CascadingTemporalMemory` now accepts a `decay` parameter; defaults to `ebbinghaus`
- CLI: `--decay ebbinghaus|exponential|linear|default`
- Ablation result: Ebbinghaus achieves 5.67× cascade efficiency vs 5.45× for original heuristic

**Fix 3 — Bounded Chunked RAG (realistic production simulation)**
- `memory/rag_chunked.py`: `ChunkedRAGMemory` with overlapping 120-char chunks and FIFO eviction at `max_chunks=200`
  - Models real production RAG: chunk splitting reduces retrieval certainty; bounded index causes early-fact eviction
  - Shows realistic 85–87% recall at T=100 vs ideal RAG's 100% — contrast is the key finding
- Registered as `rag_chunked` backend in benchmark runner and CLI

**Fix 4 — Research paper**
- `paper/memorylens_paper.md`: 6-section academic paper with proper citations (Ebbinghaus 1885, MemGPT, RAGAS, Jost 1897, Atkinson & Shiffrin 1968), ablation tables, multi-seed results tables, and related work comparison against RAGAS, TruLens, DeepEval, MemGPT, A-MEM

**Tests**: 10 new tests covering decay functions, ChunkedRAGMemory, stats aggregation, and persona pool structure (24 total, all passing)

---

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
