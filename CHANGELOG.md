# Changelog

All notable changes to MemoryLens are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.0] — 2026-07-04

### Added
- **PyPI-ready packaging** — published as **`memorylens-bench`** (the bare
  `memorylens` name was already taken on PyPI by an unrelated project); the
  import namespace and CLI remain `memorylens`. All code now lives under a single `memorylens` package
  (`memorylens.memory`, `memorylens.simulator`, `memorylens.evaluation`, `memorylens.utils`).
  Fixed an invalid `build-backend` that made the sdist unbuildable, added the
  `memorylens` console command (`memorylens.cli:main`), and split dependencies into a
  lean core plus `[dashboard]`, `[server]`, `[faiss]`, `[groq]`, `[openai]`,
  `[anthropic]`, `[all]`, `[dev]` extras. `python main.py` still works.
- **GraphMemory** (`graph`) — NetworkX knowledge-graph backend; fact updates replace
  edges in place so stale values cannot survive (#20)
- **FAISSMemory** (`faiss`) — FAISS `IndexFlatIP` vector backend as an optional extra (#14)
- **contradiction_score** — flags contexts that surface both the old and new value of an
  updated fact; wired into checkpoints, CSV logs, and the dashboard (#21)
- **Scenario framework** — `Scenario` dataclass + registry (#22) with three domain
  scenarios: `edtech` (#18), `support` (#23), `medical` (#24); `--scenario` and
  `--list-scenarios` CLI flags
- **FastAPI server** — `uvicorn memorylens.api:app`; job-based POST `/v1/benchmarks`,
  GET `/v1/backends`, `/v1/scenarios`, `/health` (#25)
- **Dashboard Run History tab** — overlay Recall@T curves from past `experiment_logs/`
  runs and compare final metrics side-by-side (#27)
- **SQLite persistent storage** (`memorylens/utils/storage.py`) — queryable database
  alongside the JSON/CSV logs; `log_run()` writes to it, `list_runs()` queries it
  first, `Storage.compare_runs()` compares recall across runs, and
  `python -m memorylens.utils.migrate_legacy_logs` imports legacy JSON logs
  (#26, contributed by @Sugaria0427)
- 23 new integration tests (45 total): GraphMemory, contradiction_score, scenarios,
  FAISS, API lifecycle, SQLite storage, and cascading regressions (#31)
- CI matrix expanded to Python 3.10–3.13 on Linux plus Windows and macOS, with a
  package build + `twine check` + wheel-install job (#30)

### Fixed
- **Cascading cold-tier recall regression** — the newest-first cold-summary merge
  introduced with the drift fix truncated away the oldest fact summaries, collapsing
  cascading recall at T=100 from ~75% to ~8%. Merging is oldest-first again (stale
  values are already rewritten in place by the update patcher) and empty "No key
  facts." summaries are no longer appended to the cold tier. Regression tests added.
- Experiment CSV logger crashed on every run since the `has_llm_eval` flag was added
  (a bool was indexed as a dict); it now skips non-backend keys and rotates the CSV
  when the metric schema changes
- Benchmark results in the README were stale and did not reproduce; all tables are
  regenerated from the current code

### Removed
- Dead references to an unpublished research paper (`paper/memorylens_paper.md` never
  existed in the repository); unverifiable "the only framework" marketing claims;
  fabricated ₹-cost projections in the dashboard (replaced with a clearly labelled
  illustrative $-projection)

---

## [0.3.0] — 2026-05-24

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

**Fix 4 — Research paper** *(never merged into the repository; stale references to it were removed in 0.4.0)*

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
