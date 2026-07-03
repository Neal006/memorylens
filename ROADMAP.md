# MemoryLens Roadmap

MemoryLens is an open-source benchmark for **LLM memory decay** — measuring how AI memory architectures forget over long conversations. This document tracks what's shipped, what's in progress, and what's next.

Want to pick something up? Check [CONTRIBUTING.md](CONTRIBUTING.md) and claim an item by opening an issue.

---

## Shipped — v0.4 (current)

### Packaging & Distribution
- [x] **`pip install memorylens`** — proper PyPI package: single `memorylens` namespace, valid build backend, `memorylens` CLI entry point, lean core deps with `[dashboard]` / `[server]` / `[faiss]` / provider extras
- [x] CI matrix: Python 3.10–3.13 on Linux + Windows + macOS, plus a package build/validation job

### New Backends
- [x] **EntityMemory** — structured key-value fact extraction
- [x] **GraphMemory** — NetworkX knowledge graph with in-place fact updates
- [x] **FAISSMemory** — FAISS vector index (optional dependency)

### Scenarios
- [x] **Scenario framework** — `Scenario` dataclass + registry; scenarios plug into CLI, API, and multi-seed runs
- [x] **EdTech** (student-tutor), **Customer Support** (ticket lifecycle), **Medical** (synthetic patient consultations)

### Metrics & Fixes
- [x] **contradiction_score** — detects when context surfaces both old and new values of an updated fact
- [x] **Cold-tier recall regression fixed** — a cold-summary merge-order bug silently destroyed early facts (cascading recall at T=100 fell to ~8%); root-caused and fixed with regression tests
- [x] Ebbinghaus + exponential forgetting-curve fitting (`--fit-curves`)

### Tooling
- [x] **FastAPI server** — async job-based REST API (`uvicorn memorylens.api:app`)
- [x] **Dashboard Run History tab** — compare past benchmark runs side-by-side
- [x] Experiment logger schema migration (rotates CSV on metric changes)

---

## Shipped — v0.3

- [x] Multi-seed statistical validation — 5 personas, mean ± std, 95% CI (`--seeds`)
- [x] Ebbinghaus decay + ablation (linear / exponential / Ebbinghaus / heuristic)
- [x] Bounded Chunked RAG (chunking + FIFO index eviction)
- [x] Two-stage LLM answer+judge pipeline; 5 providers (Groq · OpenAI · Anthropic · OpenRouter · Ollama)
- [x] SummaryMemory backend, experiment logger, zero-API-key mode

---

## Next — v0.5 (open for contributions)

### Benchmark Realism
- [ ] **Harder fact templates** — free-form paraphrased injections so extraction-based backends (entity, graph) can't pattern-match; today's templated facts let them score 100% by construction
- [ ] **Varied persona injection timing** — randomise injection turns per seed so timing-deterministic backends show real cross-seed variance
- [ ] **Longer horizons** — 500–1,000-turn runs where bounded indexes and context budgets genuinely saturate

### New Backends
- [ ] **Qdrant backend** — hosted vector DB, benchmark at 10K+ turns
- [ ] **Redis-backed memory** — persistence across session boundaries
- [ ] **LangChain memory adapter** — wrap `ConversationSummaryMemory` / `VectorStoreRetrieverMemory` as backends

### Integrations & Deployment
- [ ] **Publish to PyPI** (package is release-ready; needs a maintainer `twine upload`)
- [ ] **Docker image**
- [ ] **HuggingFace Spaces / Streamlit Cloud demo** (community contribution welcome — see issue #28)
- [ ] **RAGAS adapter** — export checkpoints as RAGAS-compatible samples

### Engineering
- [ ] **Async benchmark runner** — parallel backend evaluation
- [ ] **Plugin architecture** — register custom backends/metrics via entry points

---

## How to Contribute to the Roadmap

1. **Pick any unchecked item** above
2. Open an issue to claim it (prevents duplicate work)
3. Reference the roadmap item in your PR

Have an idea not on this list? Open a [Feature Request](https://github.com/Neal006/memorylens/issues/new?template=feature_request.md).
