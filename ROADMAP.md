# MemoryLens Roadmap

MemoryLens is the **open-source benchmark for LLM memory decay** — the only evaluation framework that measures how AI memory architectures forget over long conversations. This document tracks what's shipped, what's in progress, and what's next.

Want to pick something up? Check [CONTRIBUTING.md](CONTRIBUTING.md) and claim an item by opening an issue.

---

## Shipped — v0.3 (current)

### Core Benchmark
- [x] Five memory backends: Naive · Ideal RAG · Chunked RAG · Cascading Temporal · SummaryMemory
- [x] Five evaluation metrics: Recall@T · Precision@K · Temporal Drift · Memory Noise Ratio · Cascade Efficiency
- [x] Multi-seed statistical validation — 5 diverse personas, mean ± std, 95% CI
- [x] Ebbinghaus forgetting curve decay + ablation (linear / exponential / Ebbinghaus / heuristic)
- [x] Bounded Chunked RAG — realistic production simulation with FIFO eviction

### LLM Evaluation
- [x] Two-stage LLM answer+judge pipeline (real recall, not string match)
- [x] 5-provider LLM backend: Groq · OpenAI · Anthropic · OpenRouter · Ollama
- [x] Gap analysis: content recall vs LLM recall (content can overestimate by 5–15pp)

### Tooling
- [x] CLI with `--seeds`, `--decay`, `--llm`, `--provider`, `--list-providers`
- [x] Streamlit dashboard with Content/LLM/Gap tabbed charts, provider selector
- [x] Zero-API-key mode — all content-based metrics work without any key
- [x] Experiment logger (JSON + CSV) in `experiment_logs/`
- [x] GitHub Actions CI: Python 3.10 + 3.11, 24 tests on every push

### Documentation
- [x] Research paper: [paper/memorylens_paper.md](paper/memorylens_paper.md)
- [x] CITATION.cff — citable software with Zenodo integration
- [x] `docs/why-memory-evaluation-matters.md`
- [x] `docs/comparison-with-existing-tools.md`
- [x] `docs/adding-a-new-backend.md`

---

## Next — v0.4 (open for contributions)

### High Priority Fixes
- [ ] **Update-aware Cascading** — when a fact update event fires, patch existing Cold tier summaries to reflect the new value. This eliminates the temporal drift regression where cold summaries retain stale facts. ([open issue](https://github.com/Neal006/memorylens/issues))
- [ ] **Confidence interval charts** — add ± std error bars to all decay curves in the Streamlit dashboard when multi-seed results are loaded
- [ ] **Varied persona injection timing** — currently all 5 personas share identical fact injection timing (T=0,1,2,3,4,5,7,9), causing structure-deterministic backends (Naive, Cascading) to produce std=0%. Fix: randomise injection turns ±3 turns per persona per seed so timing-based backends show real variance across seeds. This is a known limitation documented in the README.

### New Memory Backends
- [ ] **EntityMemory** — extract named entities into a structured key-value store; benchmark whether structured storage beats unstructured retrieval ([guide](docs/adding-a-new-backend.md))
- [ ] **Qdrant backend** — production vector DB replacing NumPy cosine similarity; benchmark at 10K+ conversation turns
- [ ] **Graph memory** — entities + relationships stored as a knowledge graph; test multi-hop fact retrieval
- [ ] **Redis-backed memory** — persistent cross-session memory; test recall across session boundaries

### Scenarios
- [ ] **EdTech scenario** — student/teacher memory: track subject performance, weak topics, learning styles across 200-turn tutoring session
- [ ] **Customer support scenario** — 100K customer histories; benchmark memory under high cardinality
- [ ] **Medical scenario** — patient history across multi-session clinical conversations (anonymised synthetic data)

### Integrations
- [ ] **LangGraph wrapper** — run the full benchmark as a LangGraph state machine for agent-native evaluation
- [ ] **RAGAS adapter** — export MemoryLens checkpoints as RAGAS-compatible evaluation samples
- [ ] **LangChain memory adapter** — wrap LangChain `ConversationSummaryMemory` and `VectorStoreRetrieverMemory` as MemoryLens backends

---

## Later — v0.5

### Deployment
- [ ] **Streamlit Community Cloud** — live public demo URL (no install needed)
- [ ] **HuggingFace Spaces** — mirror for ML community discoverability
- [ ] **Docker image** — `docker run neal006/memorylens`
- [ ] **`pip install memorylens`** — proper PyPI package

### Research Track
- [ ] **arXiv preprint** — publish [paper/memorylens_paper.md](paper/memorylens_paper.md) as arXiv:XXXX.XXXXX
- [ ] **HuggingFace dataset** — synthetic conversation logs as a public dataset card
- [ ] **Ebbinghaus curve fitting** — fit the actual Recall@T decay data to the forgetting curve and report stability parameters per backend

### Engineering
- [ ] **Async benchmark runner** — parallel backend evaluation for 5× faster multi-seed runs
- [ ] **Plugin architecture** — register custom backends and metrics via Python entry points
- [ ] **Streaming evaluation** — real-time memory quality monitoring for live LLM deployments

---

## How to Contribute to the Roadmap

1. **Pick any unchecked item** above
2. Open an issue to claim it (prevents duplicate work)
3. Reference the roadmap item in your PR

Have an idea not on this list? Open a [Feature Request](https://github.com/Neal006/memorylens/issues/new?template=feature_request.md).
