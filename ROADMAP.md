# MemoryLens Roadmap

This document tracks what's planned, what's in progress, and what's been shipped.
Want to pick something up? Check the [Contributing guide](CONTRIBUTING.md) and claim an item by opening an issue.

---

## Now — v0.2 (current)

- [x] Three memory backends: Naive, RAG, Cascading Temporal
- [x] Five metrics: Recall@T, Precision@K, Temporal Drift, Memory Noise Ratio, Cascade Efficiency
- [x] Streamlit dashboard with demo + live benchmark mode
- [x] CLI runner with JSON + LaTeX export
- [x] LLM-as-judge evaluator (Groq)
- [x] Experiment logger (JSON + CSV)
- [x] GitHub Actions CI (Python 3.10 + 3.11)
- [x] Zero-API-key quick demo

---

## Next — v0.3 (open for contributions)

### Critical Fixes
- [ ] **Update-aware Cascading** — patch Cold tier summaries when a fact is updated, eliminating temporal drift regression ([#good-first-issue])
- [ ] **Multi-seed benchmarking** — run N seeds, report mean ± std for all metrics ([#good-first-issue])
- [ ] **Confidence interval charts** — error bars on all decay curves in the dashboard

### New Memory Backends
- [ ] **SummaryMemory** — rolling LLM-generated summary that compresses at every K turns
- [ ] **EntityMemory** — LangChain-style named-entity extraction and storage
- [ ] **VectorDB backend (Qdrant)** — production-grade vector store replacing in-memory numpy
- [ ] **Redis-backed memory** — persistent memory across sessions

### New Metrics
- [ ] **Answer Quality Score** — full LLM-as-judge evaluation over all checkpoints
- [ ] **Memory Utilisation Rate** — what fraction of stored memory is ever retrieved
- [ ] **First-Recall Latency** — at which turn does a fact first become retrievable
- [ ] **Forgetting Curve fit** — fit an Ebbinghaus forgetting curve to Recall@T data

### Integrations
- [ ] **LangGraph orchestration** — wrap the pipeline in a LangGraph state machine
- [ ] **LlamaIndex memory node** — MemoryLens as a LlamaIndex evaluator
- [ ] **RAGAS integration** — plug into the RAGAS evaluation suite

---

## Later — v0.4

### Deployment
- [ ] **Streamlit Community Cloud deployment** — live public demo URL
- [ ] **HuggingFace Spaces mirror** — discoverability in the ML community
- [ ] **Docker image** — `docker run neal006/memorylens`

### Research Track
- [ ] **HuggingFace dataset card** — upload synthetic conversation logs as a public dataset
- [ ] **2-page research PDF** — LaTeX paper covering methodology, results, and findings
- [ ] **arXiv preprint** — publish the evaluation methodology

### Domain Scenarios
- [ ] **EdTech scenario** — student/teacher memory: track subject performance, weak topics, learning styles
- [ ] **Customer support scenario** — simulate a support agent remembering 100K customer histories
- [ ] **Medical scenario** — patient history memory across multi-session clinical conversations

### Engineering
- [ ] **`pip install memorylens`** — proper PyPI package via `pyproject.toml`
- [ ] **Async benchmark runner** — parallel backend evaluation for 10× faster runs
- [ ] **Plugin architecture** — register custom backends and metrics via entry points

---

## How to Contribute to the Roadmap

1. **Pick any unchecked item** above
2. Open an issue to claim it (prevents duplicate work)
3. Reference the roadmap item in your PR

Have an idea not on this list? Open a [Feature Request](https://github.com/Neal006/memorylens/issues/new?template=feature_request.md).
