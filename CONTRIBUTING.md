# Contributing to MemoryLens

First off — thank you for taking the time to contribute. MemoryLens is an open research tool and every contribution, from a typo fix to a new memory backend, matters.

---

## Table of Contents

- [Quick orientation](#quick-orientation)
- [Development setup](#development-setup)
- [Project structure explained](#project-structure-explained)
- [How to add a new memory backend](#how-to-add-a-new-memory-backend)
- [How to add a new metric](#how-to-add-a-new-metric)
- [Running tests](#running-tests)
- [Submitting a PR](#submitting-a-pr)
- [Good first issues](#good-first-issues)
- [Style guide](#style-guide)

---

## Quick orientation

MemoryLens has three moving parts:

```
Simulator  →  Memory Backend  →  Evaluator  →  Dashboard
(generate     (store + retrieve   (measure       (visualise
 fake conv.)   context)            decay)         results)
```

Each layer is independently extensible. You can add a backend without touching the evaluator, and add a metric without touching the dashboard.

---

## Development setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR-USERNAME/memorylens.git
cd memorylens

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify everything works (no API key needed)
python quick_demo.py

# 5. Run the test suite
python tests/test_pipeline.py
```

Set `TRANSFORMERS_NO_TF=1` and `USE_TF=0` in your environment if you have TensorFlow installed — this prevents a protobuf conflict.

---

## Project structure explained

```
memorylens/
│
├── simulator/               # Synthetic conversation engine
│   ├── facts.py             # Fact definitions + BENCHMARK_FACTS list
│   └── conversation.py      # Generates turn-by-turn conversation events
│
├── memory/                  # Memory backend implementations
│   ├── base.py              # Abstract base class — every backend implements this
│   ├── naive.py             # Naive: full history, truncate oldest on overflow
│   ├── rag.py               # RAG: embed all messages, retrieve top-K by cosine sim
│   └── cascading.py         # Cascading Temporal: hot/warm/cold three-tier memory
│
├── evaluation/              # Metrics and orchestration
│   ├── metrics.py           # Content-based metric functions (no LLM needed)
│   ├── benchmark.py         # Benchmark runner — wires simulator + memory + metrics
│   ├── llm_judge.py         # Optional: Groq-powered answer quality judge
│   └── logger.py            # Experiment logger → JSON + CSV
│
├── utils/
│   ├── embeddings.py        # sentence-transformers wrapper (cached model load)
│   └── llm.py               # Groq API wrapper with retry logic
│
├── tests/
│   ├── test_imports.py      # CI smoke test: all imports resolve
│   └── test_pipeline.py     # 8 integration tests (no API key needed)
│
├── dashboard.py             # Streamlit visualisation layer
├── main.py                  # CLI entry point
├── quick_demo.py            # Zero-API-key demo script
└── demo_results.json        # Pre-computed results for instant dashboard demo
```

---

## How to add a new memory backend

This is the most impactful type of contribution. The interface is simple — 4 methods.

**Step 1 — Create `memory/your_backend.py`:**

```python
from typing import List, Dict
from .base import BaseMemory

class YourMemory(BaseMemory):
    name = "your_backend"   # used in CLI --backends flag

    def __init__(self):
        # initialise your data structures
        pass

    def add_message(self, role: str, content: str, turn: int) -> None:
        # store a new message
        pass

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        # return a list of {"role": ..., "content": ...} dicts
        # these are what get measured by the evaluator
        pass

    def reset(self) -> None:
        # clear all stored state
        pass
```

**Step 2 — Register in `evaluation/benchmark.py`:**

```python
def _make_memory(name: str) -> BaseMemory:
    if name == "naive":      return NaiveMemory(...)
    if name == "rag":        return RAGMemory()
    if name == "cascading":  return CascadingTemporalMemory()
    if name == "your_backend": return YourMemory()   # add this line
```

**Step 3 — Add a test in `tests/test_pipeline.py`:**

```python
def test_your_backend_recall_early():
    mem = YourMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.75
    print(f"PASS: your_backend recall early ({rate:.0%})")
```

**Step 4 — Run the full benchmark against your backend:**

```bash
python main.py --backends your_backend naive rag --output my_results.json
```

That's it. Open a PR with the three files changed.

---

## How to add a new metric

All metrics live in `evaluation/metrics.py`. Each metric is a plain function — no classes, no magic.

```python
def your_metric(memory: BaseMemory, facts: List[Fact], current_turn: int) -> float:
    """
    Your Metric — one sentence description.
    Returns a float in [0, 1] (or unbounded if it's a ratio).
    """
    # implement here
    return score
```

Then wire it into the benchmark runner in `evaluation/benchmark.py` at the checkpoint evaluation block, and add a chart for it in `dashboard.py`.

---

## Running tests

```bash
# All tests (no API key needed)
python tests/test_pipeline.py

# Import smoke test only
python tests/test_imports.py

# Full demo with real numbers
python quick_demo.py
```

CI runs both test files on Python 3.10 and 3.11 on every push.

---

## Submitting a PR

1. **Fork** the repo and create a branch: `git checkout -b feat/your-feature`
2. Make your changes with tests
3. Run `python tests/test_pipeline.py` — all 8 tests must pass
4. Open a PR against `main` — fill in the PR template
5. A maintainer will review within 48 hours

**PR checklist:**
- [ ] Tests pass locally
- [ ] New feature has at least one test
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`
- [ ] Docstring added to new functions

---

## Good first issues

If you're new to the project, these are well-scoped starting points:

| Issue | Difficulty | Skills needed |
|-------|-----------|---------------|
| Add `SummaryMemory` backend — rolling LLM summary every K turns | Medium | Python, LLM APIs |
| Multi-seed benchmark — run N seeds, report mean ± std | Easy | Python, numpy |
| Update-aware Cascading — patch Cold tier on fact updates | Medium | Python, algorithmic |
| Add confidence interval error bars to dashboard charts | Easy | Plotly |
| Add `--output-format csv` flag to CLI | Easy | Python, argparse |
| Write a Docker deployment guide | Easy | Docker |
| EdTech fact scenario — student/teacher memory tracking | Easy | Python |
| Fit an Ebbinghaus forgetting curve to Recall@T data | Medium | scipy, numpy |

Browse all open issues: [github.com/Neal006/memorylens/issues](https://github.com/Neal006/memorylens/issues)

---

## Style guide

- **Python**: follow PEP 8, 100-char line limit
- **Docstrings**: one-line summary + explain what the return value represents
- **Type hints**: all public function signatures must be typed
- **Commit messages**: `type: short description` where type is one of `feat / fix / docs / test / refactor`
- **No LLM calls in core metrics** — all `evaluation/metrics.py` functions must be deterministic and work without an API key

---

Questions? Open a [Discussion](https://github.com/Neal006/memorylens/discussions) or drop a comment on any issue.
