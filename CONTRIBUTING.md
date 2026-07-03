# Contributing to MemoryLens

MemoryLens is an open benchmark for measuring how AI memory systems forget over long conversations. Every contribution — a new backend, a benchmark scenario, a test, a documentation fix — directly improves the quality of LLM memory research for everyone.

---

## Table of Contents

- [Finding something to work on](#finding-something-to-work-on)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [How to add a new memory backend](#how-to-add-a-new-memory-backend)
- [How to add a new metric](#how-to-add-a-new-metric)
- [How to add a new domain scenario](#how-to-add-a-new-domain-scenario)
- [Running tests](#running-tests)
- [Commit style](#commit-style)
- [Submitting a PR](#submitting-a-pr)
- [Code standards](#code-standards)

---

## Finding something to work on

| I am... | Start here |
|---------|-----------|
| Brand new to open source | [`good first issue`](https://github.com/Neal006/memorylens/issues?q=is%3Aopen+label%3A%22good+first+issue%22) — labeled, scoped, with step-by-step hints |
| Comfortable with Python | [`difficulty: beginner`](https://github.com/Neal006/memorylens/issues?q=is%3Aopen+label%3A%22difficulty%3A+beginner%22) or [`difficulty: intermediate`](https://github.com/Neal006/memorylens/issues?q=is%3Aopen+label%3A%22difficulty%3A+intermediate%22) |
| Experienced ML engineer | [`difficulty: advanced`](https://github.com/Neal006/memorylens/issues?q=is%3Aopen+label%3A%22difficulty%3A+advanced%22) or [`help wanted`](https://github.com/Neal006/memorylens/issues?q=is%3Aopen+label%3A%22help+wanted%22) |
| A technical writer | [`area: documentation`](https://github.com/Neal006/memorylens/issues?q=is%3Aopen+label%3A%22area%3A+documentation%22) — Markdown only, no code needed |

**Before starting:** Comment on the issue to let others know you're working on it. This prevents duplicate effort.

---

## Quick start

```bash
# 1. Fork the repo on GitHub, then clone your fork:
git clone https://github.com/<your-username>/memorylens
cd memorylens

# 2. Create a virtual environment (recommended):
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install in editable mode with dev + server dependencies:
pip install -e ".[server,dev]"

# 4. Copy the environment file (API key is optional):
cp .env.example .env
# Open .env and add your GROQ_API_KEY if you have one.
# The zero-API-key path works fully without any key.

# 5. Run the full test suite — all tests should pass before you start:
pytest tests/ -v

# 6. Try the zero-API-key demo to confirm everything works:
python quick_demo.py
```

---

## Quick orientation

MemoryLens benchmarks **LLM memory decay** — how AI memory systems forget personal facts over long conversations. It has three layers:

```
Simulator  →  Memory Backend  →  Evaluator  →  Dashboard
(generate     (store + retrieve   (5 metrics,    (visualise
 conversation  context)           dual mode)     results)
```

Each layer is independently extensible. You can add a backend without touching the evaluator, and add a metric without touching the dashboard.

**Current backends:** `naive` · `rag` · `rag_chunked` · `cascading` · `summary` · `entity` · `graph` · `faiss`  
**Current metrics:** Recall@T · Precision@K · Temporal Drift · Contradiction · Memory Noise Ratio · Cascade Efficiency  
**Current scenarios:** `default` · `edtech` · `support` · `medical`  
**LLM eval providers:** Groq · OpenAI · Anthropic · OpenRouter · Ollama

Set `TRANSFORMERS_NO_TF=1` and `USE_TF=0` if you have TensorFlow installed alongside PyTorch.

---

## Project layout

```
memorylens/                  The installable package (pip install memorylens)
├── memory/                  Memory backend implementations — add new backends here
│   ├── base.py              Abstract BaseMemory interface (3 methods every backend must implement)
│   ├── naive.py             Naive full-history backend (simplest example)
│   ├── rag.py               Semantic retrieval backend (sentence-transformers)
│   ├── rag_chunked.py       Chunked + bounded-index RAG (production-realistic)
│   ├── cascading.py         Three-tier hot/warm/cold with Ebbinghaus temporal decay
│   ├── summary.py           Rolling-summary compression backend
│   ├── entity.py            Structured key-value entity extraction (great reference for new backends)
│   ├── graph.py             NetworkX knowledge-graph backend
│   ├── vector_faiss.py      FAISS vector index backend (optional dep: memorylens[faiss])
│   └── decay.py             Temporal decay functions (ebbinghaus, exponential, linear)
├── evaluation/
│   ├── metrics.py           All benchmark metrics — add new metrics here
│   ├── benchmark.py         Benchmark orchestrator — registers backends, runs eval loop
│   ├── stats.py             Multi-seed aggregation and forgetting-curve fitting
│   └── logger.py            Experiment logging (JSON + CSV)
├── simulator/
│   ├── facts.py             Fact dataclass and BENCHMARK_FACTS
│   ├── conversation.py      generate_conversation() — builds the simulated chat
│   ├── personas.py          5 diverse personas for multi-seed runs
│   └── scenarios/           Scenario registry (default, edtech, support, medical)
├── utils/
│   ├── embeddings.py        Local sentence-transformer embeddings (no API key needed)
│   └── providers.py         LLM provider abstraction (Groq, OpenAI, Anthropic, Ollama…)
├── api.py                   FastAPI REST server (optional dep: memorylens[server])
└── cli.py                   CLI entry point (`memorylens` command)

tests/                       Integration tests — run with: pytest tests/ -v
dashboard.py                 Streamlit visualisation dashboard
main.py                      Backward-compatible wrapper around memorylens.cli
quick_demo.py                Zero-API-key demo
docs/                        Guides and comparison docs
```

---

## How to add a new memory backend

The most impactful contribution type. Full guide with a worked EntityMemory example: [docs/adding-a-new-backend.md](docs/adding-a-new-backend.md)

**Quick version — 4 steps:**

**Step 1 — Create `memorylens/memory/your_backend.py`:**

```python
from typing import List, Dict
from .base import BaseMemory

class YourMemory(BaseMemory):
    name = "your_backend"   # used in --backends flag

    def __init__(self):
        pass  # initialise your data structures

    def add_message(self, role: str, content: str, turn: int) -> None:
        pass  # store the message

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        # return [{"role": "user", "content": "..."}, ...]
        # this list is what the evaluator measures
        pass

    def reset(self) -> None:
        pass  # clear all state
```

**Step 2 — Register in `memorylens/evaluation/benchmark.py`:**

```python
from memorylens.memory.your_backend import YourMemory

def _make_memory(name: str, decay: str = "ebbinghaus") -> BaseMemory:
    if name == "your_backend":
        return YourMemory()
    # ... existing cases ...
```

Add `"your_backend"` to `VALID_BACKENDS`.

**Step 3 — Add one test in `tests/test_pipeline.py`:**

```python
def test_your_backend_recall_early():
    from memorylens.memory.your_backend import YourMemory
    mem = YourMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.5  # adjust threshold to your backend's expected performance
    print(f"PASS: your_backend recall early ({rate:.0%})")
```

**Step 4 — Run the full benchmark:**

```bash
python main.py --backends your_backend naive rag cascading --output my_results.json
```

Open a PR with the three files changed. A maintainer will review within 48 hours.

---

## How to add a new metric

All metrics live in `memorylens/evaluation/metrics.py`. Each is a plain function — no classes.

```python
def your_metric(memory: BaseMemory, facts: List[Fact], current_turn: int) -> float:
    """
    Your Metric — one sentence description.
    Returns float in [0, 1] (or unbounded if it's a ratio like Cascade Efficiency).
    Must work without any API key — use content-based checks only.
    """
    return score
```

Wire it into the `CheckpointResult` dataclass in `memorylens/evaluation/benchmark.py` and add a chart in `dashboard.py`. `contradiction_score` is the most recent worked example — trace it through metrics.py → benchmark.py → dashboard.py.

---

## How to add a new domain scenario

Copy `memorylens/simulator/scenarios/medical.py` as a starting template:

```python
# memorylens/simulator/scenarios/your_scenario.py
from memorylens.simulator.facts import Fact
from memorylens.simulator.scenarios.base import Scenario

YOUR_PERSONA_POOL = [
    [
        Fact("name",  "Alice Chen",  injected_at=0),
        Fact("role",  "developer",   injected_at=2),
        Fact("city",  "Singapore",   injected_at=4, updated_at=40, updated_value="Sydney"),
        # 8 facts total; at least 2 should have updated_at set.
        # Old and new values must not be substrings of each other.
    ],
    # ... 2+ more personas with the same fact keys
]

YOUR_FILLER_TURNS = ["Can you help me with...", ...]  # 20 domain questions

YOUR_SCENARIO = Scenario(
    name="your_scenario",
    description="One-line description shown by --list-scenarios.",
    persona_pool=YOUR_PERSONA_POOL,
    filler_turns=YOUR_FILLER_TURNS,
)
```

Then register it in the `SCENARIOS` dict in `memorylens/simulator/scenarios/__init__.py` — the CLI (`--scenario your_scenario`), the API, and the tests pick it up automatically.

---

## Running tests

```bash
# Full test suite (no API key needed):
pytest tests/ -v

# Run only tests matching a keyword:
pytest tests/ -v -k "cascading"
pytest tests/ -v -k "recall or drift"

# Quick demo (confirms zero-API-key path works):
python quick_demo.py

# Full benchmark run:
python main.py --backends naive rag cascading --turns 50

# Multi-seed with confidence intervals:
python main.py --seeds 5
```

CI runs the suite on Python 3.10–3.13 (Linux) plus Windows and macOS on every push, and builds + validates the PyPI package. All tests must pass without an API key.

---

## Commit style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add GraphMemory backend using NetworkX
fix: patch stale cold-tier summaries on fact update
docs: add customer_support scenario to README
test: add contradiction_score integration tests
chore: add networkx to requirements.txt
refactor: extract _extract_entity() helper from EntityMemory
```

---

## Submitting a PR

1. Fork the repo and create a branch: `git checkout -b feat/your-feature`
2. Make your changes with tests
3. Run `pytest tests/ -v` — all tests must pass
4. Open a PR against `main` — fill in the PR template
5. Reference the issue: `Closes #<issue-number>` in the PR description

**PR checklist:**
- [ ] All existing tests pass: `pytest tests/ -v`
- [ ] New tests added for new functionality
- [ ] Docstrings added on all new public functions/classes
- [ ] Type hints used on all new function signatures
- [ ] If adding a backend: registered in `VALID_BACKENDS` and `_make_memory()`
- [ ] If adding a scenario: registered in the `SCENARIOS` dict in `memorylens/simulator/scenarios/__init__.py`
- [ ] README updated if new CLI flags or user-facing features were added
- [ ] No API key required to run any new tests

---

## Code standards

| Rule | Why |
|------|-----|
| **Type hints** on all public functions | Enables IDE autocomplete, catches bugs early |
| **Docstrings** on all public classes and functions | Helps contributors understand intent without reading implementation |
| **No new top-level dependencies** without issue discussion | Keeps install size predictable |
| **All new metrics return `float` in `[0, 1]`** | Ensures dashboard and aggregation code work without guards |
| **All tests pass without an API key** | Keeps CI fast and accessible to all contributors |
| **PEP 8**, 100-char line limit | Consistency |

---

## Getting help

- **Stuck on an issue?** Comment on it — maintainers respond promptly
- **General questions?** Open a [Discussion](https://github.com/Neal006/memorylens/discussions)
- **Best reference for new backends:** `memorylens/memory/entity.py` — the shortest, cleanest example
- **CI failing?** Run `pytest tests/ -v` locally first; the error message is usually self-explanatory

Welcome aboard — we're glad you're here!
