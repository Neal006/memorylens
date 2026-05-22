# Contributing to MemoryLens

Thank you for contributing to the **open-source benchmark for LLM memory decay**. Every contribution — a bug report, a new memory backend, a documentation fix — makes the benchmark better for the entire AI/ML community.

---

## Table of Contents

- [Quick orientation](#quick-orientation)
- [Development setup](#development-setup)
- [How to add a new memory backend](#how-to-add-a-new-memory-backend)
- [How to add a new metric](#how-to-add-a-new-metric)
- [How to add a new persona / scenario](#how-to-add-a-new-persona--scenario)
- [Running tests](#running-tests)
- [Submitting a PR](#submitting-a-pr)
- [Good first issues](#good-first-issues)
- [Style guide](#style-guide)

---

## Quick orientation

MemoryLens benchmarks **LLM memory decay** — how AI memory systems forget personal facts over long conversations. It has three layers:

```
Simulator  →  Memory Backend  →  Evaluator  →  Dashboard
(generate     (store + retrieve   (5 metrics,    (visualise
 conversation  context)           dual mode)     results)
```

Each layer is independently extensible. You can add a backend without touching the evaluator, and add a metric without touching the dashboard.

**Current backends:** `naive` · `rag` · `rag_chunked` · `cascading` · `summary`  
**Current metrics:** Recall@T · Precision@K · Temporal Drift · Memory Noise Ratio · Cascade Efficiency  
**LLM eval providers:** Groq · OpenAI · Anthropic · OpenRouter · Ollama

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
python tests/test_pipeline.py

# 5. Optional: run multi-seed benchmark
python main.py --seeds 5
```

Set `TRANSFORMERS_NO_TF=1` and `USE_TF=0` if you have TensorFlow installed.

---

## How to add a new memory backend

The most impactful contribution type. Full guide with a worked EntityMemory example: [docs/adding-a-new-backend.md](docs/adding-a-new-backend.md)

**Quick version — 4 steps:**

**Step 1 — Create `memory/your_backend.py`:**

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

**Step 2 — Register in `evaluation/benchmark.py`:**

```python
from memory.your_backend import YourMemory

def _make_memory(name: str, decay: str = "ebbinghaus") -> BaseMemory:
    if name == "your_backend":
        return YourMemory()
    # ... existing cases ...
```

Add `"your_backend"` to `VALID_BACKENDS`.

**Step 3 — Add one test in `tests/test_pipeline.py`:**

```python
def test_your_backend_recall_early():
    from memory.your_backend import YourMemory
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

All metrics live in `evaluation/metrics.py`. Each is a plain function — no classes.

```python
def your_metric(memory: BaseMemory, facts: List[Fact], current_turn: int) -> float:
    """
    Your Metric — one sentence description.
    Returns float in [0, 1] (or unbounded if it's a ratio like Cascade Efficiency).
    Must work without any API key — use content-based checks only.
    """
    return score
```

Wire it into the `CheckpointResult` dataclass in `evaluation/benchmark.py` and add a chart in `dashboard.py`.

---

## How to add a new persona / scenario

MemoryLens ships with 5 personas for multi-seed validation (`simulator/personas.py`). Adding more personas or domain scenarios (medical, customer support, education) strengthens the benchmark.

**Add a persona:**

```python
# simulator/personas.py — add to PERSONA_POOL
[
    Fact("name",       "Yuki Tanaka",     injected_at=0),
    Fact("city",       "Tokyo",           injected_at=1, updated_at=40, updated_value="Osaka"),
    Fact("occupation", "nurse",           injected_at=2),
    # ... 8 facts total, matching the keys in BENCHMARK_FACTS ...
]
```

**Add a domain scenario** (e.g., medical): create `simulator/medical_facts.py` with a `MEDICAL_FACTS` list and a matching `generate_medical_conversation()` function. Then run:

```bash
python main.py --backends naive rag cascading   # with your scenario wired in
```

---

## Running tests

```bash
# All 24 integration tests (no API key needed)
python tests/test_pipeline.py

# Import smoke test
python tests/test_imports.py

# Quick demo with real benchmark numbers
python main.py

# Multi-seed with confidence intervals
python main.py --seeds 5
```

CI runs both test files on Python 3.10 and 3.11 on every push.

---

## Submitting a PR

1. **Fork** the repo and create a branch: `git checkout -b feat/your-feature`
2. Make your changes with tests
3. Run `python tests/test_pipeline.py` — all 24 tests must pass
4. Open a PR against `main` — fill in the PR template
5. A maintainer will review within 48 hours

**PR checklist:**
- [ ] All 24 tests pass locally
- [ ] New backend/metric has at least one test
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`
- [ ] `VALID_BACKENDS` updated in `evaluation/benchmark.py` if adding a backend

---

## Good first issues

Open, well-scoped tasks — each has clear acceptance criteria:

| Task | Difficulty | Where | Label |
|------|-----------|-------|-------|
| Update-aware Cascading — patch Cold tier summaries when a fact updates | Medium | `memory/cascading.py` | `good first issue` |
| Add confidence interval error bars to decay charts | Easy | `dashboard.py` | `good first issue` |
| EdTech scenario — student/teacher memory (subject performance, weak topics) | Easy | `simulator/` | `good first issue` |
| `pip install memorylens` — complete `pyproject.toml` setup and PyPI publish | Easy | root | `good first issue` |
| Qdrant vector DB backend (replaces NumPy cosine) | Medium | `memory/` | `enhancement` |
| EntityMemory backend — named-entity extraction into key-value store | Medium | `memory/` | `new-backend` |
| Medical scenario — patient history across multi-session conversations | Medium | `simulator/` | `enhancement` |
| LangGraph orchestration wrapper | Hard | new | `enhancement` |
| arXiv preprint from `paper/memorylens_paper.md` | Medium | `paper/` | `research` |

Browse all: [github.com/Neal006/memorylens/issues](https://github.com/Neal006/memorylens/issues)

---

## Style guide

- **Python**: PEP 8, 100-char line limit
- **Docstrings**: one-line summary + describe the return value
- **Type hints**: all public function signatures must be typed
- **Commit messages**: `type: short description` — type ∈ `feat / fix / docs / test / refactor`
- **Core metrics must be deterministic**: `evaluation/metrics.py` functions must work without any API key

---

Questions? Open a [Discussion](https://github.com/Neal006/memorylens/discussions) or comment on any issue.
