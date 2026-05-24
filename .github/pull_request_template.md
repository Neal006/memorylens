## Summary

<!-- 1-3 sentences: what does this PR do and why? -->

## Related issue

Closes #<!-- Add the issue number this PR addresses -->

## Type of change

- [ ] Bug fix
- [ ] New memory backend (`memory/`)
- [ ] New evaluation metric (`evaluation/metrics.py`)
- [ ] New benchmark scenario (`simulator/scenarios/`)
- [ ] Dashboard / visualisation improvement (`dashboard.py`)
- [ ] API or infrastructure (`api/`, `utils/`)
- [ ] Documentation only
- [ ] Refactor / performance
- [ ] CI / tooling / dependencies

## How was this tested?

```bash
# 1. All existing tests still pass:
pytest tests/ -v

# 2. Commands specific to this change (copy from the issue's "Getting started" section):

```

## Checklist

- [ ] All existing tests pass: `pytest tests/ -v`
- [ ] New tests added for new functionality (at least 1 test per new class/function)
- [ ] Docstrings added on all new public classes and functions
- [ ] Type hints used on all new function signatures
- [ ] No API key required to run any new tests
- [ ] If adding a backend: registered in `VALID_BACKENDS` and `_make_memory()` in `benchmark.py`
- [ ] If adding a scenario: `--scenario` flag added to `main.py`
- [ ] If adding a CLI flag: docstring example in `main.py` updated
- [ ] README updated if there are new user-facing features or CLI flags
- [ ] No hardcoded API keys or secrets in any file

## Benchmark impact (if applicable)

<!-- If your change affects benchmark numbers, show before/after.
     Run: python main.py --backends naive rag cascading --turns 100 -->

| Metric | Before | After |
|--------|--------|-------|
| Recall@T=100 | | |
| Tokens/query@T=100 | | |
| Cascade Efficiency | | |

🤖 Generated with [Claude Code](https://claude.ai/claude-code)
