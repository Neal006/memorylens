"""
Integration tests for the MemoryLens pipeline.
No LLM calls — purely content-based metric evaluation.
"""

import os
import sys

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.facts import BENCHMARK_FACTS, Fact
from simulator.conversation import generate_conversation
from memory.naive import NaiveMemory
from memory.rag import RAGMemory
from memory.cascading import CascadingTemporalMemory
from memory.summary import SummaryMemory
from evaluation.metrics import (
    recall_at_t, temporal_drift_score, memory_noise_ratio, precision_at_k
)


def _populate(memory, facts, turns: int):
    events = generate_conversation(facts, turns)
    for ev in events:
        ack = "Got it." if ev["is_fact"] else "Sure."
        memory.add_message("user", ev["content"], ev["turn"])
        memory.add_message("assistant", ack, ev["turn"])
    return events


def test_conversation_generator():
    events = generate_conversation(BENCHMARK_FACTS, 20)
    assert len(events) == 20
    fact_events = [e for e in events if e["is_fact"]]
    assert len(fact_events) == len(BENCHMARK_FACTS)
    print("PASS: conversation generator")


def test_naive_recall_early():
    mem = NaiveMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate == 1.0, f"Expected 100% recall at T=15 for naive, got {rate:.0%}"
    print(f"PASS: naive recall early ({rate:.0%})")


def test_rag_recall_early():
    mem = RAGMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.75, f"Expected ≥75% recall at T=15 for RAG, got {rate:.0%}"
    print(f"PASS: RAG recall early ({rate:.0%})")


def test_cascading_recall_early():
    mem = CascadingTemporalMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.75, f"Expected ≥75% recall at T=15 for cascading, got {rate:.0%}"
    print(f"PASS: cascading recall early ({rate:.0%})")


def test_temporal_drift_not_applicable_before_update():
    mem = NaiveMemory()
    _populate(mem, BENCHMARK_FACTS, 10)
    city_fact = next(f for f in BENCHMARK_FACTS if f.key == "city")
    result = temporal_drift_score(mem, city_fact, 9)
    assert result["applicable"] is False
    print("PASS: temporal drift not applicable before update turn")


def test_temporal_drift_after_update():
    mem = NaiveMemory(max_context_tokens=8000)
    _populate(mem, BENCHMARK_FACTS, 50)
    city_fact = next(f for f in BENCHMARK_FACTS if f.key == "city")
    # city updated at T=40, checking at T=49
    result = temporal_drift_score(mem, city_fact, 49)
    assert result["applicable"] is True
    # Naive keeps all history so old AND new value are in context → drift could be 0 or 0.5
    assert 0.0 <= result["drift"] <= 1.0
    print(f"PASS: temporal drift after update (drift={result['drift']:.2f})")


def test_token_count_ordering():
    """RAG should use fewer tokens than naive at T=100."""
    naive = NaiveMemory(max_context_tokens=1200)
    rag = RAGMemory()
    _populate(naive, BENCHMARK_FACTS, 100)
    _populate(rag, BENCHMARK_FACTS, 100)

    name_fact = BENCHMARK_FACTS[0]
    naive_tokens = naive.token_count(name_fact.query_text(), 99)
    rag_tokens = rag.token_count(name_fact.query_text(), 99)

    assert rag_tokens < naive_tokens, (
        f"RAG ({rag_tokens} tokens) should be cheaper than naive ({naive_tokens} tokens)"
    )
    print(f"PASS: token ordering — naive={naive_tokens}, rag={rag_tokens}")


def test_noise_ratio_range():
    mem = RAGMemory()
    _populate(mem, BENCHMARK_FACTS, 30)
    known = [f.value for f in BENCHMARK_FACTS if f.injected_at < 30]
    noise = memory_noise_ratio(mem, "best sorting algorithm?", known, 29)
    assert 0.0 <= noise <= 1.0, f"Noise ratio out of range: {noise}"
    print(f"PASS: noise ratio in range ({noise:.2f})")


# ── SummaryMemory tests ────────────────────────────────────────────────────

def test_summary_extractive_fallback_recall_early():
    """SummaryMemory with extractive compression (no LLM) recalls facts at T=15."""
    mem = SummaryMemory(window_size=20, use_llm=False)
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.75, f"Expected >=75% recall at T=15 for summary, got {rate:.0%}"
    print(f"PASS: summary extractive recall early ({rate:.0%})")


def test_summary_compresses_overflow():
    """After enough messages, summary should be non-empty and recent buffer bounded."""
    mem = SummaryMemory(window_size=10, use_llm=False)
    _populate(mem, BENCHMARK_FACTS, 30)
    assert len(mem.recent) <= mem.window_size, (
        f"recent buffer {len(mem.recent)} exceeds window_size {mem.window_size}"
    )
    assert len(mem.summary) > 0, "summary should be non-empty after overflow"
    print(f"PASS: summary compression (recent={len(mem.recent)}, summary_len={len(mem.summary)})")


def test_summary_context_contains_summary_and_recent():
    """get_context() must return the summary block followed by recent messages."""
    mem = SummaryMemory(window_size=6, use_llm=False)
    _populate(mem, BENCHMARK_FACTS, 20)
    ctx = mem.get_context("What is my name?", 19)
    roles = [m["role"] for m in ctx]
    assert "system" in roles, "context should include a system summary block"
    assert "user" in roles,   "context should include recent user messages"
    print(f"PASS: summary context structure (chunks={len(ctx)}, roles={set(roles)})")


def test_summary_reset_clears_state():
    """reset() must clear both recent buffer and summary string."""
    mem = SummaryMemory(window_size=10, use_llm=False)
    _populate(mem, BENCHMARK_FACTS, 30)
    mem.reset()
    assert len(mem.recent) == 0,   "recent buffer should be empty after reset"
    assert mem.summary == "",       "summary should be empty string after reset"
    print("PASS: summary reset clears state")


def test_summary_token_cost_bounded():
    """SummaryMemory tokens/query should stay roughly constant after compression."""
    mem = SummaryMemory(window_size=20, use_llm=False)
    _populate(mem, BENCHMARK_FACTS, 100)
    name_fact = BENCHMARK_FACTS[0]
    tokens = mem.token_count(name_fact.query_text(), 99)
    # Should NOT grow linearly with history — bounded by window + summary
    assert tokens < 2000, f"token cost {tokens} seems unbounded (expected < 2000)"
    print(f"PASS: summary token cost bounded ({tokens} tokens at T=100)")


def test_summary_benchmark_registration():
    """'summary' backend must be resolvable from the benchmark runner."""
    from evaluation.benchmark import _make_memory
    mem = _make_memory("summary")
    assert mem.name == "summary"
    print(f"PASS: summary registered in benchmark runner ({mem!r})")


# ── Decay function tests ────────────────────────────────────────────────────

def test_decay_functions_range():
    """All decay functions must return values in [0, 1] for all valid inputs."""
    from memory.decay import _REGISTRY
    for name, fn in _REGISTRY.items():
        for age in [0, 1, 5, 10, 50, 99, 100]:
            v = fn(age, 100)
            assert 0.0 <= v <= 1.0, f"{name}({age}, 100) = {v} out of range"
    print("PASS: all decay functions in [0,1] range")


def test_ebbinghaus_is_monotone_decreasing():
    """Ebbinghaus decay must be monotonically non-increasing with age."""
    from memory.decay import decay_ebbinghaus
    prev = 1.0
    for age in range(0, 101):
        v = decay_ebbinghaus(age, 100)
        assert v <= prev + 1e-9, f"Ebbinghaus not monotone: age={age}, v={v}, prev={prev}"
        prev = v
    print("PASS: Ebbinghaus decay is monotone decreasing")


def test_cascading_uses_pluggable_decay():
    """CascadingTemporalMemory should accept and store the decay name."""
    from memory.cascading import CascadingTemporalMemory
    for name in ["default", "linear", "exponential", "ebbinghaus"]:
        mem = CascadingTemporalMemory(decay=name)
        assert mem.decay_name == name, f"Expected decay_name={name}, got {mem.decay_name}"
    print("PASS: CascadingTemporalMemory accepts all decay variants")


# ── ChunkedRAGMemory tests ──────────────────────────────────────────────────

def test_chunked_rag_recall_early():
    """ChunkedRAGMemory should recall facts with >= 75% accuracy at T=15."""
    from memory.rag_chunked import ChunkedRAGMemory
    mem = ChunkedRAGMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.75, f"Expected >=75% recall at T=15 for chunked RAG, got {rate:.0%}"
    print(f"PASS: ChunkedRAGMemory recall early ({rate:.0%})")


def test_chunked_rag_bounded_index():
    """ChunkedRAGMemory must not exceed max_chunks capacity."""
    from memory.rag_chunked import ChunkedRAGMemory
    mem = ChunkedRAGMemory(max_chunks=50)
    _populate(mem, BENCHMARK_FACTS, 100)
    assert len(mem.chunks) <= 50, (
        f"Chunk index {len(mem.chunks)} exceeds max_chunks=50"
    )
    assert len(mem.embeddings) == len(mem.chunks), (
        "Embedding count mismatch with chunk count"
    )
    print(f"PASS: ChunkedRAGMemory bounded index (chunks={len(mem.chunks)})")


def test_chunked_rag_tokens_less_than_naive():
    """ChunkedRAGMemory should use fewer tokens than naive at T=100."""
    from memory.rag_chunked import ChunkedRAGMemory
    naive = NaiveMemory(max_context_tokens=1200)
    chunked = ChunkedRAGMemory()
    _populate(naive, BENCHMARK_FACTS, 100)
    _populate(chunked, BENCHMARK_FACTS, 100)
    name_fact = BENCHMARK_FACTS[0]
    naive_t = naive.token_count(name_fact.query_text(), 99)
    chunked_t = chunked.token_count(name_fact.query_text(), 99)
    assert chunked_t < naive_t, (
        f"ChunkedRAG ({chunked_t}) should be cheaper than naive ({naive_t})"
    )
    print(f"PASS: ChunkedRAG token cost < naive ({chunked_t} vs {naive_t})")


def test_chunked_rag_benchmark_registration():
    """'rag_chunked' backend must be resolvable from the benchmark runner."""
    from evaluation.benchmark import _make_memory
    mem = _make_memory("rag_chunked")
    assert mem.name == "rag_chunked"
    print(f"PASS: rag_chunked registered in benchmark runner ({mem!r})")


# ── Stats / multi-seed tests ────────────────────────────────────────────────

def test_stats_aggregate_metric():
    """aggregate_metric must return correct mean and std."""
    from evaluation.stats import aggregate_metric
    result = aggregate_metric([0.8, 0.9, 0.7, 0.85, 0.75])
    assert abs(result["mean"] - 0.8) < 0.01, f"Mean wrong: {result['mean']}"
    assert result["std"] > 0, "Std should be > 0 for varied values"
    assert result["ci95_lo"] <= result["mean"] <= result["ci95_hi"]
    print(f"PASS: aggregate_metric (mean={result['mean']:.3f} +/- {result['std']:.3f})")


def test_persona_pool_structure():
    """Each persona must have 8 facts with the same keys as BENCHMARK_FACTS."""
    from simulator.personas import PERSONA_POOL
    expected_keys = {f.key for f in BENCHMARK_FACTS}
    for i, persona in enumerate(PERSONA_POOL):
        persona_keys = {f.key for f in persona}
        assert persona_keys == expected_keys, (
            f"Persona {i} has different fact keys: {persona_keys}"
        )
    print(f"PASS: persona pool structure ({len(PERSONA_POOL)} personas, {len(expected_keys)} keys each)")


# ── SQLite Storage tests ──────────────────────────────────────────────────────

def test_storage_save_and_get_run():
    from utils.storage import Storage
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = Storage(db_path)
        display = {
            "checkpoints": [10, 25],
            "naive": {
                "recall": [1.0, 0.8], "precision": [0.9, 0.7],
                "drift": [0.0, 0.1], "noise": [0.5, 0.8],
                "tokens": [100, 500],
            }
        }
        store.save_run("test_run", {"total_turns": 25, "backends": ["naive"]}, display)
        loaded = store.get_run("test_run")
        assert loaded is not None
        assert loaded["checkpoints"] == [10, 25]
        assert loaded["naive"]["recall"] == [1.0, 0.8]
        assert loaded["naive"]["tokens"] == [100, 500]
    finally:
        store.close()
        os.unlink(db_path)
    print("PASS: storage save and get run")


def test_storage_list_runs():
    from utils.storage import Storage
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = Storage(db_path)
        display = {"checkpoints": [10], "naive": {"recall": [0.5], "precision": [0.5],
                   "drift": [0], "noise": [0], "tokens": [100]}}
        store.save_run("run_b", {"total_turns": 10}, display)
        store.save_run("run_a", {"total_turns": 10}, display)
        runs = store.list_runs(limit=10)
        assert len(runs) >= 2
        ids = [r["run_id"] for r in runs]
        assert "run_a" in ids and "run_b" in ids, f"Missing runs in {ids}"
    finally:
        store.close()
        os.unlink(db_path)
    print("PASS: storage list runs")


def test_storage_compare_runs():
    from utils.storage import Storage
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = Storage(db_path)
        display = {"checkpoints": [10], "naive": {"recall": [0.8], "precision": [0.8],
                   "drift": [0], "noise": [0], "tokens": [100]}}
        store.save_run("run_a", {}, display)
        display["naive"]["recall"] = [0.9]
        store.save_run("run_b", {}, display)
        comp = store.compare_runs("run_a", "run_b")
        assert comp["run_a"]["run_id"] == "run_a"
        assert comp["run_b"]["run_id"] == "run_b"
        assert comp["run_a"]["backends"]["naive"] == [0.8]
        assert comp["run_b"]["backends"]["naive"] == [0.9]
    finally:
        store.close()
        os.unlink(db_path)
    print("PASS: storage compare runs")


def test_storage_get_run_not_found():
    from utils.storage import Storage
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = Storage(db_path)
        assert store.get_run("nonexistent") is None
    finally:
        store.close()
        os.unlink(db_path)
    print("PASS: storage get_run returns None for missing run")


# ── Logger + SQLite integration tests ────────────────────────────────────────

def _clean_csv_row(run_id: str) -> None:
    """Remove a test run_id from runs_summary.csv to avoid accumulation."""
    import csv
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "experiment_logs", "runs_summary.csv"
    )
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("run_id") != run_id:
                rows.append(row)
    if rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        os.unlink(csv_path)


def test_logger_writes_sqlite():
    """log_run must write to SQLite, not just JSON."""
    from evaluation.logger import log_run
    from utils.storage import Storage
    import os

    display = {
        "checkpoints": [10],
        "naive": {"recall": [0.5], "precision": [0.5],
                  "drift": [0], "noise": [0], "tokens": [100]},
    }
    config = {"total_turns": 10, "backends": ["naive"]}

    run_id = "_test_sqlite_logger"
    json_path = log_run(display, config, run_id=run_id)
    assert os.path.exists(json_path), "JSON file must exist (backward compat)"

    # Verify SQLite has the data
    store = Storage()
    loaded = store.get_run(run_id)
    assert loaded is not None, "SQLite must contain the run"
    assert loaded["naive"]["recall"] == [0.5]

    # Cleanup test artifacts (JSON, SQLite, CSV)
    os.unlink(json_path)
    store.conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
    store.conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    store.conn.commit()
    _clean_csv_row(run_id)
    print("PASS: logger writes to SQLite")


def test_list_runs_returns_sqlite_runs():
    """list_runs must return SQLite-backed runs, not just filesystem scans."""
    from evaluation.logger import list_runs
    from utils.storage import Storage

    store = Storage()
    display = {"checkpoints": [10], "naive": {"recall": [0.6], "precision": [0.6],
               "drift": [0], "noise": [0], "tokens": [100]}}
    store.save_run("_test_list_runs", {"total_turns": 10}, display)

    runs = list_runs()
    ids = [r["run_id"] for r in runs]
    assert "_test_list_runs" in ids, "list_runs must include SQLite runs"

    # Cleanup
    store.conn.execute("DELETE FROM results WHERE run_id = ?", ("_test_list_runs",))
    store.conn.execute("DELETE FROM runs WHERE run_id = ?", ("_test_list_runs",))
    store.conn.commit()
    print("PASS: list_runs returns SQLite runs")


if __name__ == "__main__":
    tests = [
        test_conversation_generator,
        test_naive_recall_early,
        test_rag_recall_early,
        test_cascading_recall_early,
        test_temporal_drift_not_applicable_before_update,
        test_temporal_drift_after_update,
        test_token_count_ordering,
        test_noise_ratio_range,
        # SummaryMemory
        test_summary_extractive_fallback_recall_early,
        test_summary_compresses_overflow,
        test_summary_context_contains_summary_and_recent,
        test_summary_reset_clears_state,
        test_summary_token_cost_bounded,
        test_summary_benchmark_registration,
        # Decay functions
        test_decay_functions_range,
        test_ebbinghaus_is_monotone_decreasing,
        test_cascading_uses_pluggable_decay,
        # ChunkedRAG
        test_chunked_rag_recall_early,
        test_chunked_rag_bounded_index,
        test_chunked_rag_tokens_less_than_naive,
        test_chunked_rag_benchmark_registration,
        # Stats / multi-seed
        test_stats_aggregate_metric,
        test_persona_pool_structure,
        # SQLite Storage
        test_storage_save_and_get_run,
        test_storage_list_runs,
        test_storage_compare_runs,
        test_storage_get_run_not_found,
        # Logger + SQLite integration
        test_logger_writes_sqlite,
        test_list_runs_returns_sqlite_runs,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {t.__name__} — {e}")
            failed += 1

    print(f"\n{'All tests passed!' if failed == 0 else f'{failed} test(s) failed.'}")
    sys.exit(failed)
