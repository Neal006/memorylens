"""
Integration tests for the MemoryLens pipeline.
No LLM calls — purely content-based metric evaluation.
"""

import os
import sys

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memorylens.simulator.facts import BENCHMARK_FACTS, Fact
from memorylens.simulator.conversation import generate_conversation
from memorylens.memory.naive import NaiveMemory
from memorylens.memory.rag import RAGMemory
from memorylens.memory.cascading import CascadingTemporalMemory
from memorylens.memory.summary import SummaryMemory
from memorylens.evaluation.metrics import (
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
    from memorylens.evaluation.benchmark import _make_memory
    mem = _make_memory("summary")
    assert mem.name == "summary"
    print(f"PASS: summary registered in benchmark runner ({mem!r})")


# ── Decay function tests ────────────────────────────────────────────────────

def test_decay_functions_range():
    """All decay functions must return values in [0, 1] for all valid inputs."""
    from memorylens.memory.decay import _REGISTRY
    for name, fn in _REGISTRY.items():
        for age in [0, 1, 5, 10, 50, 99, 100]:
            v = fn(age, 100)
            assert 0.0 <= v <= 1.0, f"{name}({age}, 100) = {v} out of range"
    print("PASS: all decay functions in [0,1] range")


def test_ebbinghaus_is_monotone_decreasing():
    """Ebbinghaus decay must be monotonically non-increasing with age."""
    from memorylens.memory.decay import decay_ebbinghaus
    prev = 1.0
    for age in range(0, 101):
        v = decay_ebbinghaus(age, 100)
        assert v <= prev + 1e-9, f"Ebbinghaus not monotone: age={age}, v={v}, prev={prev}"
        prev = v
    print("PASS: Ebbinghaus decay is monotone decreasing")


def test_cascading_uses_pluggable_decay():
    """CascadingTemporalMemory should accept and store the decay name."""
    from memorylens.memory.cascading import CascadingTemporalMemory
    for name in ["default", "linear", "exponential", "ebbinghaus"]:
        mem = CascadingTemporalMemory(decay=name)
        assert mem.decay_name == name, f"Expected decay_name={name}, got {mem.decay_name}"
    print("PASS: CascadingTemporalMemory accepts all decay variants")


# ── ChunkedRAGMemory tests ──────────────────────────────────────────────────

def test_chunked_rag_recall_early():
    """ChunkedRAGMemory should recall facts with >= 75% accuracy at T=15."""
    from memorylens.memory.rag_chunked import ChunkedRAGMemory
    mem = ChunkedRAGMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.75, f"Expected >=75% recall at T=15 for chunked RAG, got {rate:.0%}"
    print(f"PASS: ChunkedRAGMemory recall early ({rate:.0%})")


def test_chunked_rag_bounded_index():
    """ChunkedRAGMemory must not exceed max_chunks capacity."""
    from memorylens.memory.rag_chunked import ChunkedRAGMemory
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
    from memorylens.memory.rag_chunked import ChunkedRAGMemory
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
    from memorylens.evaluation.benchmark import _make_memory
    mem = _make_memory("rag_chunked")
    assert mem.name == "rag_chunked"
    print(f"PASS: rag_chunked registered in benchmark runner ({mem!r})")


# ── Stats / multi-seed tests ────────────────────────────────────────────────

def test_stats_aggregate_metric():
    """aggregate_metric must return correct mean and std."""
    from memorylens.evaluation.stats import aggregate_metric
    result = aggregate_metric([0.8, 0.9, 0.7, 0.85, 0.75])
    assert abs(result["mean"] - 0.8) < 0.01, f"Mean wrong: {result['mean']}"
    assert result["std"] > 0, "Std should be > 0 for varied values"
    assert result["ci95_lo"] <= result["mean"] <= result["ci95_hi"]
    print(f"PASS: aggregate_metric (mean={result['mean']:.3f} +/- {result['std']:.3f})")


def test_persona_pool_structure():
    """Each persona must have 8 facts with the same keys as BENCHMARK_FACTS."""
    from memorylens.simulator.personas import PERSONA_POOL
    expected_keys = {f.key for f in BENCHMARK_FACTS}
    for i, persona in enumerate(PERSONA_POOL):
        persona_keys = {f.key for f in persona}
        assert persona_keys == expected_keys, (
            f"Persona {i} has different fact keys: {persona_keys}"
        )
    print(f"PASS: persona pool structure ({len(PERSONA_POOL)} personas, {len(expected_keys)} keys each)")


# ── GraphMemory tests ───────────────────────────────────────────────────────

def test_graph_recall_early():
    """GraphMemory should recall all injected facts at T=15 (templated extraction)."""
    from memorylens.memory.graph import GraphMemory
    mem = GraphMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate == 1.0, f"Expected 100% recall at T=15 for graph, got {rate:.0%}"
    print(f"PASS: graph recall early ({rate:.0%})")


def test_graph_update_replaces_edge():
    """After a fact update, the graph must surface the new value and drop the old."""
    from memorylens.memory.graph import GraphMemory
    mem = GraphMemory()
    _populate(mem, BENCHMARK_FACTS, 50)  # city updates at T=40
    ctx = mem.get_context("What is my city?", 49)
    combined = " ".join(m["content"].lower() for m in ctx)
    assert "mumbai" in combined,        "updated city value missing from graph context"
    assert "bangalore" not in combined, "stale city value still present in graph context"
    print("PASS: graph update replaces edge (no stale value)")


def test_graph_reset_and_registration():
    from memorylens.evaluation.benchmark import _make_memory
    mem = _make_memory("graph")
    assert mem.name == "graph"
    mem.add_message("user", "My name is Test User.", 0)
    mem.reset()
    assert mem.get_context("What is my name?", 1) == []
    print("PASS: graph reset + benchmark registration")


def test_cascading_cold_tier_retains_early_facts():
    """Regression (cold merge order): facts injected at T=0-9 must survive
    compression into the cold tier and stay recallable at T=100."""
    mem = CascadingTemporalMemory()
    _populate(mem, BENCHMARK_FACTS, 100)
    results = [recall_at_t(mem, f, 99) for f in BENCHMARK_FACTS]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.85, f"Cold-tier recall regressed: {rate:.0%} at T=100"
    print(f"PASS: cascading cold-tier retains early facts ({rate:.0%} at T=100)")


def test_cascading_drift_zero_after_update():
    """Regression (issue #16): cold summaries must be patched on fact updates."""
    mem = CascadingTemporalMemory()
    _populate(mem, BENCHMARK_FACTS, 100)  # city updates at T=40, age at T=60
    updated = [f for f in BENCHMARK_FACTS if f.updated_at]
    for f in updated:
        drift = temporal_drift_score(mem, f, 99)
        assert drift["drift"] == 0.0, f"{f.key}: stale value survived, drift={drift}"
    print("PASS: cascading drift stays 0.0 after updates")


# ── contradiction_score tests ───────────────────────────────────────────────

def test_contradiction_not_applicable_before_update():
    from memorylens.evaluation.metrics import contradiction_score
    mem = NaiveMemory()
    _populate(mem, BENCHMARK_FACTS, 10)
    city = next(f for f in BENCHMARK_FACTS if f.key == "city")
    result = contradiction_score(mem, city, 9)
    assert result["applicable"] is False
    print("PASS: contradiction not applicable before update turn")


def test_contradiction_naive_surfaces_both_values():
    """Naive keeps full history, so both old and new city must co-occur → 1.0."""
    from memorylens.evaluation.metrics import contradiction_score
    mem = NaiveMemory(max_context_tokens=8000)
    _populate(mem, BENCHMARK_FACTS, 50)  # city: Bangalore → Mumbai at T=40
    city = next(f for f in BENCHMARK_FACTS if f.key == "city")
    result = contradiction_score(mem, city, 49)
    assert result["applicable"] is True
    assert result["contradiction"] == 1.0, (
        f"Expected contradiction=1.0 for naive full history, got {result}"
    )
    print("PASS: contradiction detected in naive full history")


def test_contradiction_graph_is_zero():
    """GraphMemory patches facts in place, so no contradiction can survive."""
    from memorylens.evaluation.metrics import contradiction_score
    from memorylens.memory.graph import GraphMemory
    mem = GraphMemory()
    _populate(mem, BENCHMARK_FACTS, 50)
    city = next(f for f in BENCHMARK_FACTS if f.key == "city")
    result = contradiction_score(mem, city, 49)
    assert result["contradiction"] == 0.0, f"Graph should never contradict, got {result}"
    print("PASS: contradiction zero for graph backend")


def test_contradiction_in_benchmark_output():
    """Benchmark display dict must include the contradiction series."""
    from memorylens.evaluation.benchmark import run_benchmark, results_to_display_dict
    raw = run_benchmark(total_turns=10, eval_checkpoints=[10], backends=["naive"])
    display = results_to_display_dict(raw)
    assert "contradiction" in display["naive"], "contradiction series missing from display dict"
    print("PASS: contradiction wired into benchmark output")


# ── Scenario tests ──────────────────────────────────────────────────────────

def test_scenario_registry_complete():
    from memorylens.simulator.scenarios import SCENARIOS
    assert set(SCENARIOS) == {"default", "edtech", "support", "medical"}
    for s in SCENARIOS.values():
        s.validate()
        assert len(s.facts) == 8, f"{s.name}: expected 8 facts, got {len(s.facts)}"
        assert any(f.updated_at for f in s.facts), f"{s.name}: needs at least one fact update"
    print(f"PASS: scenario registry ({list(SCENARIOS)})")


def test_scenario_unknown_raises():
    from memorylens.simulator.scenarios import get_scenario
    try:
        get_scenario("nonexistent")
        assert False, "Expected ValueError for unknown scenario"
    except ValueError:
        pass
    print("PASS: unknown scenario raises ValueError")


def test_scenario_values_disjoint_on_update():
    """Old/new values must not be substrings of each other, or drift and
    contradiction metrics silently break."""
    from memorylens.simulator.scenarios import SCENARIOS
    for s in SCENARIOS.values():
        for persona in s.persona_pool:
            for f in persona:
                if f.updated_value:
                    old, new = f.value.lower(), f.updated_value.lower()
                    assert old not in new and new not in old, (
                        f"{s.name}/{f.key}: '{f.value}' and '{f.updated_value}' overlap"
                    )
    print("PASS: scenario update values are substring-disjoint")


def test_scenario_benchmark_runs():
    """A non-default scenario must run end-to-end through the benchmark."""
    from memorylens.evaluation.benchmark import run_benchmark
    from memorylens.simulator.scenarios import get_scenario
    s = get_scenario("support")
    raw = run_benchmark(
        total_turns=15, eval_checkpoints=[15], facts=s.facts,
        backends=["naive"], filler_turns=s.filler_turns,
    )
    cp = raw["naive"].checkpoints[0]
    assert cp.recall == 1.0, f"Naive should have full recall at T=15, got {cp.recall}"
    print(f"PASS: support scenario end-to-end (recall={cp.recall:.0%})")


# ── FAISS backend (optional dependency) ─────────────────────────────────────

def test_faiss_backend_or_missing_dep_error():
    try:
        import faiss  # noqa: F401
    except ImportError:
        from memorylens.evaluation.benchmark import _make_memory
        try:
            _make_memory("faiss")
            assert False, "Expected ImportError when faiss is not installed"
        except ImportError as e:
            assert "memorylens[faiss]" in str(e), f"Error should name the extra: {e}"
        print("SKIP: faiss not installed (missing-dep error message verified)")
        return

    from memorylens.memory.vector_faiss import FAISSMemory
    mem = FAISSMemory()
    _populate(mem, BENCHMARK_FACTS, 15)
    active = [f for f in BENCHMARK_FACTS if f.injected_at < 15]
    results = [recall_at_t(mem, f, 14) for f in active]
    rate = sum(r["recalled"] for r in results) / len(results)
    assert rate >= 0.75, f"Expected >=75% recall at T=15 for faiss, got {rate:.0%}"
    print(f"PASS: faiss recall early ({rate:.0%})")


# ── API server (optional dependency) ────────────────────────────────────────

def test_api_benchmark_lifecycle():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("SKIP: fastapi/httpx not installed")
        return
    import time
    from memorylens.api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert "naive" in client.get("/v1/backends").json()["data"]
    scenario_names = [s["name"] for s in client.get("/v1/scenarios").json()["data"]]
    assert "medical" in scenario_names

    assert client.post("/v1/benchmarks", json={"backends": ["bogus"]}).status_code == 422

    resp = client.post("/v1/benchmarks", json={
        "turns": 10, "checkpoints": [10], "backends": ["naive"],
    })
    assert resp.status_code == 202
    job_id = resp.json()["data"]["job_id"]

    for _ in range(150):
        job = client.get(f"/v1/benchmarks/{job_id}").json()["data"]
        if job["status"] != "running":
            break
        time.sleep(0.2)
    assert job["status"] == "completed", f"Job did not complete: {job.get('error')}"
    assert "naive" in job["results"]
    print("PASS: API benchmark lifecycle (submit, poll, results)")


# ── SQLite Storage tests ─────────────────────────────────────────────────────

def _temp_store():
    import tempfile
    from memorylens.utils.storage import Storage
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Storage(f.name), f.name


def test_storage_save_and_get_run():
    store, db_path = _temp_store()
    try:
        display = {
            "checkpoints": [10, 25],
            "naive": {
                "recall": [1.0, 0.8], "precision": [0.9, 0.7],
                "drift": [0.0, 0.1], "noise": [0.5, 0.8],
                "tokens": [100, 500], "contradiction": [0.0, 1.0],
            }
        }
        store.save_run("test_run", {"total_turns": 25, "backends": ["naive"]}, display)
        loaded = store.get_run("test_run")
        assert loaded is not None
        assert loaded["checkpoints"] == [10, 25]
        assert loaded["naive"]["recall"] == [1.0, 0.8]
        assert loaded["naive"]["tokens"] == [100, 500]
        assert loaded["naive"]["contradiction"] == [0.0, 1.0]
    finally:
        store.close()
        os.unlink(db_path)
    print("PASS: storage save and get run")


def test_storage_list_runs():
    store, db_path = _temp_store()
    try:
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
    store, db_path = _temp_store()
    try:
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
    store, db_path = _temp_store()
    try:
        assert store.get_run("nonexistent") is None
    finally:
        store.close()
        os.unlink(db_path)
    print("PASS: storage get_run returns None for missing run")


def test_storage_save_run_idempotent():
    """Calling save_run twice with the same run_id must not duplicate rows."""
    store, db_path = _temp_store()
    try:
        display = {"checkpoints": [10], "naive": {"recall": [0.8], "precision": [0.8],
                   "drift": [0], "noise": [0], "tokens": [100]}}
        store.save_run("dup_test", {}, display)
        store.save_run("dup_test", {}, display)  # same run_id again
        loaded = store.get_run("dup_test")
        assert loaded is not None
        assert len(loaded["naive"]["recall"]) == 1, "Duplicate rows detected!"
        assert loaded["naive"]["recall"] == [0.8]
    finally:
        store.close()
        os.unlink(db_path)
    print("PASS: storage save_run idempotent")


# ── Logger + SQLite integration tests ────────────────────────────────────────

def _clean_csv_row(run_id: str) -> None:
    """Remove a test run_id from runs_summary.csv to avoid accumulation."""
    import csv
    from memorylens.utils.storage import LOG_DIR
    csv_path = os.path.join(LOG_DIR, "runs_summary.csv")
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
    from memorylens.evaluation.logger import log_run
    from memorylens.utils.storage import Storage

    display = {
        "checkpoints": [10],
        "naive": {"recall": [0.5], "precision": [0.5],
                  "drift": [0], "noise": [0], "tokens": [100]},
    }
    config = {"total_turns": 10, "backends": ["naive"]}

    run_id = "_test_sqlite_logger"
    json_path = log_run(display, config, run_id=run_id)
    assert os.path.exists(json_path), "JSON file must exist (backward compat)"

    store = Storage()
    loaded = store.get_run(run_id)
    assert loaded is not None, "SQLite must contain the run"
    assert loaded["naive"]["recall"] == [0.5]

    # Cleanup test artifacts (JSON, SQLite, CSV)
    os.unlink(json_path)
    store.conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
    store.conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    store.conn.commit()
    store.close()
    _clean_csv_row(run_id)
    print("PASS: logger writes to SQLite")


def test_list_runs_returns_sqlite_runs():
    """list_runs must return SQLite-backed runs, not just filesystem scans."""
    from memorylens.evaluation.logger import list_runs
    from memorylens.utils.storage import Storage

    store = Storage()
    display = {"checkpoints": [10], "naive": {"recall": [0.6], "precision": [0.6],
               "drift": [0], "noise": [0], "tokens": [100]}}
    store.save_run("_test_list_runs", {"total_turns": 10}, display)

    runs = list_runs()
    ids = [r["run_id"] for r in runs]
    assert "_test_list_runs" in ids, "list_runs must include SQLite runs"

    store.conn.execute("DELETE FROM results WHERE run_id = ?", ("_test_list_runs",))
    store.conn.execute("DELETE FROM runs WHERE run_id = ?", ("_test_list_runs",))
    store.conn.commit()
    store.close()
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
        # Cascading regressions
        test_cascading_cold_tier_retains_early_facts,
        test_cascading_drift_zero_after_update,
        # GraphMemory
        test_graph_recall_early,
        test_graph_update_replaces_edge,
        test_graph_reset_and_registration,
        # contradiction_score
        test_contradiction_not_applicable_before_update,
        test_contradiction_naive_surfaces_both_values,
        test_contradiction_graph_is_zero,
        test_contradiction_in_benchmark_output,
        # Scenarios
        test_scenario_registry_complete,
        test_scenario_unknown_raises,
        test_scenario_values_disjoint_on_update,
        test_scenario_benchmark_runs,
        # Optional deps
        test_faiss_backend_or_missing_dep_error,
        test_api_benchmark_lifecycle,
        # SQLite Storage
        test_storage_save_and_get_run,
        test_storage_list_runs,
        test_storage_compare_runs,
        test_storage_get_run_not_found,
        test_storage_save_run_idempotent,
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
