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
