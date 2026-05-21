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
