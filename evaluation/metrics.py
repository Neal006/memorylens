from typing import Dict, List, Optional
from memory.base import BaseMemory
from simulator.facts import Fact


def recall_at_t(memory: BaseMemory, fact: Fact, current_turn: int) -> Dict:
    """
    Recall@T — does the retrieved context contain the correct current fact value?
    Content-based check: no LLM call needed, fully reproducible.
    """
    context = memory.get_context(fact.query_text(), current_turn)
    expected = fact.current_value(current_turn).lower()

    recalled = any(expected in msg.get("content", "").lower() for msg in context)
    tokens = memory.token_count(fact.query_text(), current_turn)

    return {
        "recalled": recalled,
        "expected": fact.current_value(current_turn),
        "tokens": tokens,
        "context_chunks": len(context),
    }


def temporal_drift_score(memory: BaseMemory, fact: Fact, current_turn: int) -> Dict:
    """
    Temporal Drift — after a fact update, does the context still surface the OLD value?
    Returns drift ∈ [0, 1]: 1 = context only shows stale data, 0 = fully updated.
    Only applicable to facts that have an update.
    """
    if not fact.updated_at or current_turn < fact.updated_at:
        return {"drift": 0.0, "applicable": False}

    context = memory.get_context(fact.query_text(), current_turn)
    old_val = fact.value.lower()
    new_val = (fact.updated_value or "").lower()

    old_hits = sum(1 for m in context if old_val in m.get("content", "").lower())
    new_hits = sum(1 for m in context if new_val in m.get("content", "").lower())

    total = old_hits + new_hits
    if total == 0:
        drift = 0.5  # ambiguous — neither value found
    else:
        drift = old_hits / total  # proportion of stale references

    return {
        "drift": drift,
        "old_hits": old_hits,
        "new_hits": new_hits,
        "applicable": True,
    }


def memory_noise_ratio(memory: BaseMemory, off_topic_query: str, known_facts: List[str], current_turn: int) -> float:
    """
    Memory Noise Ratio — of the retrieved context chunks, what fraction is irrelevant?
    Measured by checking whether chunks contain any known injected fact values.
    """
    context = memory.get_context(off_topic_query, current_turn)
    if not context:
        return 0.0

    relevant = 0
    for msg in context:
        content = msg.get("content", "").lower()
        if any(fv.lower() in content for fv in known_facts):
            relevant += 1

    return 1.0 - relevant / len(context)


def precision_at_k(memory: BaseMemory, facts: List[Fact], current_turn: int, k: int = 5) -> float:
    """
    Precision@K — of the first K context chunks retrieved for all facts combined,
    what fraction is actually relevant (contains a fact value)?
    """
    all_fact_values = [f.current_value(current_turn).lower() for f in facts if f.injected_at <= current_turn]
    if not all_fact_values:
        return 0.0

    combined_query = " ".join(f.query_text() for f in facts[:3])
    context = memory.get_context(combined_query, current_turn)[:k]
    if not context:
        return 0.0

    relevant = sum(
        1 for msg in context
        if any(fv in msg.get("content", "").lower() for fv in all_fact_values)
    )
    return relevant / len(context)


def cascade_efficiency(
    cascading_memory: BaseMemory,
    naive_memory: BaseMemory,
    facts: List[Fact],
    current_turn: int,
) -> float:
    """
    Cascade Efficiency — composite score showing how much better cascading is
    vs naive on the recall-per-token frontier.

    Score = (cascading_recall / cascading_tokens) / (naive_recall / naive_tokens)

    > 1.0 means cascading delivers more recall per token than naive.
    = 1.0 means equivalent.
    < 1.0 means naive is more efficient (shouldn't happen at scale).
    """
    active = [f for f in facts if f.injected_at <= current_turn]
    if not active:
        return 1.0

    def _stats(mem: BaseMemory):
        results = [recall_at_t(mem, f, current_turn) for f in active]
        r = sum(x["recalled"] for x in results) / len(results)
        t = sum(x["tokens"] for x in results) / len(results)
        return r, max(t, 1)

    c_recall, c_tokens = _stats(cascading_memory)
    n_recall, n_tokens = _stats(naive_memory)

    cascading_rpt = c_recall / c_tokens
    naive_rpt = n_recall / n_tokens

    if naive_rpt == 0:
        return float("inf")
    return round(cascading_rpt / naive_rpt, 4)
