from typing import TYPE_CHECKING, Dict, List, Optional
from memorylens.memory.base import BaseMemory
from memorylens.simulator.facts import Fact

if TYPE_CHECKING:
    from memorylens.utils.providers import LLMProvider


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
    if fact.updated_at is None or not fact.updated_value or current_turn < fact.updated_at:
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


def contradiction_score(memory: BaseMemory, fact: Fact, current_turn: int) -> Dict:
    """
    Contradiction — after a fact update, does the retrieved context surface
    BOTH the old and the new value at once?

    A context containing both "Bangalore" and "Mumbai" for the same fact key
    forces the downstream LLM to arbitrate between conflicting values — a
    distinct failure mode from drift (which measures stale-only retrieval).

    Returns contradiction ∈ {0.0, 1.0}; applicable only after fact.updated_at.
    """
    if fact.updated_at is None or not fact.updated_value or current_turn < fact.updated_at:
        return {"contradiction": 0.0, "applicable": False}

    context = memory.get_context(fact.query_text(), current_turn)
    old_val = fact.value.lower()
    new_val = (fact.updated_value or "").lower()

    old_present = any(old_val in m.get("content", "").lower() for m in context)
    new_present = any(new_val in m.get("content", "").lower() for m in context)

    return {
        "contradiction": 1.0 if (old_present and new_present) else 0.0,
        "old_present": old_present,
        "new_present": new_present,
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


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based metrics  (require a provider — degrade gracefully without one)
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = (
    "You are a helpful assistant with access to a conversation history. "
    "Answer the user's question using ONLY the information in the context. "
    "Reply with the answer value only — no explanation, no extra words."
)

_JUDGE_SYSTEM = (
    "You are a strict fact-checker. Given a question, the correct answer, "
    "and a model's response, reply with ONLY 'correct' or 'wrong'."
)


def llm_recall_at_t(
    memory: BaseMemory,
    fact: Fact,
    current_turn: int,
    provider: "LLMProvider",
) -> Dict:
    """
    LLM Recall@T — the model is actually asked the question and its answer
    is judged for correctness.

    Two-stage pipeline:
      1. ANSWER  — LLM answers fact.query_text() given memory context
      2. JUDGE   — a second LLM call checks if the answer is correct

    Returns
    -------
    dict with keys:
      llm_recalled  : bool   — judge says the answer is correct
      answer        : str    — what the LLM actually said
      expected      : str    — ground-truth value
      judge_verdict : str    — 'correct' | 'wrong' | 'error'
      tokens        : int    — context token estimate
    """
    from memorylens.utils.providers import _clean_messages

    context = memory.get_context(fact.query_text(), current_turn)
    expected = fact.current_value(current_turn)

    # ── Stage 1: Answer ──────────────────────────────────────────────────────
    messages = _clean_messages(
        [{"role": "system", "content": _ANSWER_SYSTEM}]
        + context
        + [{"role": "user", "content": fact.query_text() + " Answer with just the value."}]
    )
    answer = provider.chat(messages, max_tokens=60, temperature=0.0)
    tokens = memory.token_count(fact.query_text(), current_turn)

    if answer.startswith("[PROVIDER_ERROR"):
        return {
            "llm_recalled": False, "answer": answer,
            "expected": expected, "judge_verdict": "error", "tokens": tokens,
        }

    # ── Stage 2: Judge ───────────────────────────────────────────────────────
    judge_prompt = (
        f"Question: {fact.query_text()}\n"
        f"Correct answer: {expected}\n"
        f"Model response: {answer}\n"
        f"Is the model response correct? Reply with ONLY 'correct' or 'wrong'."
    )
    verdict_raw = provider.chat(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": judge_prompt},
        ],
        max_tokens=10,
        temperature=0.0,
    ).lower().strip()

    verdict = "correct" if "correct" in verdict_raw else "wrong"

    return {
        "llm_recalled":  verdict == "correct",
        "answer":        answer,
        "expected":      expected,
        "judge_verdict": verdict,
        "tokens":        tokens,
    }


def llm_temporal_drift(
    memory: BaseMemory,
    fact: Fact,
    current_turn: int,
    provider: "LLMProvider",
) -> Dict:
    """
    LLM Temporal Drift — asks the LLM for the *current* value of an updated
    fact and checks whether it returns the new or old value.

    Only meaningful after fact.updated_at has passed.
    """
    from memorylens.utils.providers import _clean_messages

    if fact.updated_at is None or not fact.updated_value or current_turn < fact.updated_at:
        return {"llm_drift": 0.0, "applicable": False}

    context = memory.get_context(fact.query_text(), current_turn)
    new_val  = (fact.updated_value or "").lower()
    old_val  = fact.value.lower()

    messages = _clean_messages(
        [{"role": "system", "content": _ANSWER_SYSTEM}]
        + context
        + [{"role": "user", "content":
            f"What is my current {fact.key.replace('_', ' ')}? "
            "Reply with the current value only."}]
    )
    answer = provider.chat(messages, max_tokens=30, temperature=0.0).lower()

    using_old = old_val in answer and new_val not in answer
    drift = 1.0 if using_old else 0.0

    return {
        "llm_drift":  drift,
        "answer":     answer,
        "expected":   fact.updated_value,
        "old_value":  fact.value,
        "applicable": True,
    }
