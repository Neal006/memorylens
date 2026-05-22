from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Callable, Optional

from simulator.facts import Fact, BENCHMARK_FACTS
from simulator.conversation import generate_conversation
from memory.naive import NaiveMemory
from memory.rag import RAGMemory
from memory.cascading import CascadingTemporalMemory
from memory.summary import SummaryMemory
from memory.base import BaseMemory
from evaluation.metrics import (
    recall_at_t, temporal_drift_score, memory_noise_ratio, precision_at_k,
    cascade_efficiency, llm_recall_at_t, llm_temporal_drift,
)

if TYPE_CHECKING:
    from utils.providers import LLMProvider

OFF_TOPIC_QUERY = "What is the best sorting algorithm for large datasets?"

_NAN = float("nan")


@dataclass
class CheckpointResult:
    turn:         int
    # ── Content-based (always available, fast) ───────────────────────────────
    recall:       float         # substring match on retrieved chunks
    precision:    float
    drift:        float
    noise:        float
    tokens:       int
    cascade_eff:  float = 1.0
    # ── LLM-based (available when a provider is configured) ──────────────────
    llm_recall:   float = _NAN  # actual LLM answer judged correct/wrong
    llm_drift:    float = _NAN  # LLM gives old vs new value after update
    has_llm_eval: bool  = False


@dataclass
class BackendResult:
    name: str
    checkpoints: List[CheckpointResult] = field(default_factory=list)
    raw_recalls:  List[Dict]            = field(default_factory=list)
    provider_name: Optional[str]        = None   # which LLM was used, if any


def _make_memory(name: str) -> BaseMemory:
    if name == "naive":
        return NaiveMemory(max_context_tokens=1200)
    if name == "rag":
        return RAGMemory()
    if name == "cascading":
        return CascadingTemporalMemory()
    if name == "summary":
        return SummaryMemory(window_size=20, use_llm=None)
    raise ValueError(
        f"Unknown backend: '{name}'. Choose from: naive, rag, cascading, summary"
    )


def run_benchmark(
    total_turns:      int                        = 100,
    eval_checkpoints: Optional[List[int]]        = None,
    facts:            Optional[List[Fact]]       = None,
    backends:         Optional[List[str]]        = None,
    provider:         Optional["LLMProvider"]    = None,
    progress:         Optional[Callable[[str], None]] = None,
) -> Dict[str, BackendResult]:
    """
    Run the full MemoryLens benchmark.

    Parameters
    ----------
    provider : LLMProvider | None
        When supplied, a second LLM-evaluation pass runs at every checkpoint
        alongside the fast content-based pass.  When None, only content-based
        metrics are computed.
    """
    if eval_checkpoints is None:
        eval_checkpoints = [10, 25, 50, 75, 100]
    if facts is None:
        facts = BENCHMARK_FACTS
    if backends is None:
        backends = ["naive", "rag", "cascading"]

    total_turns = max(total_turns, max(eval_checkpoints))
    events = generate_conversation(facts, total_turns)
    checkpoint_set = set(eval_checkpoints)
    results: Dict[str, BackendResult] = {}

    if provider and progress:
        progress(f"LLM provider: {provider.name}")
    elif progress:
        progress("No LLM provider — running content-only mode (fast)")

    # Shadow memories for cascade_efficiency
    _naive_shadow   = _make_memory("naive")
    _cascade_shadow = _make_memory("cascading")

    for backend_name in backends:
        if progress:
            progress(f"▶  Backend: {backend_name}")

        memory = _make_memory(backend_name)
        result = BackendResult(
            name=backend_name,
            provider_name=provider.name if provider else None,
        )
        known_values: List[str] = []

        for event in events:
            turn = event["turn"]
            ack  = "Understood." if event["is_fact"] else "I can help with that."

            memory.add_message("user", event["content"], turn)
            memory.add_message("assistant", ack, turn)

            if backend_name == "naive":
                _naive_shadow.add_message("user", event["content"], turn)
                _naive_shadow.add_message("assistant", ack, turn)
            elif backend_name == "cascading":
                _cascade_shadow.add_message("user", event["content"], turn)
                _cascade_shadow.add_message("assistant", ack, turn)

            if event["is_fact"]:
                for f in facts:
                    if f.key == event["fact_key"]:
                        val = f.current_value(turn)
                        if val not in known_values:
                            known_values.append(val)

            if (turn + 1) in checkpoint_set:
                cp = turn + 1
                if progress:
                    progress(f"  Evaluating @ T={cp} ...")

                active_facts = [f for f in facts if f.injected_at <= turn]

                # ── Content-based pass (always) ───────────────────────────
                recalls    = [recall_at_t(memory, f, turn) for f in active_facts]
                avg_recall = sum(r["recalled"] for r in recalls) / max(1, len(recalls))
                avg_tokens = sum(r["tokens"]   for r in recalls) / max(1, len(recalls))

                for r in recalls:
                    result.raw_recalls.append({"turn": cp, **r})

                prec = precision_at_k(memory, active_facts, turn)

                drift_facts = [
                    f for f in active_facts
                    if f.updated_at and f.updated_at <= turn
                ]
                avg_drift = (
                    sum(temporal_drift_score(memory, f, turn)["drift"] for f in drift_facts)
                    / len(drift_facts)
                    if drift_facts else 0.0
                )

                noise = memory_noise_ratio(memory, OFF_TOPIC_QUERY, known_values, turn)

                eff = 1.0
                if backend_name == "cascading" and "naive" in backends:
                    eff = cascade_efficiency(
                        _cascade_shadow, _naive_shadow, active_facts, turn
                    )

                # ── LLM pass (when provider is available) ─────────────────
                llm_recall_val = _NAN
                llm_drift_val  = _NAN
                has_llm        = False

                if provider:
                    has_llm = True
                    llm_results = [
                        llm_recall_at_t(memory, f, turn, provider)
                        for f in active_facts
                    ]
                    llm_recall_val = (
                        sum(r["llm_recalled"] for r in llm_results)
                        / max(1, len(llm_results))
                    )

                    if drift_facts:
                        drift_llm_results = [
                            llm_temporal_drift(memory, f, turn, provider)
                            for f in drift_facts
                        ]
                        applicable = [r for r in drift_llm_results if r["applicable"]]
                        llm_drift_val = (
                            sum(r["llm_drift"] for r in applicable) / len(applicable)
                            if applicable else 0.0
                        )
                    else:
                        llm_drift_val = 0.0

                result.checkpoints.append(CheckpointResult(
                    turn         = cp,
                    recall       = round(avg_recall, 4),
                    precision    = round(prec, 4),
                    drift        = round(avg_drift, 4),
                    noise        = round(noise, 4),
                    tokens       = int(avg_tokens),
                    cascade_eff  = round(eff, 4),
                    llm_recall   = round(llm_recall_val, 4) if has_llm else _NAN,
                    llm_drift    = round(llm_drift_val,  4) if has_llm else _NAN,
                    has_llm_eval = has_llm,
                ))

        results[backend_name] = result
        if progress:
            progress(f"  ✓ {backend_name} done.")

    return results


def results_to_display_dict(results: Dict[str, BackendResult]) -> Dict:
    """Convert BackendResult objects into a JSON-serialisable dict for the dashboard."""
    import math
    checkpoints = sorted({cp.turn for r in results.values() for cp in r.checkpoints})
    display: Dict = {"checkpoints": checkpoints, "has_llm_eval": False}

    for name, result in results.items():
        cp_map = {cp.turn: cp for cp in result.checkpoints}
        has_llm = any(cp.has_llm_eval for cp in result.checkpoints)
        if has_llm:
            display["has_llm_eval"] = True

        display[name] = {
            "recall":       [cp_map[t].recall      for t in checkpoints if t in cp_map],
            "precision":    [cp_map[t].precision   for t in checkpoints if t in cp_map],
            "drift":        [cp_map[t].drift       for t in checkpoints if t in cp_map],
            "noise":        [cp_map[t].noise       for t in checkpoints if t in cp_map],
            "tokens":       [cp_map[t].tokens      for t in checkpoints if t in cp_map],
            "cascade_eff":  [cp_map[t].cascade_eff for t in checkpoints if t in cp_map],
            # LLM metrics — None where not available
            "llm_recall":   [
                None if math.isnan(cp_map[t].llm_recall) else cp_map[t].llm_recall
                for t in checkpoints if t in cp_map
            ],
            "llm_drift":    [
                None if math.isnan(cp_map[t].llm_drift) else cp_map[t].llm_drift
                for t in checkpoints if t in cp_map
            ],
            "provider":     result.provider_name,
        }

    return display
