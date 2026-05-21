from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional

from simulator.facts import Fact, BENCHMARK_FACTS
from simulator.conversation import generate_conversation
from memory.naive import NaiveMemory
from memory.rag import RAGMemory
from memory.cascading import CascadingTemporalMemory
from memory.base import BaseMemory
from evaluation.metrics import (
    recall_at_t, temporal_drift_score, memory_noise_ratio, precision_at_k
)

OFF_TOPIC_QUERY = "What is the best sorting algorithm for large datasets?"


@dataclass
class CheckpointResult:
    turn: int
    recall: float
    precision: float
    drift: float
    noise: float
    tokens: int


@dataclass
class BackendResult:
    name: str
    checkpoints: List[CheckpointResult] = field(default_factory=list)
    raw_recalls: List[Dict] = field(default_factory=list)


def _make_memory(name: str) -> BaseMemory:
    return {"naive": NaiveMemory, "rag": RAGMemory, "cascading": CascadingTemporalMemory}[name]()


def run_benchmark(
    total_turns: int = 100,
    eval_checkpoints: Optional[List[int]] = None,
    facts: Optional[List[Fact]] = None,
    backends: Optional[List[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, BackendResult]:

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

    for backend_name in backends:
        if progress:
            progress(f"▶ Starting backend: {backend_name}")

        memory = _make_memory(backend_name)
        result = BackendResult(name=backend_name)
        known_values: List[str] = []

        for event in events:
            turn = event["turn"]
            memory.add_message("user", event["content"], turn)

            # Simulate a short assistant acknowledgment so history alternates roles
            ack = "Understood." if event["is_fact"] else "I can help with that."
            memory.add_message("assistant", ack, turn)

            if event["is_fact"]:
                key = event["fact_key"]
                for f in facts:
                    if f.key == key:
                        val = f.current_value(turn)
                        if val not in known_values:
                            known_values.append(val)

            if (turn + 1) in checkpoint_set:
                cp = turn + 1
                if progress:
                    progress(f"  Evaluating {backend_name} @ T={cp} ...")

                active_facts = [f for f in facts if f.injected_at <= turn]

                # --- Recall@T ---
                recalls = [recall_at_t(memory, f, turn) for f in active_facts]
                avg_recall = sum(r["recalled"] for r in recalls) / max(1, len(recalls))
                avg_tokens = sum(r["tokens"] for r in recalls) / max(1, len(recalls))

                for r in recalls:
                    result.raw_recalls.append({"turn": cp, **r})

                # --- Precision@K ---
                prec = precision_at_k(memory, active_facts, turn)

                # --- Temporal Drift ---
                drift_facts = [f for f in active_facts if f.updated_at and f.updated_at <= turn]
                if drift_facts:
                    drifts = [temporal_drift_score(memory, f, turn)["drift"] for f in drift_facts]
                    avg_drift = sum(drifts) / len(drifts)
                else:
                    avg_drift = 0.0

                # --- Noise Ratio ---
                noise = memory_noise_ratio(memory, OFF_TOPIC_QUERY, known_values, turn)

                result.checkpoints.append(CheckpointResult(
                    turn=cp,
                    recall=round(avg_recall, 4),
                    precision=round(prec, 4),
                    drift=round(avg_drift, 4),
                    noise=round(noise, 4),
                    tokens=int(avg_tokens),
                ))

        results[backend_name] = result
        if progress:
            progress(f"  ✓ {backend_name} complete.")

    return results


def results_to_display_dict(results: Dict[str, BackendResult]) -> Dict:
    """Convert BackendResult objects into a JSON-serialisable dict for the dashboard."""
    checkpoints = sorted({cp.turn for r in results.values() for cp in r.checkpoints})
    display: Dict = {"checkpoints": checkpoints}

    for name, result in results.items():
        cp_map = {cp.turn: cp for cp in result.checkpoints}
        display[name] = {
            "recall":    [cp_map[t].recall    for t in checkpoints if t in cp_map],
            "precision": [cp_map[t].precision for t in checkpoints if t in cp_map],
            "drift":     [cp_map[t].drift     for t in checkpoints if t in cp_map],
            "noise":     [cp_map[t].noise     for t in checkpoints if t in cp_map],
            "tokens":    [cp_map[t].tokens    for t in checkpoints if t in cp_map],
        }

    return display
