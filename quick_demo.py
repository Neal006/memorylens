"""
quick_demo.py — Run the full MemoryLens evaluation pipeline with NO API key.

Uses only local embeddings (sentence-transformers) and content-based metrics.
All evaluation is deterministic and reproducible.

Usage:
    python quick_demo.py
    python quick_demo.py --turns 50
"""

import os
import sys
import argparse

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
sys.path.insert(0, os.path.dirname(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryLens quick demo (no API key needed)")
    parser.add_argument("--turns",  type=int, default=100)
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    checkpoints = [t for t in [10, 25, 50, 75, 100] if t <= args.turns]
    if not checkpoints:
        checkpoints = [args.turns]

    from memorylens.simulator.facts import BENCHMARK_FACTS
    from memorylens.simulator.conversation import generate_conversation
    from memorylens.memory.naive import NaiveMemory
    from memorylens.memory.rag import RAGMemory
    from memorylens.memory.cascading import CascadingTemporalMemory
    from memorylens.evaluation.metrics import (
        recall_at_t, temporal_drift_score, memory_noise_ratio,
        precision_at_k, cascade_efficiency,
    )

    if not args.quiet:
        print("=" * 60)
        print("  MemoryLens — Quick Demo  (no API key required)")
        print("=" * 60)
        print(f"  Turns: {args.turns}   Checkpoints: {checkpoints}")
        print(f"  Facts: {len(BENCHMARK_FACTS)}")
        print()
        print("  Loading sentence-transformer model...")

    facts = BENCHMARK_FACTS
    events = generate_conversation(facts, args.turns)

    backends = {
        "naive":     NaiveMemory(max_context_tokens=1200),
        "rag":       RAGMemory(),
        "cascading": CascadingTemporalMemory(),
    }

    # Storage for results
    recall_table:  dict = {n: {} for n in backends}
    tokens_table:  dict = {n: {} for n in backends}
    drift_table:   dict = {n: {} for n in backends}
    noise_table:   dict = {n: {} for n in backends}
    eff_table:     dict = {"cascading": {}}

    checkpoint_set = set(checkpoints)
    known_values: list = []

    for ev in events:
        turn = ev["turn"]
        ack = "Got it." if ev["is_fact"] else "Sure."
        for mem in backends.values():
            mem.add_message("user", ev["content"], turn)
            mem.add_message("assistant", ack, turn)

        if ev["is_fact"]:
            for f in facts:
                if f.key == ev["fact_key"]:
                    val = f.current_value(turn)
                    if val not in known_values:
                        known_values.append(val)

        if (turn + 1) in checkpoint_set:
            cp = turn + 1
            active = [f for f in facts if f.injected_at <= turn]

            for name, mem in backends.items():
                recalls = [recall_at_t(mem, f, turn) for f in active]
                recall_table[name][cp] = sum(r["recalled"] for r in recalls) / len(recalls)
                tokens_table[name][cp] = int(sum(r["tokens"] for r in recalls) / len(recalls))

                drift_facts = [f for f in active if f.updated_at and f.updated_at <= turn]
                if drift_facts:
                    drifts = [temporal_drift_score(mem, f, turn)["drift"] for f in drift_facts]
                    drift_table[name][cp] = sum(drifts) / len(drifts)
                else:
                    drift_table[name][cp] = 0.0

                noise_table[name][cp] = memory_noise_ratio(
                    mem, "best sorting algorithm?", known_values, turn
                )

            # Cascade efficiency
            eff_table["cascading"][cp] = cascade_efficiency(
                backends["cascading"], backends["naive"], active, turn
            )

    if not args.quiet:
        print("\n  RECALL@T")
        print(f"  {'Backend':<12}  " + "  ".join(f"T={c:<4}" for c in checkpoints))
        print("  " + "-" * 52)
        for name in backends:
            vals = "  ".join(f"{recall_table[name].get(c, 0)*100:5.1f}%" for c in checkpoints)
            print(f"  {name:<12}  {vals}")

        print("\n  TOKENS / QUERY")
        print(f"  {'Backend':<12}  " + "  ".join(f"T={c:<4}" for c in checkpoints))
        print("  " + "-" * 52)
        for name in backends:
            vals = "  ".join(f"{tokens_table[name].get(c, 0):6d}" for c in checkpoints)
            print(f"  {name:<12}  {vals}")

        print("\n  TEMPORAL DRIFT")
        print(f"  {'Backend':<12}  " + "  ".join(f"T={c:<4}" for c in checkpoints))
        print("  " + "-" * 52)
        for name in backends:
            vals = "  ".join(f"{drift_table[name].get(c, 0)*100:5.1f}%" for c in checkpoints)
            print(f"  {name:<12}  {vals}")

        print("\n  CASCADE EFFICIENCY (cascading recall-per-token vs naive)")
        vals = "  ".join(f"{eff_table['cascading'].get(c, 1.0):5.2f}x" for c in checkpoints)
        print(f"  {'cascading':<12}  {vals}")

        # Illustrative cost projection: 100K queries/month at $1 per 1M input tokens
        qpm = 100_000
        cost_per_token = 1 / 1_000_000
        final_cp = checkpoints[-1]
        print("\n  PROJECTED COST @ 100K queries/month  ($1 per 1M input tokens, illustrative)")
        print(f"  {'Backend':<12}  {'Tokens/Q':>9}  {'Monthly($)':>11}  {'Recall':>8}")
        print("  " + "-" * 52)
        for name in backends:
            tok = tokens_table[name].get(final_cp, 0)
            cost = tok * qpm * cost_per_token
            rec = recall_table[name].get(final_cp, 0)
            print(f"  {name:<12}  {tok:>9,}  ${cost:>9,.2f}  {rec:>7.1%}")

        print()
        print("  >> Run 'streamlit run dashboard.py' to see full visualisation")
        print("=" * 60)


if __name__ == "__main__":
    main()
