"""
MemoryLens CLI

Usage (content-only, no API key needed):
    python main.py

Usage (full LLM evaluation, auto-detects available provider):
    python main.py --llm

Usage (force a specific provider):
    python main.py --llm --provider openai
    python main.py --llm --provider anthropic
    python main.py --llm --provider groq
    python main.py --llm --provider openrouter
    python main.py --llm --provider ollama

Other options:
    python main.py --turns 50 --backends naive rag --log
    python main.py --list-providers
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MemoryLens: End-to-end LLM Memory Decay Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--turns",       type=int,  default=100)
    parser.add_argument("--checkpoints", nargs="+", type=int,
                        default=[10, 25, 50, 75, 100])
    parser.add_argument("--backends",    nargs="+",
                        default=["naive", "rag", "cascading"],
                        help="naive | rag | cascading | summary")
    parser.add_argument("--output",      type=str,  default="results.json")
    parser.add_argument("--log",         action="store_true",
                        help="Save run to experiment_logs/")
    parser.add_argument("--llm",         action="store_true",
                        help="Run real LLM evaluation pass (needs an API key or Ollama)")
    parser.add_argument("--provider",    type=str,  default=None,
                        help="Force a provider: groq | openai | anthropic | openrouter | ollama")
    parser.add_argument("--list-providers", action="store_true",
                        help="Print available providers and exit")
    args = parser.parse_args()

    # ── List providers ────────────────────────────────────────────────────────
    if args.list_providers:
        from utils.providers import list_available, _REGISTRY
        available = list_available()
        print("\nProvider status:")
        for name in _REGISTRY:
            status = "available" if name in available else "not available"
            print(f"  {name:<15} {status}")
        print()
        sys.exit(0)

    # ── Resolve LLM provider ─────────────────────────────────────────────────
    provider = None
    if args.llm:
        from utils.providers import get_provider
        try:
            provider = get_provider(args.provider)
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        if provider is None:
            print(
                "ERROR: --llm requested but no provider is available.\n"
                "  Set one of: GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "OPENROUTER_API_KEY\n"
                "  or start Ollama locally.\n"
                "  Run --list-providers to see status."
            )
            sys.exit(1)

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  MemoryLens — LLM Memory Decay Benchmark")
    print("=" * 60)
    print(f"  Turns       : {args.turns}")
    print(f"  Checkpoints : {sorted(args.checkpoints)}")
    print(f"  Backends    : {args.backends}")
    print(f"  LLM eval    : {'ON  (' + provider.name + ')' if provider else 'OFF (content-only)'}")
    print("=" * 60)

    from evaluation.benchmark import run_benchmark, results_to_display_dict

    raw = run_benchmark(
        total_turns=args.turns,
        eval_checkpoints=sorted(args.checkpoints),
        backends=args.backends,
        provider=provider,
        progress=print,
    )

    display = results_to_display_dict(raw)
    checkpoints = display["checkpoints"]

    # ── Results table ─────────────────────────────────────────────────────────
    col = "  ".join(f"T={c:3d}" for c in checkpoints)
    sep = "-" * 60

    print(f"\n{'CONTENT Recall@T':}")
    print(f"  {'Backend':<14}  {col}")
    print(sep)
    for name in args.backends:
        if name not in display:
            continue
        vals = "  ".join(f"{v*100:5.1f}%" for v in display[name]["recall"])
        print(f"  {name:<14}  {vals}")

    if display.get("has_llm_eval"):
        print(f"\n{'LLM Recall@T (ground truth)':}")
        print(f"  {'Backend':<14}  {col}")
        print(sep)
        for name in args.backends:
            if name not in display:
                continue
            llm_vals = display[name].get("llm_recall", [])
            vals = "  ".join(
                f"{v*100:5.1f}%" if v is not None else "  N/A "
                for v in llm_vals
            )
            print(f"  {name:<14}  {vals}")

        print(f"\n  Gap = Content Recall - LLM Recall")
        print(f"  {'Backend':<14}  {col}")
        print(sep)
        for name in args.backends:
            if name not in display:
                continue
            content = display[name]["recall"]
            llm     = display[name].get("llm_recall", [None]*len(content))
            vals = "  ".join(
                f"{(c - l)*100:+5.1f}%" if l is not None else "  N/A "
                for c, l in zip(content, llm)
            )
            print(f"  {name:<14}  {vals}")

    print(f"\n  Tokens/Query @ T={checkpoints[-1]}")
    print(sep)
    for name in args.backends:
        if name not in display:
            continue
        tok = display[name]["tokens"][-1]
        print(f"  {name:<14}  {tok:,}")

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(args.output, "w") as fh:
        json.dump(display, fh, indent=2)
    print(f"\nResults saved -> {args.output}")

    if args.log:
        from evaluation.logger import log_run
        path = log_run(display, {
            "total_turns": args.turns,
            "backends":    args.backends,
            "provider":    provider.name if provider else None,
        })
        print(f"Experiment logged -> {path}")

    print("Visualise: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
