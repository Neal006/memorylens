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

Multi-seed benchmark (reports mean +/- std across N personas):
    python main.py --seeds 5
    python main.py --seeds 5 --llm

Decay formula ablation (compare forgetting curve variants):
    python main.py --decay ebbinghaus   (default -- Ebbinghaus 1885)
    python main.py --decay exponential
    python main.py --decay linear
    python main.py --decay default      (original heuristic)

Realistic chunked RAG backend:
    python main.py --backends naive rag_chunked cascading

Forgetting-curve analysis (fit Ebbinghaus + exponential to recall@T data):
    python main.py --fit-curves
    python main.py --seeds 5 --fit-curves

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
                        help="naive | rag | rag_chunked | cascading | summary")
    parser.add_argument("--output",      type=str,  default="results.json")
    parser.add_argument("--log",         action="store_true",
                        help="Save run to experiment_logs/")
    parser.add_argument("--llm",         action="store_true",
                        help="Run real LLM evaluation pass (needs an API key or Ollama)")
    parser.add_argument("--provider",    type=str,  default=None,
                        help="Force a provider: groq | openai | anthropic | openrouter | ollama")
    parser.add_argument("--list-providers", action="store_true",
                        help="Print available providers and exit")
    parser.add_argument("--seeds",       type=int,  default=1,
                        help="Number of persona seeds to run (max 5). >1 reports mean +/- std.")
    parser.add_argument("--decay",       type=str,  default="ebbinghaus",
                        choices=["ebbinghaus", "exponential", "linear", "default"],
                        help="Temporal decay function for CascadingMemory warm tier")
    parser.add_argument("--fit-curves",  action="store_true",
                        help="After benchmarking, fit Ebbinghaus + exponential decay curves "
                             "to recall@T data and report half-life / stability / R²")
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

    multi_seed = args.seeds > 1

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  MemoryLens -- LLM Memory Decay Benchmark")
    print("=" * 65)
    print(f"  Turns       : {args.turns}")
    print(f"  Checkpoints : {sorted(args.checkpoints)}")
    print(f"  Backends    : {args.backends}")
    print(f"  Decay       : {args.decay}")
    if multi_seed:
        print(f"  Seeds       : {args.seeds} (multi-seed -- will report mean +/- std)")
    print(f"  LLM eval    : {'ON  (' + provider.name + ')' if provider else 'OFF (content-only)'}")
    print("=" * 65)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    if multi_seed:
        from evaluation.benchmark import run_benchmark_multi_seed
        aggregated = run_benchmark_multi_seed(
            n_seeds=args.seeds,
            total_turns=args.turns,
            eval_checkpoints=sorted(args.checkpoints),
            backends=args.backends,
            provider=provider,
            decay=args.decay,
            progress=print,
        )
        _print_multi_seed_results(aggregated, args.backends)
        _save(aggregated, args.output)
        if args.log:
            from evaluation.logger import log_run
            path = log_run(aggregated, {
                "total_turns": args.turns,
                "backends":    args.backends,
                "seeds":       args.seeds,
                "decay":       args.decay,
                "provider":    provider.name if provider else None,
            })
            print(f"Experiment logged -> {path}")
    else:
        from evaluation.benchmark import run_benchmark, results_to_display_dict
        raw = run_benchmark(
            total_turns=args.turns,
            eval_checkpoints=sorted(args.checkpoints),
            backends=args.backends,
            provider=provider,
            decay=args.decay,
            progress=print,
        )
        display = results_to_display_dict(raw)
        _print_single_seed_results(display, args.backends)
        _save(display, args.output)
        if args.log:
            from evaluation.logger import log_run
            path = log_run(display, {
                "total_turns": args.turns,
                "backends":    args.backends,
                "decay":       args.decay,
                "provider":    provider.name if provider else None,
            })
            print(f"Experiment logged -> {path}")

    # ── Forgetting-curve analysis ─────────────────────────────────────────────
    if args.fit_curves:
        from evaluation.stats import fit_forgetting_curve
        checkpoints = sorted(args.checkpoints)
        print("\nFORGETTING CURVE FIT  (Ebbinghaus + Exponential)")
        print("-" * 65)
        if multi_seed:
            for name in args.backends:
                if name not in aggregated:
                    continue
                mean_recalls = [stat["mean"] for stat in aggregated[name]["recall"]]
                fit = fit_forgetting_curve(checkpoints, mean_recalls)
                _print_curve_fit(name, fit)
        else:
            for name in args.backends:
                if name not in display:
                    continue
                fit = fit_forgetting_curve(checkpoints, display[name]["recall"])
                _print_curve_fit(name, fit)

    print("Visualise: streamlit run dashboard.py")


# ── Output helpers ────────────────────────────────────────────────────────────


def _print_curve_fit(backend: str, fit: dict) -> None:
    if "error" in fit:
        print(f"  {backend:<14}  {fit['error']}")
        return
    exp = fit["exponential"]
    ebb = fit["ebbinghaus"]
    hl_exp = f"{exp['half_life']:.1f} turns" if exp["half_life"] is not None else "N/A"
    hl_ebb = f"{ebb['half_life']:.1f} turns" if ebb["half_life"] is not None else "N/A"
    r2_exp = f"{exp['r2']:.3f}"              if exp["r2"]        is not None else "N/A"
    r2_ebb = f"{ebb['r2']:.3f}"             if ebb["r2"]        is not None else "N/A"
    stab   = f"{ebb['stability']:.4f}"       if ebb["stability"] is not None else "N/A"
    k_val  = f"{exp['k']:.6f}"              if exp["k"]         is not None else "N/A"
    print(f"  {backend}")
    print(f"    Exponential  k={k_val}  half-life={hl_exp}  R²={r2_exp}")
    print(f"    Ebbinghaus   S={stab}   half-life={hl_ebb}  R²={r2_ebb}")




def _print_single_seed_results(display: dict, backends: list) -> None:
    checkpoints = display["checkpoints"]
    col = "  ".join(f"T={c:3d}" for c in checkpoints)
    sep = "-" * 65

    print(f"\nCONTENT Recall@T")
    print(f"  {'Backend':<14}  {col}")
    print(sep)
    for name in backends:
        if name not in display:
            continue
        vals = "  ".join(f"{v*100:5.1f}%" for v in display[name]["recall"])
        print(f"  {name:<14}  {vals}")

    if display.get("has_llm_eval"):
        print(f"\nLLM Recall@T (answer+judge)")
        print(f"  {'Backend':<14}  {col}")
        print(sep)
        for name in backends:
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
        for name in backends:
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
    print("-" * 65)
    for name in backends:
        if name not in display:
            continue
        tok = display[name]["tokens"][-1]
        print(f"  {name:<14}  {tok:,}")


def _print_multi_seed_results(agg: dict, backends: list) -> None:
    checkpoints = agg["checkpoints"]
    n = agg["n_seeds"]
    sep = "-" * 72

    print(f"\nCONTENT Recall@T  (mean +/- std, n={n} personas)")
    print(f"  {'Backend':<14}  " + "  ".join(f"T={c:3d}" for c in checkpoints))
    print(sep)
    for name in backends:
        if name not in agg:
            continue
        cols = []
        for stat in agg[name]["recall"]:
            cols.append(f"{stat['mean']*100:5.1f}+/-{stat['std']*100:4.1f}%")
        print(f"  {name:<14}  " + "  ".join(cols))

    print(f"\n  Tokens/Query @ T={checkpoints[-1]}  (mean +/- std)")
    print(sep)
    for name in backends:
        if name not in agg:
            continue
        stat = agg[name]["tokens"][-1]
        print(f"  {name:<14}  {stat['mean']:,.0f} +/- {stat['std']:,.0f}")


def _save(data: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nResults saved -> {path}")


if __name__ == "__main__":
    main()
