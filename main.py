"""
MemoryLens CLI — run the benchmark from the terminal.

Usage:
    python main.py                            # default settings
    python main.py --turns 50 --backends rag cascading
    python main.py --output my_results.json
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MemoryLens: LLM Memory Decay Evaluation Framework"
    )
    parser.add_argument("--turns",       type=int,   default=100)
    parser.add_argument("--checkpoints", nargs="+",  type=int, default=[10, 25, 50, 75, 100])
    parser.add_argument("--backends",    nargs="+",  default=["naive", "rag", "cascading"])
    parser.add_argument("--output",      type=str,   default="results.json")
    parser.add_argument("--log",         action="store_true", help="Save run to experiment_logs/")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    from evaluation.benchmark import run_benchmark, results_to_display_dict

    print("=" * 55)
    print("  MemoryLens — LLM Memory Decay Benchmark")
    print("=" * 55)
    print(f"  Turns      : {args.turns}")
    print(f"  Checkpoints: {args.checkpoints}")
    print(f"  Backends   : {args.backends}")
    print("=" * 55)

    raw = run_benchmark(
        total_turns=args.turns,
        eval_checkpoints=sorted(args.checkpoints),
        backends=args.backends,
        progress=print,
    )

    display = results_to_display_dict(raw)
    checkpoints = display["checkpoints"]

    print("\n" + "=" * 55)
    print("  RESULTS — Recall@T")
    print("  {:20s}  {}".format("Backend", "  ".join(f"T={c:3d}" for c in checkpoints)))
    print("-" * 55)
    for name in args.backends:
        if name not in display:
            continue
        vals = "  ".join(f"{v*100:5.1f}%" for v in display[name]["recall"])
        print(f"  {name:20s}  {vals}")

    print("\n  RESULTS — Avg Tokens/Query")
    print("  {:20s}  {}".format("Backend", "  ".join(f"T={c:3d}" for c in checkpoints)))
    print("-" * 55)
    for name in args.backends:
        if name not in display:
            continue
        vals = "  ".join(f"{v:6d}" for v in display[name]["tokens"])
        print(f"  {name:20s}  {vals}")

    with open(args.output, "w") as fh:
        json.dump(display, fh, indent=2)
    print(f"\nResults saved → {args.output}")

    if args.log:
        from evaluation.logger import log_run
        path = log_run(display, {"total_turns": args.turns, "backends": args.backends})
        print(f"Experiment logged → {path}")

    print("Visualise: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
