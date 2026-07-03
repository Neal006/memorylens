"""MemoryLens — a benchmark for measuring LLM memory decay across conversation turns."""

__version__ = "0.4.0"

from memorylens.memory.base import BaseMemory
from memorylens.simulator.facts import Fact
from memorylens.evaluation.benchmark import (
    run_benchmark,
    run_benchmark_multi_seed,
    results_to_display_dict,
    VALID_BACKENDS,
)

__all__ = [
    "__version__",
    "BaseMemory",
    "Fact",
    "run_benchmark",
    "run_benchmark_multi_seed",
    "results_to_display_dict",
    "VALID_BACKENDS",
]
