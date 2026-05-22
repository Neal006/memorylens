"""CI smoke test: verify all modules import correctly."""
import os
import sys

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.facts import BENCHMARK_FACTS
from simulator.conversation import generate_conversation
from memory.naive import NaiveMemory
from memory.rag import RAGMemory
from memory.cascading import CascadingTemporalMemory
from memory.summary import SummaryMemory
from evaluation.metrics import (
    recall_at_t, precision_at_k, temporal_drift_score,
    memory_noise_ratio, cascade_efficiency,
)
from evaluation.benchmark import run_benchmark, results_to_display_dict
from evaluation.logger import log_run, list_runs
from evaluation.llm_judge import judge_answer

print(f"All imports OK | Facts: {len(BENCHMARK_FACTS)}")
