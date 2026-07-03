"""CI smoke test: verify all modules import correctly."""
import os
import sys

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import memorylens
from memorylens.simulator.facts import BENCHMARK_FACTS
from memorylens.simulator.conversation import generate_conversation
from memorylens.simulator.personas import PERSONA_POOL
from memorylens.simulator.scenarios import SCENARIOS, get_scenario, Scenario
from memorylens.memory.naive import NaiveMemory
from memorylens.memory.rag import RAGMemory
from memorylens.memory.rag_chunked import ChunkedRAGMemory
from memorylens.memory.cascading import CascadingTemporalMemory
from memorylens.memory.summary import SummaryMemory
from memorylens.memory.entity import EntityMemory
from memorylens.memory.graph import GraphMemory
from memorylens.memory.decay import get_decay_fn, _REGISTRY as DECAY_REGISTRY
from memorylens.evaluation.metrics import (
    recall_at_t, precision_at_k, temporal_drift_score,
    memory_noise_ratio, contradiction_score, cascade_efficiency,
    llm_recall_at_t, llm_temporal_drift,
)
from memorylens.evaluation.benchmark import run_benchmark, results_to_display_dict, run_benchmark_multi_seed, VALID_BACKENDS
from memorylens.evaluation.stats import aggregate_metric, aggregate_checkpoint_series
from memorylens.evaluation.logger import log_run, list_runs
from memorylens.evaluation.llm_judge import judge_answer
from memorylens.utils.providers import get_provider, list_available, LLMProvider, _REGISTRY as PROVIDER_REGISTRY

# Providers
assert set(PROVIDER_REGISTRY.keys()) == {"groq", "openai", "anthropic", "openrouter", "ollama"}, (
    f"Provider registry mismatch: {set(PROVIDER_REGISTRY.keys())}"
)
p = get_provider(None)
assert p is None or isinstance(p, LLMProvider), f"Unexpected get_provider result: {p!r}"

# Decay registry
assert set(DECAY_REGISTRY.keys()) == {"default", "linear", "exponential", "ebbinghaus"}, (
    f"Decay registry mismatch: {set(DECAY_REGISTRY.keys())}"
)
for name in DECAY_REGISTRY:
    fn = get_decay_fn(name)
    v = fn(5, 100)
    assert 0.0 <= v <= 1.0, f"{name} decay out of [0,1] range: {v}"

# Personas
assert len(PERSONA_POOL) >= 5, f"Expected at least 5 personas, got {len(PERSONA_POOL)}"

# Backend registry
for backend in ("rag_chunked", "entity", "graph", "faiss"):
    assert backend in VALID_BACKENDS, f"'{backend}' missing from VALID_BACKENDS"

# Scenario registry
assert set(SCENARIOS.keys()) == {"default", "edtech", "support", "medical"}, (
    f"Scenario registry mismatch: {set(SCENARIOS.keys())}"
)
for s in SCENARIOS.values():
    s.validate()

# Package metadata
assert memorylens.__version__

print(
    f"All imports OK | "
    f"v{memorylens.__version__} | "
    f"Facts: {len(BENCHMARK_FACTS)} | "
    f"Backends: {VALID_BACKENDS} | "
    f"Scenarios: {list(SCENARIOS.keys())} | "
    f"Providers: {list(PROVIDER_REGISTRY.keys())} | "
    f"Personas: {len(PERSONA_POOL)}"
)
