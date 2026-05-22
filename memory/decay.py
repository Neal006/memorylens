"""
memory/decay.py — Temporal decay functions for the Cascading memory tier.

Three scientifically-grounded alternatives + the original default:

  default     — original heuristic: linear with 0.6 slope + 0.2 floor
  linear      — pure linear decay, no floor (Wickelgren 1972)
  exponential — exponential decay e^{-k*t} (Jost 1897, k=1.0)
  ebbinghaus  — Ebbinghaus (1885) forgetting curve: e^{-t / sqrt(1+t)}

All functions take (age: int, window: int) → float in [0, 1].
  age    = current_turn - message_turn   (0 = brand-new message)
  window = total conversation turns so far
"""

import math


def decay_default(age: int, window: int) -> float:
    """Original heuristic — linear with 0.6 slope, hard floor at 0.2."""
    return max(0.2, 1.0 - age / max(1, window) * 0.6)


def decay_linear(age: int, window: int) -> float:
    """Pure linear decay (Wickelgren 1972 power-law approximation baseline)."""
    return max(0.0, 1.0 - age / max(1, window))


def decay_exponential(age: int, window: int, k: float = 1.0) -> float:
    """Exponential decay — Jost (1897): R(t) = e^{-k*t/window}."""
    t = age / max(1, window)
    return math.exp(-k * t)


def decay_ebbinghaus(age: int, window: int, stability: float = 1.0) -> float:
    """
    Ebbinghaus (1885) forgetting curve: R(t) = e^{-t / (stability * sqrt(1+t))}.

    The stability parameter maps to the concept of memory consolidation:
    higher stability = slower forgetting.  Default=1.0 matches the original
    Ebbinghaus data on nonsense syllables; meaningful content should use ~2–3.
    """
    t = age / max(1, window)
    if t <= 0:
        return 1.0
    denominator = stability * math.sqrt(1.0 + t)
    return math.exp(-t / denominator)


_REGISTRY = {
    "default":     decay_default,
    "linear":      decay_linear,
    "exponential": decay_exponential,
    "ebbinghaus":  decay_ebbinghaus,
}


def get_decay_fn(name: str):
    """Return the decay function for `name`. Raises ValueError on unknown names."""
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown decay function '{name}'. "
            f"Choose from: {list(_REGISTRY)}"
        )
    return fn
