"""
evaluation/stats.py — Statistical helpers for multi-seed aggregation.

Provides mean ± std + 95% confidence intervals over multiple benchmark runs.
"""

import math
from typing import List, Optional, Tuple


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float], mean: Optional[float] = None) -> float:
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _ci95(values: List[float]) -> Tuple[float, float]:
    """Return (lower, upper) 95% confidence interval using t-distribution approx."""
    n = len(values)
    if n < 2:
        m = values[0] if values else 0.0
        return (m, m)
    m = _mean(values)
    s = _std(values, m)
    # t-critical values for common n (fallback to 1.96 for large n)
    t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
               7: 2.447, 8: 2.365, 9: 2.306, 10: 2.228}
    t = t_table.get(n, 1.96)
    margin = t * s / math.sqrt(n)
    return (m - margin, m + margin)


def aggregate_metric(values: List[float]) -> dict:
    """Return a summary dict for a list of scalar metric values."""
    m   = _mean(values)
    s   = _std(values, m)
    lo, hi = _ci95(values)
    return {
        "mean":    round(m,  4),
        "std":     round(s,  4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "n":       len(values),
        "values":  [round(v, 4) for v in values],
    }


def aggregate_checkpoint_series(
    series: List[List[float]],
) -> List[dict]:
    """
    Aggregate a list of equal-length series (one per seed) into per-checkpoint stats.

    Parameters
    ----------
    series : list of lists — series[seed][checkpoint_idx] = metric value

    Returns
    -------
    list of stat dicts, one per checkpoint position
    """
    if not series:
        return []
    n_checkpoints = len(series[0])
    return [
        aggregate_metric([run[i] for run in series])
        for i in range(n_checkpoints)
    ]
