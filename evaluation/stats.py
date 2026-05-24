"""
evaluation/stats.py — Statistical helpers for multi-seed aggregation and
forgetting-curve analysis.

Provides:
  - mean ± std + 95% confidence intervals over multiple benchmark runs
  - fit_forgetting_curve() — fits Ebbinghaus / exponential decay to recall@T
    data and returns half-life, stability, and R² goodness-of-fit

Closes #6.
"""

import math
from typing import Dict, List, Optional, Tuple


# ── Basic statistics ──────────────────────────────────────────────────────────

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


# ── Forgetting-curve fitting ──────────────────────────────────────────────────

def _r_squared(observed: List[float], predicted: List[float]) -> float:
    """Coefficient of determination R² ∈ (-∞, 1]; 1 = perfect fit."""
    if len(observed) < 2:
        return float("nan")
    mean_obs = _mean(observed)
    ss_tot = sum((y - mean_obs) ** 2 for y in observed)
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(observed, predicted))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("-inf")
    return 1.0 - ss_res / ss_tot


def _fit_exponential(
    turns: List[float],
    recalls: List[float],
) -> Tuple[float, float, float]:
    """
    Fit R(t) = a · exp(−k · t) via log-linear least squares.

    Returns (a, k, r_squared).
    a  — intercept (recall at t=0, ideally ≈ 1.0)
    k  — decay rate (higher = faster forgetting)
    """
    # Filter out zero/negative recalls to avoid log(0)
    valid = [(t, r) for t, r in zip(turns, recalls) if r > 0]
    if len(valid) < 2:
        return (float("nan"), float("nan"), float("nan"))

    xs = [t for t, _ in valid]
    ys = [math.log(r) for _, r in valid]

    # Linear regression on log(R) = log(a) - k*t
    n    = len(xs)
    sx   = sum(xs)
    sy   = sum(ys)
    sxx  = sum(x * x for x in xs)
    sxy  = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return (float("nan"), float("nan"), float("nan"))

    k       = -(n * sxy - sx * sy) / denom
    log_a   = (sy - (-k) * sx) / n  # using -k because slope is -k
    a       = math.exp(log_a)

    predicted = [a * math.exp(-k * t) for t in turns]
    r2 = _r_squared(recalls, predicted)
    return (a, k, r2)


def _fit_ebbinghaus(
    turns: List[float],
    recalls: List[float],
    t_max:   float,
) -> Tuple[float, float]:
    """
    Fit R(t) = exp(−t_norm / (S · sqrt(1 + t_norm))) by grid-searching over S.
    t_norm = t / t_max so that t ∈ [0, 1].

    Returns (S, r_squared).
    S — stability constant (higher = slower forgetting)
    """
    if len(turns) < 2:
        return (float("nan"), float("nan"))

    t_norm_list = [t / max(t_max, 1) for t in turns]

    def _predict(s: float) -> List[float]:
        result = []
        for tn in t_norm_list:
            if tn <= 0:
                result.append(1.0)
            else:
                denom = s * math.sqrt(1.0 + tn)
                result.append(math.exp(-tn / denom))
        return result

    best_s  = 1.0
    best_r2 = float("-inf")

    # Coarse + fine grid search over S ∈ [0.01, 20]
    for s in [i * 0.1 for i in range(1, 201)]:
        predicted = _predict(s)
        r2 = _r_squared(recalls, predicted)
        if not math.isnan(r2) and r2 > best_r2:
            best_r2 = r2
            best_s  = s

    return (best_s, best_r2)


def fit_forgetting_curve(
    checkpoints: List[int],
    recalls:     List[float],
) -> Dict:
    """
    Fit forgetting-curve models to a backend's recall@T time-series and return
    interpretable memory-stability statistics.

    Models fitted
    -------------
    exponential : R(t) = a · exp(−k · t)
                  Classic single-parameter decay (Jost 1897).
    ebbinghaus  : R(t) = exp(−t_norm / (S · √(1 + t_norm)))
                  Two-parameter Ebbinghaus (1885) forgetting curve.

    Parameters
    ----------
    checkpoints : list of turn numbers at which recall was measured
    recalls     : list of recall values ∈ [0, 1] corresponding to each checkpoint

    Returns
    -------
    dict with keys:
      exponential:
        a         — initial recall estimate at t=0
        k         — decay rate (nats per turn)
        half_life — turns until recall halves (ln(2)/k)
        r2        — R² goodness-of-fit
      ebbinghaus:
        stability — S parameter (higher = more stable memory)
        half_life — turns until recall drops to 0.5
        r2        — R² goodness-of-fit
      checkpoints : input turns (echoed for convenience)
      recalls     : input recalls (echoed for convenience)
    """
    if len(checkpoints) != len(recalls) or len(checkpoints) < 2:
        return {"error": "Need at least 2 (checkpoint, recall) pairs."}

    turns   = [float(t) for t in checkpoints]
    t_max   = max(turns)

    # ── Exponential fit ──────────────────────────────────────────────────────
    a, k, r2_exp = _fit_exponential(turns, recalls)
    half_life_exp = math.log(2) / k if (not math.isnan(k) and k > 0) else float("inf")

    # ── Ebbinghaus fit ───────────────────────────────────────────────────────
    S, r2_ebb = _fit_ebbinghaus(turns, recalls, t_max)

    # Half-life for Ebbinghaus: solve exp(-tn / (S * sqrt(1+tn))) = 0.5
    # Numerically: scan t_norm values
    half_life_ebb = float("inf")
    if not math.isnan(S):
        for step in range(1, 10001):
            tn = step / 100.0
            val = math.exp(-tn / (S * math.sqrt(1.0 + tn)))
            if val <= 0.5:
                half_life_ebb = round(tn * t_max, 2)
                break

    return {
        "exponential": {
            "a":         round(a,             4) if not math.isnan(a) else None,
            "k":         round(k,             6) if not math.isnan(k) else None,
            "half_life": round(half_life_exp, 2) if not math.isinf(half_life_exp) else None,
            "r2":        round(r2_exp,        4) if not math.isnan(r2_exp) else None,
        },
        "ebbinghaus": {
            "stability": round(S,             4) if not math.isnan(S) else None,
            "half_life": half_life_ebb if not math.isinf(half_life_ebb) else None,
            "r2":        round(r2_ebb,        4) if not math.isnan(r2_ebb) else None,
        },
        "checkpoints": checkpoints,
        "recalls":     [round(r, 4) for r in recalls],
    }
