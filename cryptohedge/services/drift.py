"""Data-drift detection and model-degradation monitoring.

Used by the self-diagnostic agent to (a) detect distribution drift between a
reference and a recent window (PSI or KS), (b) track forecasting-error
degradation against a baseline, and (c) blend signals into a single confidence
score in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples (>0.2 typically signals material drift)."""
    reference = np.asarray(reference, float)
    reference = reference[np.isfinite(reference)]
    current = np.asarray(current, float)
    current = current[np.isfinite(current)]
    if len(reference) < 2 or len(current) < 2:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_hist / ref_hist.sum(), 1e-6, None)
    cur_pct = np.clip(cur_hist / cur_hist.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov drift test."""
    reference = np.asarray(reference, float)
    current = np.asarray(current, float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 2 or len(current) < 2:
        return {"statistic": 0.0, "pvalue": 1.0}
    res = stats.ks_2samp(reference, current)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def forecast_errors(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    m = min(len(actual), len(predicted))
    a, p = actual[-m:], predicted[-m:]
    err = a - p
    return {
        "rmse": float(np.sqrt(np.mean(err**2))) if m else 0.0,
        "mae": float(np.mean(np.abs(err))) if m else 0.0,
        "bias": float(np.mean(err)) if m else 0.0,
    }


def degradation_ratio(recent_error: float, baseline_error: float) -> float:
    if baseline_error <= 1e-12:
        return 1.0
    return float(recent_error / baseline_error)


def confidence_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted blend of normalised quality components, each in ``[0, 1]``."""
    total_w = sum(weights.get(k, 0.0) for k in components)
    if total_w <= 0:
        return float(np.mean(list(components.values()))) if components else 0.0
    score = sum(weights.get(k, 0.0) * float(np.clip(v, 0.0, 1.0)) for k, v in components.items())
    return float(np.clip(score / total_w, 0.0, 1.0))
