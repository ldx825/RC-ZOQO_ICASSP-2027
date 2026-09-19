from __future__ import annotations

import numpy as np


def tail_metrics(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.var(values)), float(np.max(values) - np.min(values))


def log_log_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    lx, ly = np.log(np.asarray(x)), np.log(np.maximum(np.asarray(y), 1e-15))
    slope, intercept = np.polyfit(lx, ly, 1)
    predicted = slope * lx + intercept
    total = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - np.sum((ly - predicted) ** 2) / total if total > 0 else 1.0
    return float(slope), float(intercept), float(r2)

