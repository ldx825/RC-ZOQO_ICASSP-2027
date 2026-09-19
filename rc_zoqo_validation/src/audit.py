from __future__ import annotations

import numpy as np


def sufficient_decrease(f0: float, f_plus: float, f_minus: float, a: float, q: int, gamma: float) -> bool:
    return min(f_plus, f_minus) <= f0 - gamma * a * a * q


def audit_direction(objective, x: np.ndarray, a: float, gamma: float, rng) -> tuple[bool, np.ndarray, float]:
    direction = rng.choice(np.array([-1.0, 1.0]), size=x.size)
    f0 = objective.value(x)  # cached/offline center: no forward query
    f_plus, f_minus = objective(x + a * direction), objective(x - a * direction)
    success = sufficient_decrease(f0, f_plus, f_minus, a, x.size, gamma)
    if f_plus <= f_minus:
        return success, x + a * direction, f_plus
    return success, x - a * direction, f_minus


def run_audit(objective, x: np.ndarray, a: float, gamma: float, k: int, rng):
    for index in range(1, k + 1):
        success, candidate, value = audit_direction(objective, x, a, gamma, rng)
        if success:
            return True, index, candidate, value
    return False, k, x.copy(), objective.value(x)


def audit_threshold(a: float, q: int, smoothness: float, gamma: float) -> float:
    return a * q * (smoothness + 2.0 * gamma)

