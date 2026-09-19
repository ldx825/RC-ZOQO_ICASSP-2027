from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .quantizer import UniformAffineQuantizer, grid_aligned


@dataclass
class ZOStep:
    comparison: int
    probe: float
    update: float
    plus_value: float
    minus_value: float
    clipping_occurred: bool


def zoqo_step(objective, x, quantizer: UniformAffineQuantizer, mu, eta, rng) -> tuple[np.ndarray, ZOStep]:
    x = np.asarray(x, dtype=np.float64)
    direction = rng.choice(np.array([-1.0, 1.0]), size=x.size)
    probe = grid_aligned(mu, quantizer.delta)
    update = grid_aligned(eta, quantizer.delta)
    before = quantizer.stats.clipping_count
    plus = quantizer.quantize(x + probe * direction)
    minus = quantizer.quantize(x - probe * direction)
    plus_value, minus_value = objective(plus), objective(minus)
    comparison = int(np.sign(plus_value - minus_value))
    new_x = quantizer.quantize(x - update * comparison * direction)
    return new_x, ZOStep(
        comparison, probe, update, plus_value, minus_value,
        quantizer.stats.clipping_count > before,
    )


def full_precision_sign_step(objective, x, mu, eta, rng) -> tuple[np.ndarray, ZOStep]:
    x = np.asarray(x, dtype=np.float64)
    direction = rng.choice(np.array([-1.0, 1.0]), size=x.size)
    plus_value = objective(x + mu * direction)
    minus_value = objective(x - mu * direction)
    comparison = int(np.sign(plus_value - minus_value))
    return x - eta * comparison * direction, ZOStep(
        comparison, mu, eta, plus_value, minus_value, False
    )

