from __future__ import annotations

import numpy as np

from .audit import run_audit
from .quantizer import UniformAffineQuantizer, grid_aligned


def clipping_aware_probe(x: np.ndarray, requested: float, quantizer: UniformAffineQuantizer) -> float:
    margin = float(np.min(np.minimum(x - quantizer.rmin, quantizer.rmax - x)))
    safe = max(0.0, margin)
    if safe < quantizer.delta:
        return 0.0
    return min(grid_aligned(requested, quantizer.delta), np.floor(safe / quantizer.delta) * quantizer.delta)


def rc_step(objective, x, quantizer, mu, eta, rng, clipping_aware=True):
    direction = rng.choice(np.array([-1.0, 1.0]), size=x.size)
    probe = clipping_aware_probe(x, mu, quantizer) if clipping_aware else grid_aligned(mu, quantizer.delta)
    if probe == 0:
        return x.copy(), {"resolved": False, "clipped": False, "sign": 0}
    before = quantizer.stats.clipping_count
    qp = quantizer.quantize(x + probe * direction)
    qm = quantizer.quantize(x - probe * direction)
    fp, fm = objective(qp), objective(qm)
    sign = int(np.sign(fp - fm))
    update = grid_aligned(eta, quantizer.delta)
    result = quantizer.quantize(x - update * sign * direction)
    return result, {
        "resolved": sign != 0,
        "clipped": quantizer.stats.clipping_count > before,
        "sign": sign,
    }


def resolution_audit(objective, x, quantizer, gamma, k, rng):
    return run_audit(objective, x, quantizer.delta, gamma, k, rng)

