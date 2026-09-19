from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class QuantizationStats:
    rounding_count: int = 0
    clipping_count: int = 0
    total_quantized_values: int = 0
    rounding_error_sum: float = 0.0
    clipping_distortion_sum: float = 0.0

    @property
    def clipping_ratio(self) -> float:
        return self.clipping_count / max(self.total_quantized_values, 1)

    @property
    def mean_rounding_error(self) -> float:
        return self.rounding_error_sum / max(self.total_quantized_values, 1)

    @property
    def mean_clipping_distortion(self) -> float:
        return self.clipping_distortion_sum / max(self.total_quantized_values, 1)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rounding_count": self.rounding_count,
            "clipping_count": self.clipping_count,
            "total_quantized_values": self.total_quantized_values,
            "clipping_ratio": self.clipping_ratio,
            "mean_rounding_error": self.mean_rounding_error,
            "mean_clipping_distortion": self.mean_clipping_distortion,
        }


class UniformAffineQuantizer:
    """Uniform affine quantizer with explicit rounding/clipping accounting."""

    def __init__(self, bits: int, rmin: float, rmax: float):
        if bits < 1 or not rmax > rmin:
            raise ValueError("bits >= 1 and rmax > rmin are required")
        self.bits = int(bits)
        self.rmin = float(rmin)
        self.rmax = float(rmax)
        self.levels = 2**self.bits - 1
        self.delta = (self.rmax - self.rmin) / self.levels
        self.stats = QuantizationStats()

    def reset_stats(self) -> None:
        self.stats = QuantizationStats()

    def quantize(self, x: np.ndarray | float, track: bool = True) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        raw_code = (values - self.rmin) / self.delta
        rounded_code = np.rint(raw_code)
        clipped_code = np.clip(rounded_code, 0, self.levels)
        result = self.rmin + self.delta * clipped_code
        if track:
            clipped_input = np.clip(values, self.rmin, self.rmax)
            clipped = (values < self.rmin) | (values > self.rmax)
            rounding = np.abs(result - clipped_input) > 1e-14
            self.stats.total_quantized_values += values.size
            self.stats.clipping_count += int(np.count_nonzero(clipped))
            self.stats.rounding_count += int(np.count_nonzero(rounding))
            self.stats.rounding_error_sum += float(np.abs(result - clipped_input).sum())
            self.stats.clipping_distortion_sum += float(np.abs(values - clipped_input).sum())
        return result


def grid_aligned(value: float, delta: float) -> float:
    return max(int(np.floor(value / delta)), 1) * delta


def quantize_blockwise(
    x: np.ndarray, bits: int, group_size: int | None, padding: float = 0.0
) -> tuple[np.ndarray, list[UniformAffineQuantizer]]:
    x = np.asarray(x, dtype=np.float64)
    size = x.size if group_size is None else int(group_size)
    out = np.empty_like(x)
    quantizers: list[UniformAffineQuantizer] = []
    for start in range(0, x.size, size):
        block = x[start : start + size]
        lo, hi = float(block.min() - padding), float(block.max() + padding)
        if hi <= lo:
            hi = lo + np.finfo(np.float64).eps
        quantizer = UniformAffineQuantizer(bits, lo, hi)
        out[start : start + size] = quantizer.quantize(block)
        quantizers.append(quantizer)
    return out, quantizers

