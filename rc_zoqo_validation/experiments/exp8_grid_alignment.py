#!/usr/bin/env python3
"""Experiment 8: the exact isotropic-quadratic floor is grid distance."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import CSV_DIR, LOG_DIR, ROOT, ensure_output_dirs, load_config, markdown_table, save_figure, write_csv, write_json


def nearest_grid(x: np.ndarray, delta: float) -> np.ndarray:
    """Nearest point on the infinite origin-aligned Cartesian grid."""
    return delta * np.rint(x / delta)


def main() -> None:
    cfg = load_config("addendum.yaml")
    ensure_output_dirs()
    rng = np.random.default_rng(cfg["seed"])
    lo, hi = cfg["range"]
    lam = float(cfg["lambda"])
    deltas = {b: (hi - lo) / (2**b - 1) for b in cfg["bits"]}
    raw_rows, table_rows = [], []

    for dimension in cfg["dimensions"]:
        for bit, delta in deltas.items():
            # Integer cell indices are immaterial; randomize them to test nearest-grid code.
            cells = rng.integers(-4, 5, size=(cfg["phase_samples"], dimension))
            phase = rng.uniform(-delta / 2, delta / 2, size=(cfg["phase_samples"], dimension))
            x_star = delta * cells + phase
            x_bar = nearest_grid(x_star, delta)
            distances = np.linalg.norm(x_bar - x_star, axis=1)
            gradient_floors = lam * distances
            for sample, (distance, floor) in enumerate(zip(distances, gradient_floors)):
                raw_rows.append({
                    "dimension": dimension, "bit": bit, "Delta": delta, "sample": sample,
                    "delta_grid": distance, "exact_gradient_floor": floor,
                    "normalized_phase": distance / delta,
                })
            empirical_delta_sq = float(np.mean(distances**2))
            theoretical_delta_sq = dimension * delta**2 / 12
            empirical_rms = float(np.sqrt(np.mean(gradient_floors**2)))
            theoretical_rms = lam * np.sqrt(dimension / 12) * delta
            table_rows.append({
                "dimension": dimension, "bit": bit, "Delta": delta,
                "samples": cfg["phase_samples"],
                "mean_delta_grid_squared": empirical_delta_sq,
                "theory_delta_grid_squared": theoretical_delta_sq,
                "second_moment_relative_error": abs(empirical_delta_sq / theoretical_delta_sq - 1),
                "rms_exact_gradient_floor": empirical_rms,
                "theory_rms_gradient_floor": theoretical_rms,
                "rms_relative_error": abs(empirical_rms / theoretical_rms - 1),
            })

    write_csv(CSV_DIR / "exp8_grid_alignment_raw.csv", raw_rows)
    write_csv(CSV_DIR / "table_D_grid_alignment.csv", table_rows)
    md_columns = list(table_rows[0])
    (ROOT / "outputs" / "table_D_grid_alignment.md").write_text(
        markdown_table(table_rows, md_columns), encoding="utf-8"
    )

    # Fixed absolute targets: the nearest-grid error may rise when Delta becomes finer.
    fixed_rows = []
    fixed_scalars = np.array([0.4, np.sqrt(2) / 10, np.pi / 20, 0.271828, -0.337, 0.499])
    for instance, value in enumerate(fixed_scalars[: cfg["fixed_instances"]]):
        dimension = 16
        x_star = np.full(dimension, value)
        for bit, delta in deltas.items():
            distance = float(np.linalg.norm(nearest_grid(x_star, delta) - x_star))
            fixed_rows.append({
                "instance": instance, "x_star_scalar": value, "dimension": dimension,
                "bit": bit, "Delta": delta, "exact_gradient_floor": lam * distance,
            })
    write_csv(CSV_DIR / "exp8_fixed_instances.csv", fixed_rows)

    fit_rows = []
    for dimension in cfg["dimensions"]:
        subset = [r for r in table_rows if r["dimension"] == dimension]
        x = np.log([r["Delta"] for r in subset])
        y = np.log([r["rms_exact_gradient_floor"] for r in subset])
        slope, intercept = np.polyfit(x, y, 1)
        fit_rows.append({"dimension": dimension, "log_log_slope": float(slope), "intercept": float(intercept)})

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    for dimension in cfg["dimensions"]:
        subset = sorted((r for r in table_rows if r["dimension"] == dimension), key=lambda r: r["Delta"])
        axes[0].loglog([r["Delta"] for r in subset], [r["rms_exact_gradient_floor"] for r in subset], "o-", label=f"d={dimension}")
        axes[0].loglog([r["Delta"] for r in subset], [r["theory_rms_gradient_floor"] for r in subset], "--", alpha=0.55)
    for instance in range(cfg["fixed_instances"]):
        subset = sorted((r for r in fixed_rows if r["instance"] == instance), key=lambda r: r["Delta"])
        axes[1].plot([r["Delta"] for r in subset], [r["exact_gradient_floor"] for r in subset], "o-", ms=3, label=f"x* {instance + 1}")
    axes[0].set(xlabel=r"Resolution $\Delta$", ylabel="RMS exact gradient floor", title="Phase-averaged RMS scaling")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(xlabel=r"Resolution $\Delta$", ylabel="Exact gradient floor", title="Fixed targets need not be monotone")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes: ax.grid(alpha=0.25)
    save_figure(fig, "fig8_grid_alignment")
    plt.close(fig)

    max_second_error = max(r["second_moment_relative_error"] for r in table_rows)
    max_rms_error = max(r["rms_relative_error"] for r in table_rows)
    slopes = [r["log_log_slope"] for r in fit_rows]
    passed = max_second_error < 0.08 and max_rms_error < 0.04 and all(abs(s - 1) < 0.04 for s in slopes)
    write_json(LOG_DIR / "exp8_status.json", {
        "passed": passed, "phase_samples_per_configuration": cfg["phase_samples"],
        "max_second_moment_relative_error": max_second_error,
        "max_rms_relative_error": max_rms_error, "dimensionwise_log_log_fits": fit_rows,
    })
    print(f"Experiment 8: passed={passed}, max E[delta_grid^2] rel.err={max_second_error:.4%}, max RMS rel.err={max_rms_error:.4%}")


if __name__ == "__main__":
    main()
