#!/usr/bin/env python3
"""Experiment 9: concentration-aware audit lower bound and false-stop bound."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import CSV_DIR, LOG_DIR, ROOT, ensure_output_dirs, load_config, save_figure, write_csv, write_json


def refined_bound(s: float, nu: float) -> float:
    if s <= 0.5:
        return float("nan")
    return (1.0 - 1.0 / (4.0 * s * s)) ** 2 / (3.0 - 2.0 * nu)


def empirical_success(q: int, s: float, trials: int, rng: np.random.Generator) -> float:
    # For g with equal active components, g^T r is represented exactly by a Binomial count.
    counts = rng.binomial(q, 0.5, size=trials)
    return float(np.mean(np.abs(2 * counts - q) >= np.sqrt(q) / (2.0 * s)))


def main() -> None:
    cfg = load_config("audit.yaml")
    ensure_output_dirs()
    rng = np.random.default_rng(20270910)
    a, gamma, L = cfg["a"], cfg["gamma"], cfg["L"]
    rows, false_stop_rows = [], []
    for q in cfg["q"]:
        for ratio in cfg["gradient_ratios"]:
            s = float(ratio)
            nu = 1.0 / q
            p_emp = empirical_success(q, s, cfg["audits"], rng)
            p_ref = refined_bound(s, nu)
            p_old = 3.0 / 16.0 if s >= 1.0 else float("nan")
            # Geometric stopping is the independent-direction audit model.
            if s > 0.5 and p_emp > 0:
                forward_evals = 2.0 * rng.geometric(p_emp, size=cfg["audits"])
                mean_forward = float(forward_evals.mean())
                median_forward = float(np.quantile(forward_evals, 0.5))
                p90_forward = float(np.quantile(forward_evals, 0.9))
                p95_forward = float(np.quantile(forward_evals, 0.95))
                empirical_bound_violation = bool(p_emp + 3.0 / np.sqrt(cfg["audits"]) < p_ref)
            else:
                mean_forward = median_forward = p90_forward = p95_forward = float("nan")
                empirical_bound_violation = False
            rows.append({
                "q": q, "gradient_ratio_s": s, "nu": nu, "effective_dimension": q,
                "gradient_norm": s * a * q * (L + 2.0 * gamma),
                "tau": a * q * (L + 2.0 * gamma),
                "a": a, "L": L, "gamma": gamma,
                "empirical_success_probability": p_emp,
                "old_universal_lower_bound": p_old,
                "refined_lower_bound": p_ref,
                "mean_forward_evaluations": mean_forward,
                "median_forward_evaluations": median_forward,
                "p90_forward_evaluations": p90_forward,
                "p95_forward_evaluations": p95_forward,
                "old_worst_case_forward_bound": 32.0 / 3.0,
                "refined_expected_forward_target": 2.0 / p_ref if np.isfinite(p_ref) and p_ref > 0 else float("nan"),
                "empirical_bound_violation_with_3se": int(empirical_bound_violation),
            })
            if s > 0.5:
                # Fresh complete audits: all K directions fail. Sampling is vectorized via Binomial.
                for K in cfg["K"]:
                    success_counts = rng.binomial(K, p_emp, size=cfg["audits"])
                    empirical_false = float(np.mean(success_counts == 0))
                    false_stop_rows.append({
                        "q": q, "gradient_ratio_s": s, "K": K,
                        "empirical_success_probability": p_emp,
                        "empirical_false_stop_probability": empirical_false,
                        "refined_false_stop_bound": (1.0 - p_ref) ** K,
                        "old_false_stop_bound": (13.0 / 16.0) ** K,
                    })

    write_csv(CSV_DIR / "table_E_refined_audit.csv", rows)
    write_csv(CSV_DIR / "exp9_refined_false_stop.csv", false_stop_rows)
    finite = [r for r in rows if np.isfinite(r["refined_lower_bound"])]
    violations = [r for r in finite if r["empirical_bound_violation_with_3se"]]
    # Probability panel: all q at each s, with refined lower bound for comparison.
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for q in cfg["q"]:
        subset = sorted((r for r in rows if r["q"] == q and r["gradient_ratio_s"] > 0.5), key=lambda r: r["gradient_ratio_s"])
        ax.plot([r["gradient_ratio_s"] for r in subset], [r["empirical_success_probability"] for r in subset], "o-", ms=3, label=f"empirical q={q}")
        ax.plot([r["gradient_ratio_s"] for r in subset], [r["refined_lower_bound"] for r in subset], "--", lw=1, alpha=0.7)
    ax.axhline(3.0 / 16.0, color="black", ls=":", lw=1, label="old 3/16 (s≥1)")
    ax.set(xlabel=r"$s=||g||/\tau$", ylabel="Audit success probability", title="Concentration-aware audit bound")
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=7, ncol=2); save_figure(fig, "fig9_refined_audit_bound"); plt.close(fig)
    # False-stop panel at the informative s=1.25 slice, retaining q dependence.
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for q in cfg["q"]:
        subset = [r for r in false_stop_rows if r["q"] == q and r["gradient_ratio_s"] == 1.25]
        subset.sort(key=lambda r: r["K"])
        ax.semilogy([r["K"] for r in subset], [r["empirical_false_stop_probability"] for r in subset], "o-", ms=3, label=f"empirical q={q}")
    ref = [r for r in false_stop_rows if r["q"] == 128 and r["gradient_ratio_s"] == 1.25]
    ref.sort(key=lambda r: r["K"])
    ax.semilogy([r["K"] for r in ref], [r["refined_false_stop_bound"] for r in ref], "k--", label="refined bound")
    ax.semilogy([r["K"] for r in ref], [r["old_false_stop_bound"] for r in ref], "k:", label="old bound")
    ax.set(xlabel="K", ylabel="False-stop probability", title="Refined false-stop comparison (s=1.25)")
    ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False, fontsize=8); save_figure(fig, "fig10_refined_false_stop"); plt.close(fig)
    write_json(LOG_DIR / "exp9_status.json", {
        "passed": not violations, "configurations": len(rows), "false_stop_rows": len(false_stop_rows),
        "violations_with_3se": len(violations), "violation_examples": violations[:5],
        "old_expected_forward_bound": 32.0 / 3.0,
    })
    print(f"Experiment 9: passed={not violations}, refined-bound violations with 3SE={len(violations)}")


if __name__ == "__main__":
    main()
