# RC-ZOQO Theory Validation Addendum Report

## Experiment 8 — Grid-alignment-aware floor

**SUPPORTED.** For every `d ∈ {1,16,64,256}` and all five resolutions, 1,000 independent uniform phases were sampled. The empirical second moment obeys `E[δ_grid²] = dΔ²/12` within 4.81% worst-case relative error. The RMS exact gradient floor follows `λ√(d/12)Δ` within 2.37% worst-case relative error. Dimension-wise log-log slopes are 0.9987, 0.9997, 0.9993, and 1.0000 for `d=1,16,64,256`, respectively.

The fixed-instance sweep is retained in `exp8_fixed_instances.csv` and `fig8_grid_alignment.pdf`. Several individual curves are non-monotone, confirming that strict per-instance monotonicity is not predicted; only phase-averaged RMS linearity is.

## Experiment 9 — Concentration-aware audit bound

**SUPPORTED.** All 32 existing `(q,s)` audit configurations were re-evaluated. The implementation uses `ν=||g||₄⁴/||g||₂⁴=1/q` and `effective_dimension=q` for equal active components. For every configuration with `s>0.5`, empirical success probability was not below the refined bound (zero violations after a conservative 3-standard-error check). The old `3/16` lower bound is retained for `s≥1`.

At `q=128`, for example, empirical/refined success probabilities are: `s=0.75: 0.5411/0.1034`, `s=1.0: 0.6570/0.1885`, `s=1.25: 0.6488/0.2364`, `s=2.0: 0.7892/0.2945`, and `s=4.0: 0.9275/0.3247`.

Complete-audit false-stop Monte Carlo results for every `K ∈ {5,10,12,15,23,34}` are in `exp9_refined_false_stop.csv`, with both `(1-p_refined)^K` and `(13/16)^K`. Expected forward evaluations are reported against both `2/p_refined` and the old worst-case `32/3` in `table_E_refined_audit.csv`.

## Reproducibility

```bash
cd /root/autodl-tmp/EviZO-VP/RC-ZOQO/rc_zoqo_validation
python experiments/exp8_grid_alignment.py
python experiments/exp9_refined_audit.py
```

Outputs:

- `outputs/csv/table_D_grid_alignment.csv`
- `outputs/csv/table_E_refined_audit.csv`
- `outputs/figures/fig8_grid_alignment.pdf/.png`
- `outputs/figures/fig9_refined_audit_bound.pdf/.png`
- `outputs/figures/fig10_refined_false_stop.pdf/.png`
- raw phase and false-stop CSVs, plus status JSON logs
