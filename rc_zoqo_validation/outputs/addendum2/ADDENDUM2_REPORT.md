# RC-ZOQO Theory Validation Addendum II Report

## 1. Curvature-Weighted Floor

**SUPPORTED.** Experiment 10 used `d ∈ {16,64,256}`, `Δ ∈ {0.005,0.01,0.03,0.1,0.3}`, 5,000 uniform phase samples per configuration, and isotropic plus two log-spaced Hessian spectra. The experiment evaluates nearest-grid quantization directly; no optimizer or hidden gradient is used.

## 2. Exact Moment Agreement

**SUPPORTED.** The empirical phase averages agree with
`E[f(x̄)-f*] = Δ² tr(H)/24` and
`E||∇f(x̄)||² = Δ² tr(H²)/12`.
The median relative error across all configurations is 0.20%; maximum objective-gap error is 1.09%, and maximum gradient-squared error is 1.85%. RMS-gradient log-log slopes at `d=64` are 0.99994 (isotropic), 1.00019 (`0.1..10`), and 1.00041 (`0.01..100`).

## 3. Support-Size Audit Bound

**SUPPORTED.** Experiment 11 evaluated 90 valid `(q,k,m)` configurations for `q ∈ {128,512}`, `k ∈ {1,4,16,64,q}`, and all allowed `m`, with 100,000 support/sign samples per configuration. The exact quadratic objective at `x=0` was used, so success is tested by the exact condition `|gᵀr| ≥ am(γ+L/2)`. No configuration violated the refined lower bound under the conservative 3σ Monte Carlo check.

## 4. Diffuse Gradient Regime

**SUPPORTED.** For `k=q`, `ν=1/q` and the measured lower-bound values agree with the closed form `(1-θ_m)² m/(3m-2)`. Both `m=1` and `m=q` remain non-vacuous under the configured signal scale. The empirical probabilities and closed-form comparisons are in `exp11_diffuse_corollary.csv`.

## 5. Concentrated Gradient Regime

**SUPPORTED.** For `k=1`, the empirical success probability follows the support-hit probability `m/q` under the strong-signal construction. For example, at `q=128`, probabilities for `m={1,2,4,8,16,32,64,128}` are approximately `{0.0079,0.0152,0.0315,0.0638,0.1222,0.2514,0.5018,1.0}`.

## 6. Support-Size Phase Transition

**SUPPORTED.** The predicted monotonicity of `A_m` matched the numerical monotonicity for every tested `(q,k)`: `A_m` decreases when `ν<3/(q+2)` and increases when `ν>3/(q+2)`. The complete check is in `exp11_phase_transition.csv`.

## 7. Counterexamples / Violations

No theorem-bound violations were observed. The implementation uses hypergeometric overlap sampling, which is exactly equivalent to uniform-without-replacement support sampling followed by independent Rademacher signs. All raw and summary CSVs are retained.

## 8. Recommendation for ICASSP Main Paper

**CONTINUE / SUPPORTED.** Add the curvature-weighted floor as the phase-averaged refinement of the scalar `Δ` floor, and present the support-size result as an audit-law refinement. Do not frame sparse perturbations themselves as a novelty claim; the experiment validates the resolution-audit probability law.

### Reproduction

```bash
cd /root/autodl-tmp/EviZO-VP/RC-ZOQO/rc_zoqo_validation
python experiments/exp10_curvature_weighted_floor.py
python experiments/exp11_support_size_audit.py
```

### Outputs

- `outputs/addendum2/csv/table_F_curvature_floor.csv`
- `outputs/addendum2/csv/table_G_support_audit.csv`
- `outputs/addendum2/figures/fig11_curvature_weighted_floor.pdf`
- `outputs/addendum2/figures/fig12_support_size_audit.pdf`
- `outputs/addendum2/figures/fig13_diffuse_vs_concentrated_support.pdf`
- `outputs/addendum2/figures/fig14_support_phase_transition.pdf`

