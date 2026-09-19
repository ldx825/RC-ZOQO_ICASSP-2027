# RC-ZOQO Final Theory Validation Report

Validation of `Docs/RC_ZOQO_Final_Theory_Validation.md`: fixed-state optimal audit support
(Experiment 12), bit-invariant audit calibration (Experiment 13), and floor/audit scale
matching (Experiment 13B). Deterministic synthetic quadratics only; no network access; every
claim is labelled SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED.

## 1. Experiment 12 — Fixed-State Optimal Audit Support

**SUPPORTED.**

Settings: `q ∈ {128,512}`, `k ∈ {1,4,16,64,q}`, signal regimes `r = βq ∈ {0.05,0.2,0.5,0.8}`,
`L = 1`, `γ = 0.25`, `a = 0.01`. For every `(q,k,r)` the gradient norm is held fixed at
`‖g‖ = aqC/√r` so that `β = a²qC²/‖g‖² = r/q` for **all** tested `m`, the gradient is
equal-magnitude `k`-sparse (`ν = 1/k`), and each tested `m` uses 100,000 independent
support/sign draws. The tested grid is all powers of two up to `q` plus `floor(m0)`, `ceil(m0)`
and the clipped integer `m*`. Results cover 40 configurations and 407 `(configuration, m)`
rows (`csv/table_H_fixed_state_optimal_support.csv`, `csv/exp12_theory_checks.csv`).

1. **No bound violations.** The 3σ Monte Carlo check `p_emp + 3σ ≥ p_LB` holds in all 407 rows.
   The empirical probability exceeds the refined lower bound by a median of 0.287
   (minimum −0.0010, inside the Monte Carlo envelope; maximum 0.960).
2. **`A ≤ 0` regime is decreasing.** All 12 configurations with `ν ≤ 3/(q+2)`
   (`k = 64,128` at `q = 128`; `k = 512` at `q = 512`) give a strictly decreasing refined
   bound on the tested grid, and `m* = 1` — `m = 1` is optimal, as predicted.
3. **`A > 0` regime is unimodal at the predicted root.** All 24 interior-maximizer
   configurations satisfy `argmax_m p_LB = m0` to `max|argmax − m0| = 1.3e-3` (within two
   scan grid cells); the log-derivative `−2βBm² − 3βAm + A` changes sign exactly at `m0`.
   A further 4 configurations (`k = 1`, `ν = 1`, `B = 0`) have `m0 = 1/(3β) > q`; their
   feasible maximizer is clipped to the boundary `m = q`, exactly as the specification
   requires.
4. **Empirical optimum ≠ lower-bound optimum (reported honestly).** The empirical best `m`
   equals the predicted integer `m*` in only 12/40 configurations. Example: at
   `(q,k,r) = (512,64,0.2)`, `m0 = 45.4` and `m* = 45`, while the empirical curve keeps rising
   to `m = 128` (`p_emp = 0.126, 0.833, 0.902, 0.709` at `m = 1, 46, 128, 512`). The refined
   bound is a certificate on the audit probability, not a predictor of the empirically optimal
   support size.

Figure `figures/fig15_fixed_state_optimal_support.pdf` (q = 512, r = 0.2) shows the three
representative regimes: diffuse `k = q = 512` (`m* = 1`, empirically near 1 everywhere),
intermediate `k = 16` (`m* = 108`), concentrated `k = 1` (`m* = 512`).

## 2. Experiment 13 — Bit-Invariant Audit Calibration

**SUPPORTED.**

Settings: `q ∈ {64,256}`; isotropic and two log-spaced anisotropic Hessians from
Experiment 10; `m ∈ {1,8,q}`; `Δ ∈ {0.005,0.01,0.03,0.1,0.3}`; four fixed grid-normalized
states `z` per `q` (per-coordinate bounds `b ∈ {0.5,1,2,4}`); 50,000 audit directions per
configuration with **identical supports/signs reused across Δ**. The audit is executed by
actual objective evaluations `f(x ± Δr)` at `x = x* + Δz` with the one-grid-step condition
`min_σ f(x + σΔr) ≤ f(x) − γΔ²m` (no algebraic shortcut). Results: 72 configurations,
360 rows (`csv/table_I_bit_invariant_audit.csv`).

1. **Δ-invariance.** For every fixed `(q,H,z,m)` the audit success probability is identical
   across all five grid spacings: maximum spread across Δ is exactly 0.0, against a Monte
   Carlo standard error of ≈2.2e-3 at `p ≈ 0.5`.
2. **Per-direction decisions.** The normalized decision mismatch against the `Δ = 0.005`
   reference is exactly 0.0 over all 72 × 50,000 direction vectors: the implemented audit is
   bit-invariant, not merely equal in expectation.
3. **Floor consistency.** The empirical RMS gradient floor matches `Δ‖H‖_F/√12` with maximum
   relative error 6.2e-4.
4. **Boundary observation.** For rounding-cell states (`b = 0.5`, `|z_i| ≤ 1/2`) the success
   probability is zero in all 90 rows: from a nearest-grid point no one-grid-step move
   strictly decreases the separable quadratic, so the audit has nothing to certify. This is
   consistent with the Experiment 10 floor interpretation. Non-degenerate states (`b ≥ 1`)
   provide the non-zero, Δ-flat levels of the figure — e.g. at `q = 256`, `b = 2`:
   isotropic `p = 0.625, 0.053, 0.0` and anisotropic `p = 0.472, 0.330, 0.0` for
   `m = 1, 8, q`.

Figure `figures/fig16_bit_invariant_audit.pdf`: audit success probability versus `Δ`, flat for
every fixed normalized state.

## 3. Experiment 13B — Floor / Audit Scale Matching

**SUPPORTED.** For every `(H, q, m, γ)` the ratio

`T_m / F_Δ = √12 (γ/L + 1/2) √(mq/r_H)`

is independent of `Δ` (spread < 1e-12) and agrees with the closed form to 4.4e-16 across all
180 rows of `csv/table_F_floor_audit_matching.csv`. The special case `γ = 0`, isotropic
curvature (`r_H = q`), `m = 1` gives `T_1/F_Δ = 1.7320508075688772 = √3` to machine precision.

## 4. Claims not supported

None of the three specified claims failed. The only deliberately negative finding is the
rounding-cell boundary statement (§2.4): a one-grid-step audit is never certifying from a
nearest-grid state. This complements, rather than contradicts, the floor results and the
invariance claims.

## 5. Reproduction

```bash
cd /root/autodl-tmp/EviZO-VP/RC-ZOQO/rc_zoqo_validation
python experiments/exp12_fixed_state_optimal_support.py
python experiments/exp13_bit_invariant_audit.py
python experiments/exp13b_floor_audit_matching.py
```

`python experiments/exp12_13_final.py` runs all three (total runtime ≈ 1 minute).

## 6. Outputs

- `outputs/final/csv/table_H_fixed_state_optimal_support.csv`
- `outputs/final/csv/exp12_theory_checks.csv`
- `outputs/final/csv/table_I_bit_invariant_audit.csv`
- `outputs/final/csv/table_F_floor_audit_matching.csv`
- `outputs/final/figures/fig15_fixed_state_optimal_support.pdf` (+ `.png`)
- `outputs/final/figures/fig16_bit_invariant_audit.pdf` (+ `.png`)
- `outputs/final/logs/exp12_status.json`, `exp13_status.json`, `exp13b_status.json`

**Novelty note.** Sparse support is not presented as a novelty claim: it is only the mechanism
that makes the resolution-audit analysis of Experiments 12–13B quantitative.
