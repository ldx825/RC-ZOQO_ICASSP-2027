# RC-ZOQO: Certifying Resolution Limits in Quantized Zeroth-Order Optimization

Code release for the paper **"RC-ZOQO: Certifying Resolution Limits in Quantized Zeroth-Order Optimization"** (manuscript under review).

- Paper PDF: [`paper/main.pdf`](paper/main.pdf)
- Authors: Dongxu Liu, Wenjia Shi, Dongyang Liu, Xiang Luo

This repository contains the self-contained NumPy/PyTorch validation suite behind every theorem check and experiment in the paper. No LLM, network access or pretrained checkpoint is required.

---

## 1. Code overview

### 1.1 What the paper shows

Quantized zeroth-order (ZO) optimizers keep updating on a discrete grid, but once the nominal probe or update falls below one grid step the run enters a **one-grid regime**: queries keep flowing while the grid cannot express the intended motion. The paper (i) prices that wall and (ii) turns the terminal regime into a black-box statistical decision.

* **Floor laws (Sec. 3).** For quadratics the exact phase-averaged stationarity floor is `F_Δ = Δ‖H‖_F/√12`, so each extra bit approximately halves it; `exp2` recovers this as the slope of `log(final ‖∇f‖)` versus `log Δ`.
* **Resolution audit (Sec. 4.1, Def. 2).** At a fixed state, draw up to `K` one-grid Rademacher directions with support `m ≤ q` on the non-clipping coordinates and probe symmetrically at magnitude `a`. A direction is accepted when `min(f(x+ar), f(x−ar)) ≤ f₀ − γ a² q`; the audit needs only queried function values (exact chord decomposition, no gradient/Hessian/Lipschitz constant).
* **Finite-sample certificate (Sec. 4.2, Thm. 2).** If `p` is the probability that a sampled direction is `γ`-useful, the false-stop probability is exactly `(1−p)^K`, so `K` consecutive failures certify `p ≤ 1 − δ^{1/K}` at confidence `1−δ`; accepted steps form a finite sequence (Prop. 1).
* **Stop / continue validation (Sec. 5).** On INT4/INT8 MNIST MLPs the audit stops a resolution-exhausted full model (89.3% / 33.8% fewer post-floor / total like-for-like queries, no accuracy loss) and *continues* a head-only block, recovering +1.14 points that a fixed schedule discards.

### 1.2 Where the method lives

| Module | Role |
|---|---|
| [`rc_zoqo_validation/src/quantizer.py`](rc_zoqo_validation/src/quantizer.py) | Uniform affine quantizer: grid spacing `Δ = (r_max − r_min)/(2^b − 1)`, `grid_aligned()`, and statistics that separate rounding from clipping (the paper's range-vs-resolution analysis depends on this distinction). |
| [`rc_zoqo_validation/src/zoqo.py`](rc_zoqo_validation/src/zoqo.py) | The baseline quantized ZO step (Sec. 2.1): quantized probe/update with one-grid clamping and the signed update, plus per-step query accounting. |
| [`rc_zoqo_validation/src/rc_zoqo.py`](rc_zoqo_validation/src/rc_zoqo.py) | RC-ZOQO step: `clipping_aware_probe()` shrinks the probe to the clipping-safe distance from the boundary, and `resolution_audit()` binds the audit to the quantizer. |
| [`rc_zoqo_validation/src/audit.py`](rc_zoqo_validation/src/audit.py) | The audit itself (Def. 1): `sufficient_decrease()`, `audit_direction()`, `run_audit()` over `K` directions, and `audit_threshold()`. |
| [`rc_zoqo_validation/src/objectives.py`](rc_zoqo_validation/src/objectives.py) | Quadratic / linear / logistic objectives with an explicit `query_count`; every objective call is counted. |
| [`rc_zoqo_validation/src/metrics.py`](rc_zoqo_validation/src/metrics.py) | Floor statistics (trajectory variance, oscillation range) and the `log–log` bit-law fit. |
| [`rc_zoqo_validation/src/utils.py`](rc_zoqo_validation/src/utils.py) | Output directories (`outputs/csv`, `outputs/figures`, `outputs/logs`), YAML config loading, seeding, CSV/JSON/figure helpers. |

### 1.3 Paper ↔ code correspondence

| Paper | Code |
|---|---|
| Sec. 2.1 — quantized ZO updates, one-grid clamping, `Δ_b = D/(2^b−1)` | `src/quantizer.py`, `src/zoqo.py` |
| Sec. 3 — phase-averaged floor `F_Δ = Δ‖H‖_F/√12` and the bit law | `experiments/exp2_bit_scaling.py` (slope fit via `src/metrics.py::log_log_fit`), `experiments/exp10_curvature_weighted_floor.py` |
| Sec. 3 / Appendix — range–resolution relation, blockwise feasibility | `experiments/exp4_range_resolution.py`, `experiments/exp6_blockwise.py` |
| Sec. 4.1, Def. 2 — resolution audit, γ-sufficient decrease | `src/audit.py`, `src/rc_zoqo.py`; end-to-end sweep `experiments/exp9_refined_audit.py`, `exp11_support_size_audit.py` |
| Sec. 4.1 — clipping-aware probing (clipping can reverse a comparison) | `src/rc_zoqo.py::clipping_aware_probe()`; counterexample `experiments/exp3_clipping_sign_flip.py` (MUST-PASS) |
| Sec. 4.2, Thm. 2 / Prop. 1 — `(1−p)^K` false-stop probability, finite accepted-step bound, grid alignment | `experiments/exp5_audit_probability.py` (`table_B_audit.csv`, `exp5_false_stop.csv`), `experiments/exp8_grid_alignment.py`, `exp12_fixed_state_optimal_support.py`, `exp13_bit_invariant_audit.py`, `exp13b_floor_audit_matching.py` |
| Sec. 5.1 — stopping at a resolution floor (INT4 / INT8 MNIST MLP) | `experiments/exp14_mnist_mlp.py` |
| Sec. 5.2 — continuing despite schedule entry (head-only block) | `experiments/exp15_gate_behavior.py` |
| Sec. 5 — gate on a LoRA-style adapter block, γ behaviour | `experiments/exp16_lora_gate.py`, `exp16b_lora_gate_e1.py`, `exp17_gamma_full.py` |
| MUST-PASS counterexample gate (limit cycle, clipping sign flip, range/resolution mismatch) | `experiments/exp1_fixed_grid_floor.py`, `exp3_clipping_sign_flip.py`, `exp4_range_resolution.py` |
| Logistic-regression instance of the quantized ZO loop | `experiments/exp7_logistic_regression.py` |

### 1.4 Repository layout

```
rc_zoqo_validation/
  src/            quantizer, ZOQO / RC-ZOQO steps, audit, objectives, metrics, IO helpers
  experiments/    exp1..exp17 (theory verification + practical MLP / LoRA-gate experiments)
  configs/        YAML protocol values (quadratic, clipping, blockwise, feasibility, audit, logistic, addendum)
  requirements.txt
paper/main.pdf    compiled paper
```

---

## 2. Installation

```bash
cd rc_zoqo_validation
pip install -r requirements.txt     # numpy, torch, matplotlib, PyYAML
```

Everything runs on CPU. All randomness is seeded per experiment, so results are reproducible.

## 3. Running the validation suite

```bash
cd rc_zoqo_validation

# Experiments 1-6, including the MUST-PASS gates
# (fixed-grid limit cycle, clipping sign flip, range/resolution mismatch).
# exp1/exp3/exp4 failing stops the run before the larger experiments.
python experiments/run_validation.py
```

`run_validation.py` runs the core suite and then the logistic experiment. `run_all.py` runs Experiments 1–6 only. Individual stages:

```bash
python experiments/exp1_fixed_grid_floor.py            # fixed-grid floor / limit cycle (MUST PASS)
python experiments/exp2_bit_scaling.py                 # floor vs Δ: one bit ≈ halves the floor
python experiments/exp3_clipping_sign_flip.py          # clipping flips a comparison (MUST PASS)
python experiments/exp4_range_resolution.py            # range vs resolution (MUST PASS)
python experiments/exp5_audit_probability.py           # audit success probability + (1-p)^K bound
python experiments/exp6_blockwise.py                   # blockwise feasibility vs group size
python experiments/exp7_logistic_regression.py         # logistic instance of the loop

# Addenda and final theory validation
python experiments/exp8_grid_alignment.py
python experiments/exp9_refined_audit.py
python experiments/exp10_curvature_weighted_floor.py
python experiments/exp11_support_size_audit.py
python experiments/exp12_fixed_state_optimal_support.py
python experiments/exp13_bit_invariant_audit.py
python experiments/exp13b_floor_audit_matching.py

# Practical experiments (INT4/INT8 MNIST MLP, gate behaviour, LoRA adapter block)
python experiments/exp14_mnist_mlp.py
python experiments/exp15_gate_behavior.py
python experiments/exp16_lora_gate.py
python experiments/exp16b_lora_gate_e1.py
python experiments/exp17_gamma_full.py
```

Results are written under `rc_zoqo_validation/outputs/`: raw CSVs in `outputs/csv/`, figures in `outputs/figures/`, status JSON in `outputs/logs/`, and per-experiment reports under `outputs/addendum2/`, `outputs/final/` and `outputs/practical/`. `outputs/` is not tracked in git; `experiments/make_report.py` regenerates the markdown reports from the CSVs.

`exp14_mnist_mlp.py` trains the source MLP once and caches the checkpoint in `rc_zoqo_validation/.cache/` (git-ignored) so re-runs are fast; true gradients are computed only for offline diagnostics, never for any algorithm decision.

## 4. Notes and limitations

* RC-ZOQO is a **diagnostic certificate** for existing fixed-resolution quantized ZO loops, not a replacement optimizer: it detects that a run has reached the representable floor, it does not remove it.
* The theory experiments use synthetic quadratics/logistics with explicit query counters; the practical experiments use a small MNIST MLP and a LoRA-style adapter block. Absolute numbers are therefore protocol-specific — `configs/` records the frozen protocol values, and every raw record embeds its seed and configuration.
* The floor bound is vacuous at neural-network scale (see the appendix); the practical claims are the stop/continue decisions, not the floor constant.

## 5. Citation

```bibtex
@misc{liu2026rczoqo,
  title  = {RC-ZOQO: Certifying Resolution Limits in Quantized Zeroth-Order Optimization},
  author = {Liu, Dongxu and Shi, Wenjia and Liu, Dongyang and Luo, Xiang},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## 6. Contact

Dongxu Liu — liudx9924@mails.jlu.edu.cn
