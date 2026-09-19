# RC-ZOQO Practical Experiment Report — Quantized ZO Adaptation of an MNIST MLP

Mandatory real-network experiment of `Docs/RC_ZOQO_Final_Practical_and_Paper_Spec.md` (Section A).
All numbers below are produced by `experiments/exp14_mnist_mlp.py` (3 seeds, 30 runs, ≈80 s on one
RTX 4080 SUPER).

## 0. Setup

- **Model**: MLP 784-256-128-10 (ReLU), trained with Adam (3 epochs) on the local MNIST copy
  (`/root/autodl-tmp/data/mnist`, no download); checkpoint test accuracy **0.9672**.
- **Black-box objective**: cross-entropy loss on a fixed 4096-sample training subset; test
  accuracy (standard 10k test set) is used for reporting only. True gradients are computed only
  for offline diagnostics and never for optimizer decisions.
- **Quantizer**: global uniform affine over the checkpoint parameters.
  Effective resolutions: **Δ(4-bit) = 0.04388** (15 levels), **Δ(8-bit) = 0.00258** (255 levels).
  4-bit quantization alone costs 0.40 pp accuracy (0.9672 → 0.9632) at run start; 8-bit is
  lossless (0.9672).
- **Methods**: FP two-point ZO (continuous probe/step schedule, reference); ZOQO-style
  fixed-resolution update; ZOQO + clipping-safe querying; RC-ZOQO = clipping-safe querying +
  Resolution Audit on a fixed support (`m = 1024`, `γ = 0.25`, `K = 15`, stop after the first
  rejected audit).
- **Schedule**: intended step `8Δ·ρ^t` with `ρ = exp(-ln 16 / 250)`, so the nominal update falls
  below one grid step (one-grid regime) at **t = 251**; all methods run the same 400 iterations,
  probe supports are 1024 random coordinates, quantized updates are exactly one grid step Δ
  (fixed resolution). Clipping-safe methods mask probe coordinates without one-step margin.
- **Audit**: `min_σ f(x + σΔr) ≤ f(x) − γΔ²m`, one-grid-step Rademacher directions on a fixed
  support chosen before the run.

## 1. Main result (4-bit, primary)

Means over 3 seeds (std ≤ 0.0013).

| Method | test acc (entry → end) | loss (end) | total queries | post-floor queries | audit queries | accepted / rejected | clipping ratio | wall clock |
|---|---|---|---|---|---|---|---|---|
| FP two-point ZO (reference) | 0.9672 → **0.9678** | 0.0958 | 802 | – | – | – | – | 1.1 s |
| ZOQO fixed-resolution | 0.9632 → 0.9596 | 0.1145 | 802 | 299 | 0 | 0 / 0 | 1.51% | 2.2 s |
| ZOQO clipping-safe | 0.9622 → 0.9602 | 0.1096 | 790 | 299 | 0 | 0 / 0 | 0% | 2.4 s |
| **RC-ZOQO (K=15)** | 0.9622 → **0.9622** | 0.1060 | **523** | **32** | 31 | 0 / 1 | 0% | 1.5 s |

Claim-level outcome:

1. **Resource-limited regime identified — SUPPORTED.** Every run crosses the scheduled one-grid
   boundary at t = 251 and RC-ZOQO invokes the audit exactly there; 3/3 runs stop after the first
   rejected audit (accept rate 0.0, early-stop rate 1.0).
2. **Wasted post-floor queries removed — SUPPORTED.** 32 vs 299 post-floor queries (−89.3%) and
   523 vs 802 total queries (−34.8%), with a bounded audit overhead of 2K+1 = 31 queries.
3. **No accuracy loss — SUPPORTED.** RC-ZOQO ends at 0.9622, i.e. **+0.26 pp** above
   fixed-resolution ZOQO and **+0.20 pp** above clipping-safe ZOQO; forced one-grid updates cost
   the baselines 0.20–0.36 pp, while RC-ZOQO's stopping costs 0.00 pp. The FP reference remains
   0.56 pp higher (quantization, not the audit, is the source of that gap).
4. **Clipping-safe querying — PARTIALLY SUPPORTED.** The 4-bit clipping ratio drops from 1.51% to
   0 and the unresolved-coordinate counter becomes non-zero (6 skipped iterations), but the
   accuracy difference vs fixed-resolution is inside the seed noise (+0.06 pp). At 8-bit clipping
   is negligible (≈4e-6).

K-sensitivity (4-bit, RC-ZOQO): K ∈ {5,10,15,23} produces **identical** stopping decisions and
accuracy (0.9622, stop at t = 251, one rejected audit each); only the audit overhead changes
(11 / 21 / 31 / 47 queries). The certificate decision is budget-insensitive here; the budget only
sets what you pay to obtain it.

## 2. Cross-bit sanity check (8-bit)

| Method | test acc (end) | total queries | post-floor queries | clipping ratio |
|---|---|---|---|---|
| FP two-point ZO (reference) | 0.9678 | 802 | – | – |
| ZOQO fixed-resolution | 0.9681 | 802 | 299 | 3.7e-6 |
| ZOQO clipping-safe | 0.9681 | 802 | 299 | 0 |
| **RC-ZOQO (K=15)** | **0.9683** | **535** | **32** | 0 |

At 8-bit (Δ = 0.00258) the lattice is nearly lossless; RC-ZOQO again matches the best accuracy
while removing 89.3% of the post-floor queries (0 accepted, 1 rejected audit per run).

## 3. Offline diagnostics (never used for decisions)

- **Audit margin at the floor** (4-bit, m = 1024): typical best one-step decrease 4.9e-5 versus the
  certificate threshold γΔ²m = 0.493 — a ratio of ≈ 10⁴. The audit rejects because a single grid
  step demonstrably cannot deliver a curvature-dominated decrease at this state.
- **Support-size scan (post-hoc)**: m ∈ {16, 64, 256, 1024} gives thresholds 0.0077 / 0.0308 /
  0.1232 / 0.493 against measured decreases 1.7e-5 / 2.3e-5 / −1.4e-6 / 4.9e-5. Audits reject for
  every m and accuracy is unchanged (~0.962): the stop decision is not an artifact of the chosen
  support.
- **True (subset) gradient norms**: FP 0.266 → 0.068; quantized runs 0.484 → 0.17 — the quantized
  trajectories end further from the reference optimum, consistent with a resolution floor.
- Runtime/memory: ≤ 2.4 s and 255 MB GPU peak memory per run.

## 4. Specification mapping and honesty notes

- RC-ZOQO is not required to beat ZOQO by a large margin; here it slightly exceeds it, but it does
  **not** beat the FP reference (−0.56 pp). Report exactly this.
- Accepted post-floor updates are 0 in this regime: the γΔ²m certificate is far above achievable
  single-grid-step decreases, so the audit acts as a *certified stopping rule*. The value is the
  query saving, not extra progress.
- The audit does use extra queries (bounded, 2K+1); it is triggered only in the one-grid regime.
- No claim that RC-ZOQO eliminates the representation floor, that fixed 4-bit reaches arbitrary
  accuracy, that sparse probing is novel, or that the certificate-optimal support is the
  empirical-optimal support.

## 5. Gate behaviour — the accepting side (Experiment 15)

`experiments/exp15_gate_behavior.py` complements the full-model runs with the *accepting* side of
the audit. The optimized block is the 1,290-parameter classifier head of the same MLP, the schedule
is accelerated so that the one-grid regime starts at iteration 60 (budget 200), and all policies
share the pre-floor trajectory. INT4 leaves this block at 93.85% (96.68% right after quantization),
so the floor is **not** yet reached when the regime begins.

| policy (INT4, 3 seeds) | test acc. | queries | accepted audits |
|---|---|---|---|
| stop at entry (schedule) | 93.85% | 118 | 0 |
| forced one-grid updates | 94.93% | 398 | – |
| audit γ=0.25, m=16 | 94.26% | 178 | 2.7 |
| audit γ=0.05, m=16 | 94.99% | 256 | 21.7 |
| audit γ=0, m=64 | **96.08%** | 663 | 103.7 |
| unquantized reference (FP) | 96.79% | 402 | – |

Interpretation (see `csv/table_N_gate_head_summary.csv`, `csv/table_O_gate_reference.csv`):

- The audit is a **state-based gate, not a schedule rule**: with the same state and budget, γ traces
  the continue/stop frontier (stop after ≤3 accepted steps at γ=0.25, continue 104 steps at γ=0).
- γ=0.05 matches the forced baseline with 36% fewer queries (94.99% @ 256 vs 94.93% @ 398); γ=0
  exceeds it (96.08% vs 94.93%) at higher cost; the conservative γ=0.25 gives up recoverable
  accuracy. All three are correct for their certificate level — the schedule stop (93.85%) is
  demonstrably premature.
- γ=0 never stops (140/140 accepted at m=16): on a noisy floor a pure-descent test has **no**
  stopping power, which is why the certificate needs γ>0.
- INT8: same block near its floor — the gate stops after one accepted step (96.75% @ 169 queries vs
  96.72% @ 402 for forced updates).
- Caveats: accepted steps pay their own search cost (2–5 queries/step here), and certified subset-loss
  decreases need not transfer to test accuracy; the FP row uses a fixed continuous schedule
  (0.01·ρ^t) and is a scale-sensitivity reference only.

## 5b. LoRA-style large block and γ self-calibration

**γ self-calibration.** The forcing constant can be set gradient-free from the one-grid-scale
two-point history: γ⋆ = median over steps with probe p_t = Δ of
|f(x_t + p_t r_t) − f(x_t − p_t r_t)| / (2Δ²m). On the head gate this yields
γ⋆ ∈ [0.40, 0.62] (INT4) and [1.36, 1.85] (INT8) — the conservative end of the frontier
(the statistic is inflated by the one-step curvature term), and the γ=γ⋆ row in Table 2 stops
at entry (93.85%, 149 queries, 0 accepted).

**LoRA-style block (Experiment 16/16b).** We scale the gate 13×: a rank-16 LoRA adapter
(16,640 parameters, A: 16×784, B: 256×16) around the frozen first layer, on both the converged
3-epoch backbone (`exp16_lora_gate.py`) and a deliberately under-trained 1-epoch backbone
(`exp16b_lora_gate_e1.py`). Results (`table_P`–`table_S`, `lora_gate*_status.json`):

- 4-bit stop-at-entry vs forced one-grid updates: loss 0.10461 vs 0.09925 (3-epoch),
  0.24867 vs 0.23836 (1-epoch) — forced stepping buys ≤ 0.01 loss either way.
- γ ≥ 0.05: gate stops at entry in every seed (0 accepted) — a 16.6k block on top of a
  converged layer is already at/near its resolution floor, so the stopping side of the audit
  reproduces at scale.
- γ = 0: 140/140 accepted and the loop never stops — on a noisy floor a pure-descent test has
  no stopping power; the certificate needs γ > 0.
- Honest takeaway for the paper: acceptance that pays off requires a genuinely floor-less
  block (like the INT4 classification head), not merely a bigger block.

## 5c. Same-γ cross-case evidence and query accounting (Experiment 17)

**Query accounting (Table 1).** `queries_pre_floor` is captured at one-grid entry before any
entry-iteration queries; `Post = total − pre`. Fixed ZOQO: 503 pre + 299 post = 802. RC-ZOQO
probes clipping-safe, and six pre-entry iterations have every sampled coordinate clipped, costing
zero queries: 491 pre + 32 post = 523. The trajectory-matched baseline is therefore clipping-safe
ZOQO (491 + 299 = 790). Hence post-floor saving 89.3% holds against either baseline, and the
total saving is 33.8% like-for-like (523 vs 790) — the 12-query gap to fixed ZOQO is the
clipping-safe effect, not audit overhead. At INT8 clipping rarely binds: 503 + 32 = 535 vs 802.

**Same-γ cross-case evidence (`exp17_gamma_full.py`, `table_T_gamma_full_model.csv`,
`gamma_full_status.json`).** Full-model 4-bit sweep over γ:

- γ=0.05, 3/3 seeds: reject at entry, 0 accepted, 32 post-floor queries, accuracy preserved
  (96.22%) — while in the head-only gate (Exp15) the SAME γ=0.05 continues: 21.7 accepted
  steps, 94.99% @ 256 queries, +1.14 points over the schedule stop. This is the paper's
  strongest evidence that the gate decision tracks *state*, not schedule.
- γ=0.25, 3/3: identical stop-at-entry behavior (the exp14 default).
- γ=0, 3/3: accepts one strict decrease, stalls at the next audit, ends at 95.22% (−0.8 points
  vs entry) — a pure-descent certificate at a noisy floor is not accuracy-preserving; motivates γ>0.

## 6. Reproduction and outputs

```bash
cd /root/autodl-tmp/EviZO-VP/RC-ZOQO/rc_zoqo_validation
python experiments/exp14_mnist_mlp.py        # full-model protocol, ≈ 80 s
python experiments/exp15_gate_behavior.py    # head-only gate behaviour, ≈ 30 s
```

- `outputs/practical/csv/mnist_runs_raw.csv` — all 30 runs (per-seed metrics)
- `outputs/practical/csv/table_J_mnist_4bit_summary.csv` — 4-bit main table
- `outputs/practical/csv/table_K_mnist_8bit_summary.csv` — 8-bit cross-bit table
- `outputs/practical/csv/table_L_audit_K_sensitivity.csv` — K ∈ {5,10,15,23}
- `outputs/practical/csv/table_M_audit_support_sensitivity.csv` — post-hoc m scan
- `outputs/practical/csv/gate_head_runs_raw.csv` — head-only runs (per-seed)
- `outputs/practical/csv/table_N_gate_head_summary.csv` — gate frontier over γ and m
- `outputs/practical/csv/table_O_gate_reference.csv` — stop-at-entry / forced / FP references
- `outputs/practical/figures/fig17_mnist_query_savings.pdf` — query/accuracy trade-off + post-floor cost
- `outputs/practical/figures/fig18_audit_budget_sensitivity.pdf` — K sensitivity
- `outputs/practical/figures/fig19_gate_behavior.pdf` — loss vs policy at stop (head-only)
- `outputs/practical/logs/*.json` — per-run logs, `practical_status.json`, `gate_status.json`
