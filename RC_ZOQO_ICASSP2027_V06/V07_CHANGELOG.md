# V07 change notes (relative to V06)

Working draft: `main.tex` (single paper source; compiled with `pdflatex`, 4 pages of technical
content + references filling page 5). V06 lives in git history, not as a separate file.

## What changed and why

The V06 draft opened with the floor laws and listed three generic contributions, which made the
motivation read as "some formulas about quantization". V07 leads with the practical failure mode
and states what the reader gets.

1. **Abstract and Introduction rewritten (motivation-first).**
   - Pain: in the one-grid regime every iteration still costs forward queries while the grid
     cannot express the intended motion; forced one-grid updates drift (INT4 MLP: 299 extra
     queries, −0.26 accuracy points).
   - Question: *given only function values, when has the grid become the bottleneck?* A schedule
     is explicitly shown to be a poor proxy in both directions (stops too early / spends too
     long).
   - Contributions are phrased as "what you get": (i) price of the grid (floor + bit laws),
     (ii) a state-based certificate (audit, false-stop bound, 2K+1 cost, bit invariance),
     (iii) evidence (full-model stop + gate regime).
2. **New experiment — the audit as a resolution gate (Section 5.3, Table 2).**
   Head-only block (q = 1290) with one-grid entry at iteration 60, where the floor is *not* yet
   reached: schedule stop 93.85%; forced updates 94.93% @ 398 queries; audit γ=0.05 →
   94.99% @ 256; γ=0 → 96.08% @ 663; γ=0.25 stops after 2.7 accepted steps. INT8: gate stops
   after one accepted step (96.75% @ 169 vs 96.72% @ 402). This closes the "why not just stop at
   Δ?" gap: the audit is a state-based gate, γ is the strictness knob, and γ=0 has no stopping
   power (it never stops on a noisy floor).
3. **Honesty / scope additions (no claims removed).**
   - Theorem 2 scope: the guarantee needs θ_m < 1 and is vacuous at the neural scale
     (q ≈ 2.4e5, Δ ≈ 4e-2, m ≈ 1e3); the experiments exercise the certificate test, not the bound.
   - γ guidance (γ = 0 certifies any decrease; larger γ demands curvature-dominated decrease).
   - "Certified stop" wording instead of "declared resolution-limited"; a failed audit is a
     certificate on the searched directions, not a proof that no progress exists.
   - Limitations now list: accepted steps pay search cost, strict γ can stop earlier than blind
     forced stepping, subset-objective decreases need not transfer to test accuracy.
4. **Layout**: thinned inherited prose and proofs (equations and claims unchanged), figures
   resized; no Overfull boxes remain; paper fits 4 pages + references.

## Reproduction

```bash
cd /root/autodl-tmp/EviZO-VP/RC-ZOQO/rc_zoqo_validation
python experiments/exp14_mnist_mlp.py        # full-model protocol (Tables 1)
python experiments/exp15_gate_behavior.py    # gate regime (Table 2, fig19)
cd ../RC_ZOQO_ICASSP2027_V06
pdflatex main.tex && pdflatex main.tex
```
