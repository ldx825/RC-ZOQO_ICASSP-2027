# RC-ZOQO Revision Changelog

Revision performed per `RC_ZOQO_Agent_Execution_Guide.md`, relative to the V07 working
draft (the version at the start of this revision pass).

## 1. Narrative changes

- **Abstract rewritten** to the required arc: phenomenon (one-grid regime with queries
  still flowing) → characterization (curvature-weighted floor laws) → black-box
  statistical decision (RC-ZOQO audit) → exact finite-sample certificate → experiments.
  Removed the unsupported earlier claim of "+2.2 accuracy points"; the abstract now
  reports $+1.14$.
- **Introduction restructured** into four paragraphs: (1) background, then the
  overlooked terminal phenomenon with concrete evidence ($299$ extra post-entry
  queries, $0.26$-point drift); (2) why a schedule is a poor proxy — one-grid entry
  is a *trigger* for auditing, not a stopping criterion; (3) what we do — first
  characterize the price of the grid, then turn the terminal regime into a black-box
  statistical decision; (4) three contributions (wall characterization, state-based
  certificate, stop/continue validation).
- **Theorem-context prose added** throughout: statements are no longer stacked bare;
  each is preceded by motivation and followed by interpretation. Required sentence
  included: *"The theorem certifies scarcity of useful directions under the audit
  distribution, not the nonexistence of a descent direction."*
- **Method flow** follows the fixed order: Theorem 1 (floor) → Corollary (bit law) →
  Definition 2 (audit) → Lemma 2 (chord decomposition) → Theorem 2 (finite-sample
  certificate) → Proposition 1 (finite accepted descent) → Corollary 2 (resolution
  invariance) → cell-optimality fact.
- **Experiments reframed**: theory validation now targets $p_{\gamma,m}(x)$ and
  $(1-p)^K$; the stopping side reports the $K{=}15$-failure certificate at the stop
  state; the head-only side reports $93.85\to94.99$ ($+1.14$) under the same
  $\gamma=0.05$ rule, with $\gamma=0$ ($96.08$) explicitly separated as a strictly
  looser policy. Closed-loop sentence added: full-model and head-only are the two
  sides of the same decision problem.
- **Conclusion rewritten** to only (i) resolution wall, (ii) the state-based audit,
  (iii) what theory and experiments show. No new experiments/claims introduced.

## 2. Theorem 2 changes

- The old gradient-based guarantee ($\theta_m$, Paley–Zygmund) is **no longer the
  main guarantee**; it moved to the Appendix as a classical sufficient condition.
- **New Theorem 2 (finite-sample resolution certificate)** built on the sampled
  useful-direction mass $p_{\gamma,m}(x)=\Pr_r[A_r(x)=1]$:
  - exact false-stop law: $\Pr(K\text{ failures}\mid x)=(1-p_{\gamma,m}(x))^K$;
    if $p_{\gamma,m}(x)\ge\varepsilon$, false stop $\le(1-\varepsilon)^K$;
  - exact confidence certificate: after $K$ failures, $p_{\gamma,m}(x)\le
    1-\delta^{1/K}$ with confidence $\ge1-\delta$;
  - numerics kept exact: $K{=}15,\delta{=}0.05\Rightarrow$ fewer than $18.1\%$ of
    sampled directions remain $\gamma$-useful; $p\ge0.2\Rightarrow$ false stop
    $\le0.8^{15}\approx3.52\%$.
- **New Lemma (exact chord decomposition)**: with
  $\widehat s_r=\frac{f(x+ar)-f(x-ar)}{2a}$, $\widehat c_r=\frac{f(x+ar)+f(x-ar)-2f(x)}{2a^2}$,
  one has $f(x\pm ar)-f(x)=a^2\widehat c_r\pm a\widehat s_r$ and
  $\min_\sigma f(x+\sigma ar)-f(x)=a^2\widehat c_r-a|\widehat s_r|$, so the audit is
  equivalent to $|\widehat s_r(x)|-a\widehat c_r(x)\ge\gamma am$ — no gradients,
  Hessian, global $L$, or ambient dimension $q$ required.
- **New Proposition (finite audited-phase termination)**: for $\gamma>0$ and
  $f\ge f_{\inf}$, $N_{\mathrm{acc}}\le\lfloor(f(x_0)-f_{\inf})/(\gamma a^2m)\rfloor$;
  the audited phase ends in a certified stop. Explicit disclaimer retained: no claim
  of convergence to a continuous stationary point.
- Main text also states the structural obstruction: a gradient-norm-only lower bound
  cannot generally be dimension-free for sparse random audits (concentrated gradient
  missed by the sampled support).
- Old result $(13/16)^{15}<4.5\%$ now appears **only in the Appendix**.

## 3. Appendix changes

- Reorganized into: (7.1) Lemma 1 + Theorem 1 proof; (7.2) Corollary 1 + auxiliary
  facts (range–resolution feasibility; cell optimality); (7.3) proofs of the chord
  Lemma, Theorem 2, Proposition 1, Corollary 2; (7.4) *Why the gradient-based bound
  becomes vacuous at neural scale*.
- (7.4) retains the classical material: smoothness sufficient decrease, sparse
  Rademacher second/fourth moments, $\theta_m$ theorem with Paley–Zygmund proof, and
  explains that the failure is **structural** — $\theta_m$ grows linearly in $q$
  ($\theta_m\gg1$ at $q\approx2.4\times10^5$) and an $m$-sparse support misses
  concentrated gradient energy with probability approaching $1-m/q$.
- Proofs condensed to short named subsections; duplication removed; equations that
  need no numbering were inlined with verbal cross-references (this also fixed stale
  equation-number artifacts from inlined labels).

## 4. Figure / layout changes

- **Figure 1** enlarged from `0.40\linewidth` to `0.98\columnwidth` (kept vector,
  no rasterization or trimming of axis labels); axes/legend are legible at the
  printed size; caption now states the two scientific points (unit log–log slope;
  curvature changes the coefficient) without drawing details.
- **Tables merged** into a single two-panel `\small` table (full model on top,
  head-only gate below) with a shortened caption; column padding tightened.
- Conservative spacing tuning only (display skips 2pt, float separations 2pt,
  caption skips reduced); official margins, two-column format and 9pt size unchanged.
  No negative `\vspace` and no font shrinking of body text or proofs.
- Result: technical content (including all proofs) fits pages 1–4; Page 5 contains
  References only.

## 5. Citation corrections

- Bibliography split into `references.bib` (28 entries) compiled with the bundled
  `IEEEbib.bst` via a BibTeX pipeline; all 28 cite keys resolve and all 28 entries
  are cited in the text.
- **Verified against publisher metadata / arXiv / OpenReview:**
  - ZOQO (Bar & Giryes), ICASSP 2025, doi:10.1109/ICASSP49660.2025.10887815;
  - QuZO, EMNLP 2025 main, pp. 5341–5359;
  - QZO, ICLR 2026 (OpenReview poster, arXiv:2505.13430);
  - CAQ-ZO (Shu & Zhu), arXiv:2605.10673, May 2026.
- **AWQ author list corrected** (the previous entry carried a wrong author; replaced
  with the actual MLSys 2024 author list).
- Pruned marginal entries that padded the introduction (e.g., XNOR-Net, ZeroQuant,
  SmoothQuant, Deep Compression, ZO-AdaMM, NES) and kept functional citation groups:
  quantized ZO, classical/neural ZO, quantized deployment, sufficient-decrease direct
  search, quantization numerical effects. No fabricated or unverifiable entries kept.

## 6. ICASSP compliance

- total pages: **5**
- page 1–4: technical content (abstract, introduction, theory, method, experiments,
  conclusion, Appendix proofs)
- page 5 content: **References only** (first item on the page is "8. REFERENCES")
- template: official ICASSP `spconf.sty`, `\ninept`, letter size, untouched margins
- compile status: `pdflatex → bibtex → pdflatex → pdflatex` clean;
  0 undefined references, 0 undefined citations, 0 missing files,
  0 Overfull boxes in `main.log`.

## Revision 3 — References expansion and 9pt compliance (2026-09-15)

- **Reference count 28 → 30.** Two verified entries added and cited in the
  introduction's zeroth-order-estimation cluster (no wording added):
  - `duchi2015optimal` — Duchi, Jordan, Wainwright, Wibisono, "Optimal Rates for
    Zero-Order Convex Optimization: The Power of Two Function Evaluations," IEEE
    Trans. Inf. Theory 61(5):2788–2806, 2015 (Crossref DOI 10.1109/TIT.2015.2409256);
  - `liu2020primer` — Liu et al., "A Primer on Zeroth-Order Optimization in Signal
    Processing and Machine Learning," IEEE Signal Process. Mag. 37(5):43–54, 2020
    (Crossref/OpenAlex DOI 10.1109/MSP.2020.3003837).
- **Font compliance fix.** The reference list is restored to the official 9pt
  (the `\footnotesize` before `\bibliography` was removed). Execution-guide §5 and
  the official template ("no smaller than nine points") do not allow shrinking
  references below the official size; the sibling ZO-IDEP submission also uses 9pt.
- **Bibliography list-spacing tuning (font and margins unchanged).** `\itemsep` is
  set to 0pt and list `\topsep` to 2pt via the spconf `thebibliography` definition,
  solely so that the 30 references fit the reference-only page at 9pt; the page now
  fills to the bottom of both columns (last text row y ≈ 723 of the ~725pt text
  area).
- **Checks.** Bidirectional citation audit clean: 30 cite keys = 30 entries, no
  missing keys, no uncited entries; spot re-verification of the 2025/2026 entries
  (ICASSP 2025 DOI 10.1109/ICASSP49660.2025.10887815; ICLR 2026 arXiv:2505.13430;
  CAQ-ZO arXiv:2605.10673) passed on 2026-09-15.
- **Compliance re-check.** total pages: 5 (page 5 shows "8. REFERENCES" as its
  first item and contains nothing else); 0 undefined citations; 0 Overfull boxes;
  Figure 1 unchanged.
- Candidates verified but NOT added, because page 5 at 9pt is at its capacity
  (≈ 30 references): torczon1997patterns, audet2006mesh, gray1998quantization,
  clopper1934confidence, bernstein2018signsgd.

## 7. Claims intentionally NOT changed

- All experiment numbers and settings: 89.3% post-floor reduction (32 vs 299),
  33.8% total like-for-like reduction (523 vs 790), 96.22% vs 95.96% ($0.26$-point
  drift), INT8 96.83% vs 96.81%, head-only 93.85 → 94.99 (+1.14) at $\gamma=0.05$,
  $\gamma=0$ 96.08% (looser policy), LoRA-scale block stopping-side behaviour,
  floor-law validation errors 1.09%/1.85% and slopes 0.99994–1.00041.
- RC-ZOQO positioning: a diagnostic/stopping certificate for existing quantized ZO
  loops; it does **not** remove the resolution floor, does not claim local
  optimality, and a failed audit certifies scarcity of sampled useful directions,
  not the absence of descent.
- The (13/16)^{15} < 4.5% statement remains, explicitly in the Appendix only.

---

# Revision 2 — Overview figure integration (2026-09-15)

Goal: insert the graphical overview ("Resolution-Certified Quantized Zeroth-Order
Optimization" infographic, panels A–D + end-to-end/state-based-gate strip) into the
manuscript **without violating the ICASSP 2027 5-page rule** (4 pages technical
content + references-only 5th page).

## 8. Figure integration

- Renamed the asset to `figures/fig_overview.png` (ASCII, no spaces); original
  2048×768 px (2.67:1), no white margins to crop.
- Inserted as a **two-column-spanning `figure*`** at the top of page 2, displayed at
  `0.95\textwidth` so internal labels stay readable (print ≈ 4.5–5 pt, high-res
  raster ≈ 300 dpi; zoom-readable in the PDF).
- New **Figure 1**; the floor/law plot is now **Figure 2** (all `\ref`s update
  automatically). The overview is referenced from the Introduction (panel A) and
  from Section 4.1 (panels B–C); the caption states A–D in one short paragraph.
- The figure's content was checked for consistency with the paper: floor law
  $F_\Delta$, the $\gamma$-sufficient decrease test, the $K$-failure law
  $(1-p_{\gamma,m}(x))^K$, and the certificate $p_{\gamma,m}(x)\le1-\delta^{1/K}$
  all match the text.

## 9. Space compensation (to keep 5 pages)

Because the overview occupies ≈ 0.30 page, ≈ 0.85 column of text was reclaimed
(measured by spill height on page 5, driven to zero):

- Abstract and Introduction tightened (no claim removed);
- Sections 2–5 compressed (examples, $\gamma^\star$ note, certificate commentary,
  experiment prose) with all numbers kept;
- Appendix proofs compacted (corollary bounds inlined, proofs merged into single
  flow paragraphs); subsection headings of a few short blocks merged;
- Floor figure narrowed to `0.86\columnwidth`; merged results table uses
  `\arraystretch{0.92}`; captions shortened.
- Only conservative spacing changes; official margins, two-column layout, 9 pt body
  text, and proof font sizes are unchanged.

## 10. Compliance re-verified after insertion

- total pages: **5** (pages 1–4 technical incl. all proofs; page 5 = References only);
- `Figure 1` sits at the top of page 2; no float overflows or misplaced floats;
- compile: `pdflatex → bibtex → pdflatex → pdflatex` clean;
  **0 undefined references, 0 undefined citations, 0 missing files,
  0 Overfull boxes**;
- all headline numbers (89.3%, 33.8%, +1.14, 96.08%, 18.1%, 3.52%, (13/16)^{15})
  still present in the PDF;
- no figure/table renumbering errors (`Fig. 1`/`Fig. 2` resolve correctly).

