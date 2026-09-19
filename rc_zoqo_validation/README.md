# RC-ZOQO theory validation

This is a self-contained Python/NumPy validation suite for the ICASSP 2027 RC-ZOQO work. It deliberately uses deterministic synthetic objectives first; the logistic experiment uses a reproducible synthetic binary dataset and does not access gradients during optimization.

```bash
cd /root/autodl-tmp/EviZO-VP/RC-ZOQO/rc_zoqo_validation
python experiments/run_validation.py
```

The command writes raw CSVs, publication PDFs and PNG previews under `outputs/`, plus `outputs/EXPERIMENT_REPORT.md`. Individual scripts are available in `experiments/`; `run_all.py` executes Experiments 1–6 and stops before logistic regression if a MUST PASS counterexample is found. Every objective call has an explicit query counter. Quantizer statistics distinguish rounding from clipping.

Dependencies are listed in `requirements.txt` (the current environment already provides them). No LLM or network download is required.

## Addendum experiments

Run the grid-alignment and concentration-aware audit validations with:

```bash
python experiments/exp8_grid_alignment.py
python experiments/exp9_refined_audit.py
```

Their summary is in `outputs/ADDENDUM_REPORT.md`.
## Addendum II

```bash
python experiments/exp10_curvature_weighted_floor.py
python experiments/exp11_support_size_audit.py
```

The report and all Addendum II tables/figures are under `outputs/addendum2/`.

## Final theory validation

```bash
python experiments/exp12_fixed_state_optimal_support.py
python experiments/exp13_bit_invariant_audit.py
python experiments/exp13b_floor_audit_matching.py
```

The report and all final-validation tables/figures are under `outputs/final/`.

## Practical neural-network experiment

```bash
python experiments/exp14_mnist_mlp.py
```

Quantized ZO adaptation of an MLP (784-256-128-10) on MNIST with 4-bit and 8-bit uniform
affine quantizers; methods: FP two-point ZO (reference), ZOQO fixed-resolution, ZOQO
clipping-safe, and RC-ZOQO (clipping-safe + Resolution Audit in the one-grid regime).
The report and all tables/figures are under `outputs/practical/`; the trained source
checkpoint is cached in `.cache/` (git-ignored) and reused on re-runs.

The gate experiment (head-only block, accepting side of the audit) runs with:

```bash
python experiments/exp15_gate_behavior.py
```

Its tables (`table_N`, `table_O`), figure `fig19_gate_behavior.*` and status log live under
`outputs/practical/` as well.

The gate is also exercised on a large LoRA-style adapter block (16,640 parameters, rank-16
adapters around the frozen first layer; 3-epoch and 1-epoch backbones):

```bash
python experiments/exp16_lora_gate.py    # 3-epoch backbone
python experiments/exp16b_lora_gate_e1.py  # 1-epoch backbone
```

Tables `table_P`--`table_S` and `lora_gate*_status.json` record the results: for
$\gamma\ge0.05$ the gate stops at entry (0 accepted), for $\gamma=0$ it never stops (140/140
accepted), and forced one-grid updates buy at most $0.005$--$0.010$ loss.

The same-γ cross-case sweep (full model, 4-bit) runs with:

```bash
python experiments/exp17_gamma_full.py
```

It produces `table_T_gamma_full_model.csv` / `gamma_full_status.json`: γ=0.05 and γ=0.25 stop
at entry in all seeds (0 accepted), while γ=0 accepts one step then stalls at 95.22%. Combined
with exp15 (head-only continues at γ=0.05), this is the paper's state-based gate evidence.

