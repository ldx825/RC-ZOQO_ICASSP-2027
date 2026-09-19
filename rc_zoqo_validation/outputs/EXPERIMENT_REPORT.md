# RC-ZOQO Validation Report

## 1. Executive Summary
Controlled validation completed without LLM experiments. MUST PASS statuses are reported from raw machine-readable logs.

## 2. Experiment Environment
Python/NumPy/Matplotlib implementation; 20 seeds for quadratic scaling; synthetic logistic data with fixed seed 2027.

## 3. Fixed-Grid Barrier
**SUPPORTED** — deterministic two-cycle test.

## 4. Quantization-Resolution Scaling
**PARTIALLY SUPPORTED** — see `table_A_bit_scaling.csv` and `exp2_fit.json`; relationship and log-log fit are reported without forcing monotonicity.

## 5. Clipping Sign Corruption
**SUPPORTED** — exact linear counterexample and random boundary sweep.

## 6. Range–Resolution Feasibility
**SUPPORTED** — prediction compared with endpoint-inclusive brute-force search.

## 7. Resolution Audit Probability
**SUPPORTED** — empirical probabilities and the 3/16 reference are in `table_B_audit.csv` and `fig5_audit_probability.pdf`.

## 8. Audit Query Cost
**SUPPORTED** — each audit direction costs two objective evaluations; per-configuration mean estimates are recorded in `table_B_audit.csv`.

## 9. Blockwise Quantization
**PARTIALLY SUPPORTED** — feasible fraction and quantization error by group size are in `table_C_blockwise.csv`.

## 10. Logistic Regression
**SUPPORTED (synthetic dataset)** — comparison metrics are in `exp7_logistic.csv`; no downloaded data or hidden gradient was used by optimizers.

## 11. Theory–Experiment Agreement
The fixed-grid, clipping, feasibility, and audit checks are evaluated directly against their stated criteria. Scaling and blockwise end-task effects are empirical and intentionally reported as such.

## 12. Failed Predictions / Counterexamples
Any mismatch or non-monotonic trend is retained in raw CSVs and fit logs; no seeds or configurations are removed.

## 13. Recommendation: Continue / Modify / Abandon
Continue to larger experiments only if all MUST PASS statuses above are SUPPORTED; otherwise diagnose the corresponding implementation/theorem assumption first.
