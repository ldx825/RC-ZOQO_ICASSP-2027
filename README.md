# RC-ZOQO: Certifying Resolution Limits in Quantized Zeroth-Order Optimization

Official code and paper materials for the **ICASSP 2027** submission.

> **Code:** https://github.com/ldx825/RC-ZOQO_ICASSP-2027 · **Paper PDF:** [`RC_ZOQO_ICASSP2027_V06/main.pdf`](RC_ZOQO_ICASSP2027_V06/main.pdf)

**Authors:** Dongxu Liu, Wenjia Shi, Dongyang Liu, Xiang Luo (Jilin University / UESTC)

## Repository layout

| Path | Contents |
|---|---|
| `RC_ZOQO_ICASSP2027_V06/` | Paper LaTeX source, figures, compiled `main.pdf` |
| `rc_zoqo_validation/` | Self-contained NumPy validation suite (no LLM / no network needed): theory-verification experiments, 4/8-bit MNIST MLP and LoRA-gate practical experiments; reports and figures are written under `outputs/` |

## Quick start (validation suite)

```bash
cd rc_zoqo_validation
pip install -r requirements.txt

python experiments/run_validation.py                    # Experiments 1-6 (incl. MUST-PASS counterexample gate)
python experiments/exp8_grid_alignment.py               # Addendum
python experiments/exp9_refined_audit.py
python experiments/exp10_curvature_weighted_floor.py    # Addendum II
python experiments/exp11_support_size_audit.py
python experiments/exp12_fixed_state_optimal_support.py # Final theory validation
python experiments/exp13_bit_invariant_audit.py
python experiments/exp13b_floor_audit_matching.py
python experiments/exp14_mnist_mlp.py                   # 4/8-bit MLP on MNIST
python experiments/exp15_gate_behavior.py               # Gate behavior
python experiments/exp16_lora_gate.py                   # LoRA adapter-block gate
python experiments/exp16b_lora_gate_e1.py
```

## Compiling the paper

```bash
cd RC_ZOQO_ICASSP2027_V06 && latexmk -pdf main.tex
```

ICASSP template files are included (`spconf.sty`, `IEEEbib.bst`).

## Citation

```bibtex
@inproceedings{liu2027rczoqo,
  title     = {RC-ZOQO: Certifying Resolution Limits in Quantized Zeroth-Order Optimization},
  author    = {Liu, Dongxu and Shi, Wenjia and Liu, Dongyang and Luo, Xiang},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## Contact

Dongxu Liu — liudx9924@mails.jlu.edu.cn
