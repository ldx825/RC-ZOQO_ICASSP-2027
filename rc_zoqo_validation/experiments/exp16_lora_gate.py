#!/usr/bin/env python3
"""RC-ZOQO experiment 16: LoRA-style parameter block gate + self-calibrated gamma.

Same gate protocol as exp15, but on a LoRA adapter (A:16x784, B:256x16, 16,640
parameters) around the frozen first layer of the MNIST MLP. A larger block than
the classifier head (q=1290), so it tests the gate at a more realistic scale.
Also evaluates a gradient-free self-calibration of gamma from the pre-floor
two-point history:

    gamma* = median_t |f(x_t + p_t r_t) - f(x_t - p_t r_t)| / (2 * p_t * Delta * m)

i.e., the per-iteration directional slope rescaled to one grid step, divided by
the certificate scale Delta^2 m. No gradients and no Lipschitz constant are used.
"""
from pathlib import Path
import sys
import math
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import exp14_mnist_mlp as m14
from src.utils import ROOT, write_csv, write_json

OUT = ROOT / 'outputs' / 'practical'
FIG = OUT / 'figures'
CSV = OUT / 'csv'
LOG = OUT / 'logs'

ENTRY_ITER = 60
ITERS = 200
K_AUDIT = 15
PROBE_SUPPORT = 1024
RANK = 16
GAMMAS = ('cal', 0.0, 0.05, 0.25)
SUPPORT = 64
BITS = (4, 8)
SEEDS = (0, 1, 2)
RHO = math.exp(-math.log(16.0) / ENTRY_ITER)
FP_PROBE0 = 0.002  # continuous reference step scale (adapter params are ~0.02)


class LoRAMLP(nn.Module):
    """Frozen MLP with a LoRA adapter on the first layer. Optimized params = (A, B)."""

    def __init__(self, state_dict, rank=RANK):
        super().__init__()
        for name, value in state_dict.items():
            self.register_buffer(name.replace('.', '_'), value.detach().clone())
        d_in = self.net_0_weight.shape[1]   # 784
        d_out = self.net_0_weight.shape[0]  # 256
        self.A = nn.Parameter(torch.randn(rank, d_in, device=m14.DEVICE) * 0.02)
        self.B = nn.Parameter(torch.zeros(d_out, rank, device=m14.DEVICE))

    def forward(self, x):
        merged = self.net_0_weight + self.B @ self.A
        h = F.relu(x @ merged.t() + self.net_0_bias)
        h = F.relu(h @ self.net_2_weight.t() + self.net_2_bias)
        return h @ self.net_4_weight.t() + self.net_4_bias


def run_lora(policy, bits, gamma, seed, Xte, yte, X_sub, y_sub, state_dict, iters=ITERS):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = LoRAMLP(state_dict).to(m14.DEVICE)
    weight0 = m14.model_vector(model)
    dimension = weight0.size
    objective = m14.BlackBox(model, X_sub, y_sub)

    if policy == 'fp':
        quantizer = None
        delta = float('nan')
        x = weight0.copy()
    else:
        quantizer = m14.Quantizer(weight0, bits)
        delta = quantizer.delta
        x, _ = quantizer.quantize(weight0)

    accuracy_start = m14.test_accuracy(model, x, Xte, yte)
    loss_start = objective.value(x)
    f_current = loss_start

    accepted = rejected = unresolved = 0
    audit_queries = 0
    entry_iter = None
    entry_queries = None
    entry_accuracy = None
    stop_iter = None
    gamma_cal = None
    slopes = []

    for t in range(iters):
        intended = 8.0 * delta * RHO ** t if policy != 'fp' else FP_PROBE0 * RHO ** t
        one_grid = policy != 'fp' and intended < delta / 2.0
        if one_grid and entry_iter is None:
            entry_iter = t
            entry_queries = objective.queries
            entry_accuracy = m14.test_accuracy(model, x, Xte, yte)
            if policy == 'rc':
                f_current = objective.value(x)
                audit_queries += 1
                if gamma == 'cal':
                    gamma_cal = float(np.median(slopes) / (2.0 * delta * SUPPORT)) if slopes else 0.05
                    gamma = gamma_cal
        if policy == 'stop_at_entry' and one_grid:
            stop_iter = t
            break

        if policy == 'rc' and one_grid:
            margin = np.minimum(x - quantizer.rmin, quantizer.rmax - x)
            usable = np.arange(dimension)[margin >= delta]
            if usable.size == 0:
                unresolved += 1
                stop_iter = t
                break
            success = False
            for _ in range(K_AUDIT):
                sel = rng.choice(usable, size=min(SUPPORT, usable.size), replace=False)
                signs = rng.choice(np.array([-1.0, 1.0]), size=sel.size)
                direction = np.zeros(dimension)
                direction[sel] = signs
                value_plus = objective.value(x + delta * direction)
                value_minus = objective.value(x - delta * direction)
                audit_queries += 2
                best_now = min(value_plus, value_minus)
                if best_now <= f_current - gamma * delta * delta * SUPPORT:
                    success = True
                    applied = direction if value_plus <= value_minus else -direction
                    x, _ = quantizer.quantize(x + delta * applied)
                    f_current = best_now
                    accepted += 1
                    break
            if success:
                continue
            rejected += 1
            stop_iter = t
            break

        # two-point ZO step
        if policy == 'fp':
            probe = FP_PROBE0 * RHO ** t
            step = probe
            sel = rng.choice(dimension, size=min(PROBE_SUPPORT, dimension), replace=False)
            signs = rng.choice(np.array([-1.0, 1.0]), size=sel.size)
            direction = np.zeros(dimension)
            direction[sel] = signs
            value_plus = objective.value(x + probe * direction)
            value_minus = objective.value(x - probe * direction)
            sign = np.sign(value_plus - value_minus)
            if sign != 0:
                x = x - step * sign * direction
            continue

        probe = delta * max(1.0, round(intended / delta))
        sel = rng.choice(dimension, size=min(PROBE_SUPPORT, dimension), replace=False)
        margin = np.minimum(x[sel] - quantizer.rmin, quantizer.rmax - x[sel])
        sel = sel[margin >= probe]
        if sel.size == 0:
            unresolved += 1
            continue
        signs = rng.choice(np.array([-1.0, 1.0]), size=sel.size)
        direction = np.zeros(dimension)
        direction[sel] = signs
        plus = x.copy()
        minus = x.copy()
        plus, _ = quantizer.quantize(x + probe * direction)
        minus, _ = quantizer.quantize(x - probe * direction)
        value_plus = objective.value(plus)
        value_minus = objective.value(minus)
        slopes.append(abs(value_plus - value_minus) / probe)
        sign = np.sign(value_plus - value_minus)
        if sign != 0:
            x, _ = quantizer.quantize(x - delta * sign * direction)

    accuracy_end = m14.test_accuracy(model, x, Xte, yte)
    loss_end = objective.value(x)
    return {
        'policy': policy, 'bits': bits, 'gamma': str(gamma), 'support': SUPPORT if policy == 'rc' else 0,
        'gamma_calibrated': gamma_cal, 'seed': seed, 'delta': delta,
        'entry_iter': entry_iter, 'stop_iter': stop_iter,
        'accuracy_start': accuracy_start, 'accuracy_at_entry': entry_accuracy, 'accuracy_end': accuracy_end,
        'loss_start': loss_start, 'loss_end': loss_end,
        'queries_total': objective.queries,
        'queries_post_entry': (objective.queries - entry_queries) if entry_queries is not None else None,
        'audit_queries': audit_queries, 'accepted': accepted, 'rejected': rejected, 'unresolved': unresolved,
    }


def aggregate(rows, keys):
    summary = {}
    for row in rows:
        summary.setdefault(tuple(str(row[k]) for k in keys), []).append(row)
    out = []
    for key, group in summary.items():
        entry = dict(zip(keys, key))
        entry['runs'] = len(group)
        for field in ('accuracy_end', 'accuracy_at_entry', 'loss_end', 'queries_total', 'queries_post_entry',
                      'audit_queries', 'accepted', 'rejected', 'stop_iter', 'delta', 'gamma_calibrated'):
            values = [g[field] for g in group if g[field] is not None]
            entry[f'{field}_mean'] = float(np.mean(values)) if values else None
        out.append(entry)
    return out


def main():
    for path in (FIG, CSV, LOG):
        path.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = m14.get_data()
    state_dict, checkpoint_accuracy = m14.get_checkpoint(Xtr, ytr, Xte, yte)
    torch.manual_seed(12345)
    subset = torch.randperm(Xtr.shape[0], device=m14.DEVICE)[:m14.SUBSET]
    X_sub, y_sub = Xtr[subset], ytr[subset]

    rows = []
    references = []
    for bits in BITS:
        for seed in SEEDS:
            for policy in ('stop_at_entry', 'forced', 'fp'):
                references.append(run_lora(policy, bits, 0.0, seed, Xte, yte, X_sub, y_sub, state_dict))
        for gamma in GAMMAS:
            for seed in SEEDS:
                rows.append(run_lora('rc', bits, gamma, seed, Xte, yte, X_sub, y_sub, state_dict))

    write_csv(CSV / 'lora_gate_runs_raw.csv', rows)
    summary = aggregate(rows, ['bits', 'gamma'])
    write_csv(CSV / 'table_P_lora_gate_summary.csv', summary)
    reference_summary = aggregate(references, ['bits', 'policy'])
    write_csv(CSV / 'table_Q_lora_gate_reference.csv', reference_summary)

    write_json(LOG / 'lora_gate_status.json', {
        'checkpoint_accuracy': checkpoint_accuracy, 'rank': RANK, 'adapter_dimension': 16640,
        'entry_iter': ENTRY_ITER, 'iterations': ITERS,
        'summary': summary, 'references': reference_summary,
    })
    for row in summary:
        print(row['bits'], row['gamma'], '| acc', f"{row['accuracy_end_mean']:.5f}",
              'loss', f"{row['loss_end_mean']:.5f}", 'q', f"{row['queries_total_mean']:.0f}",
              'post', f"{row['queries_post_entry_mean']:.0f}", 'accepted', f"{row['accepted_mean']:.1f}",
              'rejected', f"{row['rejected_mean']:.1f}", 'stop', f"{row['stop_iter_mean']}",
              'gamma*', row['gamma_calibrated_mean'])
    for row in reference_summary:
        print(row['bits'], row['policy'], '| acc', f"{row['accuracy_end_mean']:.5f}",
              'loss', f"{row['loss_end_mean']:.5f}", 'q', f"{row['queries_total_mean']:.0f}")


if __name__ == '__main__':
    main()
