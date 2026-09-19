#!/usr/bin/env python3
"""RC-ZOQO experiment 15: the resolution audit behaves as a state-based gate.

Complements exp14 (full-model adaptation, where the audit always stops the run) with a
small parameter block (last-layer head, q = 1290) where the same audit *accepts* one-grid
steps while a certified decrease remains available, and stops once it does not. This
isolates the "gate" behaviour from a schedule rule: same state, same budget, only gamma
changes the stopping point.

Policies compared (identical pre-floor trajectory and iteration budget):
* stop_at_entry : schedule stop, no post-floor queries
* forced        : fixed-resolution ZOQO, forced one-grid steps until the budget
* rc            : resolution audit gate with budget K, support m, forcing constant gamma
* fp            : continuous (unquantized) head-only two-point ZO reference
"""
from pathlib import Path
import sys
import math
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
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
GAMMAS = ('cal', 0.0, 0.05, 0.25)
SUPPORTS = (16, 64)
BITS = (4, 8)
SEEDS = (0, 1, 2)
RHO = math.exp(-math.log(16.0) / ENTRY_ITER)
FP_PROBE0 = 0.01  # continuous reference step scale (no quantization grid)


def head_indices(model):
    offsets = []
    cursor = 0
    for parameter in model.parameters():
        offsets.append((cursor, cursor + parameter.numel()))
        cursor += parameter.numel()
    return np.arange(offsets[-2][0], offsets[-1][1]), cursor


def run_head(policy, bits, gamma, support, seed, Xte, yte, X_sub, y_sub, state_dict, model):
    """Head-only quantized ZO run under one policy."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.load_state_dict(state_dict)
    head_index, _ = head_indices(model)
    n_head = head_index.size
    local = np.arange(n_head)

    weight0 = m14.model_vector(model)
    if policy == 'fp':
        quantizer = None
        delta = float('nan')
        x = weight0.copy()
        quantized_start = None
    else:
        quantizer = m14.Quantizer(weight0[head_index], bits)
        delta = quantizer.delta
        x = weight0.copy()
        x[head_index], _ = quantizer.quantize(weight0[head_index])
        quantized_start = x.copy()

    objective = m14.BlackBox(model, X_sub, y_sub)
    accuracy_start = m14.test_accuracy(model, x, Xte, yte)
    loss_start = objective.value(x)

    accepted = rejected = unresolved = 0
    audit_queries = 0
    f_current = loss_start
    slopes = []
    gamma_cal = None
    entry_iter = None
    entry_queries = None
    entry_accuracy = None
    stop_iter = None
    for t in range(ITERS):
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
                    gamma_cal = float(np.median(slopes) / (delta * support)) if slopes else 0.05
                    gamma = gamma_cal
        if policy == 'stop_at_entry' and one_grid:
            stop_iter = t
            break

        if policy == 'rc' and one_grid:
            margin = np.minimum(x[head_index] - quantizer.rmin, quantizer.rmax - x[head_index])
            usable = local[margin >= delta]
            if usable.size == 0:
                unresolved += 1
                stop_iter = t
                break
            success = False
            for _ in range(K_AUDIT):
                sel = rng.choice(usable, size=min(support, usable.size), replace=False)
                signs = rng.choice(np.array([-1.0, 1.0]), size=sel.size)
                direction = np.zeros_like(x)
                direction[head_index[sel]] = signs
                value_plus = objective.value(x + delta * direction)
                value_minus = objective.value(x - delta * direction)
                audit_queries += 2
                best_now = min(value_plus, value_minus)
                if best_now <= f_current - gamma * delta * delta * support:
                    success = True
                    applied = direction if value_plus <= value_minus else -direction
                    x[head_index], _ = quantizer.quantize(x[head_index] + delta * applied[head_index])
                    f_current = best_now
                    accepted += 1
                    break
            if success:
                continue
            rejected += 1
            stop_iter = t
            break

        # Two-point ZO step (probe decays, quantized step is one grid step).
        if policy == 'fp':
            probe = FP_PROBE0 * RHO ** t
            step = probe
            sel = rng.choice(local, size=min(PROBE_SUPPORT, n_head), replace=False)
            signs = rng.choice(np.array([-1.0, 1.0]), size=sel.size)
            direction = np.zeros_like(x)
            direction[head_index[sel]] = signs
            value_plus = objective.value(x + probe * direction)
            value_minus = objective.value(x - probe * direction)
            sign = np.sign(value_plus - value_minus)
            if sign != 0:
                x[head_index] = x[head_index] - step * sign * direction[head_index]
            continue

        probe = delta * max(1.0, round(intended / delta))
        sel = rng.choice(local, size=min(PROBE_SUPPORT, n_head), replace=False)
        margin = np.minimum(x[head_index[sel]] - quantizer.rmin, quantizer.rmax - x[head_index[sel]])
        sel = sel[margin >= probe]
        if sel.size == 0:
            unresolved += 1
            continue
        signs = rng.choice(np.array([-1.0, 1.0]), size=sel.size)
        direction = np.zeros_like(x)
        direction[head_index[sel]] = signs
        plus = x.copy()
        minus = x.copy()
        plus[head_index], _ = quantizer.quantize(x[head_index] + probe * direction[head_index])
        minus[head_index], _ = quantizer.quantize(x[head_index] - probe * direction[head_index])
        value_plus = objective.value(plus)
        value_minus = objective.value(minus)
        if probe == delta:
            slopes.append(abs(value_plus - value_minus) / (2.0 * probe))
        sign = np.sign(value_plus - value_minus)
        if sign != 0:
            x[head_index], _ = quantizer.quantize(x[head_index] - delta * sign * direction[head_index])

    accuracy_end = m14.test_accuracy(model, x, Xte, yte)
    loss_end = objective.value(x)
    return {
        'policy': policy, 'bits': bits, 'gamma': str(gamma), 'support': support if policy == 'rc' else 0,
        'gamma_calibrated': gamma_cal,
        'gamma_calibrated': gamma_cal,
        'seed': seed, 'delta': delta, 'entry_iter': entry_iter, 'stop_iter': stop_iter,
        'accuracy_start': accuracy_start, 'accuracy_at_entry': entry_accuracy, 'accuracy_end': accuracy_end,
        'loss_start': loss_start, 'loss_end': loss_end,
        'queries_total': objective.queries,
        'queries_post_entry': (objective.queries - entry_queries) if entry_queries is not None else None,
        'audit_queries': audit_queries, 'accepted': accepted, 'rejected': rejected, 'unresolved': unresolved,
    }


def aggregate(rows, keys):
    summary = {}
    for row in rows:
        summary.setdefault(tuple(row[k] for k in keys), []).append(row)
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
    m14.dirs()
    for path in (FIG, CSV, LOG):
        path.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = m14.get_data()
    state_dict, checkpoint_accuracy = m14.get_checkpoint(Xtr, ytr, Xte, yte)
    torch.manual_seed(12345)
    subset = torch.randperm(Xtr.shape[0], device=m14.DEVICE)[:m14.SUBSET]
    X_sub, y_sub = Xtr[subset], ytr[subset]

    rows = []
    reference = []
    for bits in BITS:
        for seed in SEEDS:
            for policy in ('stop_at_entry', 'forced', 'fp'):
                reference.append(run_head(policy, bits, 0.0, 0, seed, Xte, yte, X_sub, y_sub, state_dict, m14.MLP().to(m14.DEVICE)))
        for gamma in GAMMAS:
            for support in SUPPORTS:
                for seed in SEEDS:
                    rows.append(run_head('rc', bits, gamma, support, seed, Xte, yte, X_sub, y_sub, state_dict, m14.MLP().to(m14.DEVICE)))

    write_csv(CSV / 'gate_head_runs_raw.csv', rows)
    summary = aggregate(rows, ['bits', 'gamma', 'support'])
    write_csv(CSV / 'table_N_gate_head_summary.csv', summary)
    reference_summary = aggregate(reference, ['bits', 'policy'])
    write_csv(CSV / 'table_O_gate_reference.csv', reference_summary)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for ax, bits in zip(axes, BITS):
        labels, losses, queries = [], [], []
        for policy in ('stop_at_entry', 'forced'):
            row = next(r for r in reference_summary if r['bits'] == bits and r['policy'] == policy)
            labels.append({'stop_at_entry': 'stop at entry', 'forced': 'forced steps'}[policy])
            losses.append(row['loss_end_mean'])
            queries.append(row['queries_total_mean'])
        for gamma, support in ((0.25, 64), (0.05, 64), (0.05, 16)):
            row = next(r for r in summary if r['bits'] == bits and r['gamma'] == gamma and r['support'] == support)
            labels.append(f'RC γ={gamma}, m={support}')
            losses.append(row['loss_end_mean'])
            queries.append(row['queries_total_mean'])
        order = np.argsort(queries)
        ax.bar(range(len(labels)), [losses[i] for i in order], color=['C0', 'C1', 'C3', 'C3', 'C3'][:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([labels[i] for i in order], rotation=18, fontsize=7)
        ax.set(ylabel='subset loss at stop (lower better)', title=f'{bits}-bit head-only adaptation')
        ax.grid(alpha=.25, axis='y')
    fig.tight_layout()
    fig.savefig(FIG / 'fig19_gate_behavior.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig19_gate_behavior.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    status = {'checkpoint_accuracy': checkpoint_accuracy, 'entry_iter': ENTRY_ITER, 'iterations': ITERS,
              'head_dimension': 1290}
    for bits in BITS:
        key = f'bits{bits}'
        status[key] = {
            'stop_at_entry_loss': next(r['loss_end_mean'] for r in reference_summary
                                       if r['bits'] == bits and r['policy'] == 'stop_at_entry'),
            'forced_loss': next(r['loss_end_mean'] for r in reference_summary
                                if r['bits'] == bits and r['policy'] == 'forced'),
            'forced_queries': next(r['queries_total_mean'] for r in reference_summary
                                   if r['bits'] == bits and r['policy'] == 'forced'),
            'rc_rows': [{k: row[k] for k in ('gamma', 'support', 'loss_end_mean', 'queries_total_mean',
                                             'accepted_mean', 'rejected_mean', 'stop_iter_mean')}
                        for row in summary if row['bits'] == bits],
        }
    write_json(LOG / 'gate_status.json', status)
    print(json.dumps(status, indent=1, default=str)[:2400])


if __name__ == '__main__':
    main()
