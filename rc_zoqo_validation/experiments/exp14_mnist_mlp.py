#!/usr/bin/env python3
"""RC-ZOQO practical experiment: quantized zeroth-order adaptation of an MNIST MLP.

Implements Section A of Docs/RC_ZOQO_Final_Practical_and_Paper_Spec.md:

* model: MLP 784-256-128-10 (ReLU), trained on MNIST, then quantized ZO adaptation.
* quantizer: global uniform affine, effective resolution Delta = (rmax-rmin)/(2^b-1).
* methods: FP two-point ZO (reference), fixed-resolution ZOQO-style update,
  ZOQO + clipping-safe querying, RC-ZOQO (clipping-safe + Resolution Audit in the
  one-grid regime with a fixed audit support).
* precisions: 4-bit (primary) and 8-bit (cross-bit sanity check).
* audit: min_sigma f(x + sigma Delta r) <= f(x) - gamma Delta^2 m with K in {5,10,15,23}.
* the black-box oracle is the training loss on a fixed 4096-sample MNIST subset;
  true gradients are computed only for offline diagnostics, never for decisions.
"""
from pathlib import Path
import sys
import math
import json
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import ROOT, write_csv, write_json

DATA_ROOT = Path('/root/autodl-tmp/data/mnist')
OUT = ROOT / 'outputs' / 'practical'
FIG = OUT / 'figures'
CSV = OUT / 'csv'
LOG = OUT / 'logs'
CACHE = ROOT / '.cache'
CKPT = CACHE / 'mlp_mnist.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEEDS = (0, 1, 2)
ITERS = 400
ENTRY_ITER = 250
MU_REL = 8.0
RHO = math.exp(-math.log(16.0) / ENTRY_ITER)  # intended probe < delta/2 first at t = ENTRY_ITER
GAMMA = 0.25
M_PROBE = 1024
M_AUDIT = 1024
PATIENCE = 1
K_MAIN = 15
K_GRID = (5, 10, 15, 23)
BITS = (4, 8)
SUBSET = 4096
FP_MU0 = 0.02  # FP reference probe/update scale (continuous, no quantization grid)
METHODS = ('fp', 'zoqo_fixed', 'zoqo_safe', 'rc')


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def dirs():
    for path in (OUT, FIG, CSV, LOG, CACHE):
        path.mkdir(parents=True, exist_ok=True)


def get_data():
    transform = torchvision.transforms.ToTensor()
    train = torchvision.datasets.MNIST(root=str(DATA_ROOT), train=True, download=False, transform=transform)
    test = torchvision.datasets.MNIST(root=str(DATA_ROOT), train=False, download=False, transform=transform)
    Xtr = train.data.to(DEVICE).float().view(-1, 784) / 255.0
    ytr = train.targets.to(DEVICE)
    Xte = test.data.to(DEVICE).float().view(-1, 784) / 255.0
    yte = test.targets.to(DEVICE)
    return Xtr, ytr, Xte, yte


def get_checkpoint(Xtr, ytr, Xte, yte):
    """Train (or reuse) the source MLP checkpoint."""
    if CKPT.exists():
        payload = torch.load(CKPT, map_location=DEVICE)
        return payload['state_dict'], payload['test_accuracy']
    torch.manual_seed(0)
    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        perm = torch.randperm(Xtr.shape[0], device=DEVICE)
        for start in range(0, Xtr.shape[0], 256):
            index = perm[start:start + 256]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(Xtr[index]), ytr[index])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        accuracy = float((model(Xte).argmax(1) == yte).float().mean())
    torch.save({'state_dict': model.state_dict(), 'test_accuracy': accuracy}, CKPT)
    return model.state_dict(), accuracy


class Quantizer:
    def __init__(self, weights, bits):
        self.bits = bits
        self.rmin = float(weights.min())
        self.rmax = float(weights.max())
        self.levels = float(2 ** bits - 1)
        self.delta = (self.rmax - self.rmin) / self.levels

    def quantize(self, x):
        scaled = np.rint((x - self.rmin) / self.delta)
        clipped = int(np.sum((scaled < 0.0) | (scaled > self.levels)))
        scaled = np.clip(scaled, 0.0, self.levels)
        return self.rmin + scaled * self.delta, clipped


class BlackBox:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.queries = 0

    def set_vector(self, vector):
        tensor = torch.as_tensor(np.asarray(vector, dtype=np.float32), device=DEVICE)
        with torch.no_grad():
            offset = 0
            for parameter in self.model.parameters():
                count = parameter.numel()
                parameter.copy_(tensor[offset:offset + count].view_as(parameter))
                offset += count

    def value(self, vector):
        self.queries += 1
        self.set_vector(vector)
        with torch.no_grad():
            return float(F.cross_entropy(self.model(self.X), self.y))

    def gradient(self, vector):
        """Offline diagnostic only; never used for optimizer decisions."""
        self.set_vector(vector)
        self.model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(self.X), self.y)
        loss.backward()
        gradient = torch.cat([p.grad.detach().flatten() for p in self.model.parameters()])
        self.model.zero_grad(set_to_none=True)
        return gradient.cpu().numpy().astype(np.float64)


def model_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy().astype(np.float64)


def test_accuracy(model, vector, Xte, yte):
    with torch.no_grad():
        tensor = torch.as_tensor(np.asarray(vector, dtype=np.float32), device=DEVICE)
        offset = 0
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(tensor[offset:offset + count].view_as(parameter))
            offset += count
        correct = 0
        for start in range(0, Xte.shape[0], 4096):
            logits = model(Xte[start:start + 4096])
            correct += int((logits.argmax(1) == yte[start:start + 4096]).sum())
    return correct / Xte.shape[0]


def run_case(method, bits, seed, K, Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict, iters=ITERS, smoke=False):
    tag = f'{method}_b{bits}_seed{seed}_K{K}'
    started = time.time()
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(seed)
    model = MLP().to(DEVICE)
    model.load_state_dict(state_dict)
    objective = BlackBox(model, X_sub, y_sub)

    weight0 = model_vector(model)
    dimension = weight0.size
    support_audit = np.sort(rng.choice(dimension, size=min(M_AUDIT, dimension), replace=False))

    if method == 'fp':
        quantizer = None
        delta = float('nan')
        x = weight0.copy()
    else:
        quantizer = Quantizer(weight0, bits)
        delta = quantizer.delta
        x, initial_clipped = quantizer.quantize(weight0)

    accuracy_start = test_accuracy(model, x, Xte, yte)
    loss_start = objective.value(x)
    gradient_start = objective.gradient(x)
    diagnostic = {
        'gradient_norm_start': float(np.linalg.norm(gradient_start)),
        'gradient_inf_start': float(np.max(np.abs(gradient_start))),
    }

    clipping_moved = 0
    clipping_events = 0
    queries_audit = 0
    accepted_post_floor = 0
    rejected_audits = 0
    unresolved = 0
    audit_decreases = []
    history = []
    entry_iter = None
    entry_queries = None
    entry_accuracy = None
    stopped_early = False
    stop_iter = None
    f_current = loss_start

    for t in range(iters):
        intended = MU_REL * delta * RHO ** t if method != 'fp' else float('nan')
        one_grid = method != 'fp' and intended < delta / 2.0
        if one_grid and entry_iter is None:
            entry_iter = t
            entry_queries = objective.queries
            entry_accuracy = test_accuracy(model, x, Xte, yte)
            if method == 'rc':
                f_current = objective.value(x)
                queries_audit += 1

        if method == 'rc' and one_grid:
            # Resolution Audit: fixed support, one-grid-step Rademacher directions.
            margin = np.minimum(x - quantizer.rmin, quantizer.rmax - x)
            usable = support_audit[margin[support_audit] >= delta]
            if usable.size == 0:
                unresolved += 1
                stopped_early = True
                stop_iter = t
                break
            success = False
            best_value = None
            applied = None
            for _ in range(K):
                signs = rng.choice(np.array([-1.0, 1.0]), size=usable.size)
                direction = np.zeros(dimension)
                direction[usable] = signs
                value_plus = objective.value(x + delta * direction)
                value_minus = objective.value(x - delta * direction)
                queries_audit += 2
                best_now = min(value_plus, value_minus)
                audit_decreases.append(float(f_current - best_now))
                if best_now <= f_current - GAMMA * delta * delta * M_AUDIT:
                    success = True
                    best_value = best_now
                    applied = direction if value_plus <= value_minus else -direction
                    break
            if success:
                x, _ = quantizer.quantize(x + applied)
                f_current = best_value
                accepted_post_floor += 1
                history.append({'t': t, 'phase': 'one_grid', 'event': 'accept',
                                'queries': objective.queries, 'audit_queries': queries_audit})
                continue
            rejected_audits += 1
            stopped_early = True
            stop_iter = t
            break

        # Two-point ZO probe: probe scale decays, quantized updates use one grid step.
        if method == 'fp':
            probe = FP_MU0 * RHO ** t
            step = probe
        else:
            probe = delta * max(1.0, round(intended / delta))
            step = delta
        indices = rng.choice(dimension, size=min(M_PROBE, dimension), replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=indices.size)
        if method in ('zoqo_safe', 'rc'):
            margin = np.minimum(x - quantizer.rmin, quantizer.rmax - x)
            keep = margin[indices] >= probe
            indices = indices[keep]
            signs = signs[keep]
            if indices.size == 0:
                unresolved += 1
                history.append({'t': t, 'phase': 'one_grid' if one_grid else 'pre_floor',
                                'event': 'unresolved', 'queries': objective.queries})
                continue
        direction = np.zeros(dimension)
        direction[indices] = signs
        if method == 'fp':
            value_plus = objective.value(x + probe * direction)
            value_minus = objective.value(x - probe * direction)
            sign = np.sign(value_plus - value_minus)
            if sign != 0:
                x = x - step * sign * direction
        else:
            plus, clipped_plus = quantizer.quantize(x + probe * direction)
            minus, clipped_minus = quantizer.quantize(x - probe * direction)
            clipping_moved += int(indices.size) * 2
            clipping_events += clipped_plus + clipped_minus
            value_plus = objective.value(plus)
            value_minus = objective.value(minus)
            sign = np.sign(value_plus - value_minus)
            if sign != 0:
                x, clipped_update = quantizer.quantize(x - step * sign * direction)
                clipping_events += clipped_update
        history.append({'t': t, 'phase': 'one_grid' if one_grid else 'pre_floor', 'event': 'step',
                        'queries': objective.queries, 'probe': probe})

    accuracy_end = test_accuracy(model, x, Xte, yte)
    loss_end = objective.value(x)
    gradient_end = objective.gradient(x)
    diagnostic.update({
        'gradient_norm_end': float(np.linalg.norm(gradient_end)),
        'gradient_inf_end': float(np.max(np.abs(gradient_end))),
    })

    result = {
        'method': method, 'bits': bits, 'seed': seed, 'K': K, 'audit_support_m': M_AUDIT,
        'dimension': int(dimension), 'delta': delta,
        'iterations': iters, 'entry_iter': entry_iter, 'stopped_early': bool(stopped_early), 'stop_iter': stop_iter,
        'accuracy_start': accuracy_start, 'accuracy_end': accuracy_end, 'accuracy_at_entry': entry_accuracy,
        'loss_start': loss_start, 'loss_end': loss_end,
        'queries_total': objective.queries, 'queries_pre_floor': entry_queries, 'queries_post_floor': None,
        'queries_audit': queries_audit, 'queries_test_evaluations': 3,
        'accepted_post_floor': accepted_post_floor, 'rejected_audits': rejected_audits, 'unresolved': unresolved,
        'clipping_ratio': (clipping_events / clipping_moved) if clipping_moved else 0.0,
        'clipping_events': int(clipping_events), 'clipping_coordinates_examined': int(clipping_moved),
        'audit_mean_decrease': float(np.mean(audit_decreases)) if audit_decreases else None,
        'audit_max_decrease': float(np.max(audit_decreases)) if audit_decreases else None,
        'audit_threshold': GAMMA * delta * delta * M_AUDIT if method != 'fp' else None,
        'wall_clock_seconds': time.time() - started,
        'gpu_peak_memory_mb': (torch.cuda.max_memory_allocated() / 2 ** 20) if DEVICE == 'cuda' else None,
        'tag': tag,
    }
    if entry_queries is not None:
        result['queries_post_floor'] = objective.queries - entry_queries
    result.update(diagnostic)
    result['history'] = history
    if smoke:
        print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=2))
    return result


def aggregate(results, group_keys):
    summary = {}
    for row in results:
        key = tuple(row[k] for k in group_keys)
        summary.setdefault(key, []).append(row)
    rows = []
    for key, group in summary.items():
        row = dict(zip(group_keys, key))
        row['runs'] = len(group)
        for field in ('accuracy_end', 'accuracy_start', 'loss_end', 'queries_total', 'queries_post_floor',
                      'queries_audit', 'accepted_post_floor', 'rejected_audits', 'clipping_ratio',
                      'wall_clock_seconds', 'gpu_peak_memory_mb', 'delta', 'stop_iter', 'entry_iter',
                      'audit_mean_decrease', 'audit_threshold'):
            values = [g[field] for g in group if g.get(field) is not None]
            row[f'{field}_mean'] = float(np.mean(values)) if values else None
            row[f'{field}_std'] = float(np.std(values)) if values else None
        rows.append(row)
    return rows


def support_sweep(Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict):
    """Post-hoc diagnostic: how the audit threshold gamma*Delta^2*m scales with the support size."""
    global M_AUDIT
    rows = []
    for m_audit in (16, 64, 256, 1024):
        M_AUDIT = m_audit
        for seed in SEEDS:
            result = run_case('rc', 4, seed, K_MAIN, Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict)
            rows.append({
                'audit_support_m': m_audit, 'seed': seed,
                'audit_threshold': result['audit_threshold'],
                'audit_mean_decrease': result['audit_mean_decrease'],
                'accepted_post_floor': result['accepted_post_floor'],
                'rejected_audits': result['rejected_audits'],
                'unresolved': result['unresolved'],
                'queries_audit': result['queries_audit'],
                'queries_post_floor': result['queries_post_floor'],
                'accuracy_end': result['accuracy_end'],
                'stopped_early': result['stopped_early'],
            })
    summary = aggregate(rows, ['audit_support_m'])
    write_csv(CSV / 'table_M_audit_support_sensitivity.csv', summary)
    return rows, summary


def main(smoke=False):
    dirs()
    Xtr, ytr, Xte, yte = get_data()
    state_dict, checkpoint_accuracy = get_checkpoint(Xtr, ytr, Xte, yte)
    torch.manual_seed(12345)
    subset_index = torch.randperm(Xtr.shape[0], device=DEVICE)[:SUBSET]
    X_sub, y_sub = Xtr[subset_index], ytr[subset_index]
    print(f'checkpoint test accuracy: {checkpoint_accuracy:.4f}')

    if smoke:
        result = run_case('rc', 4, 0, 5, Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict, iters=40, smoke=True)
        print('smoke queries:', result['queries_total'], 'seconds:', result['wall_clock_seconds'])
        return

    results = []
    for seed in SEEDS:
        results.append(run_case('fp', 0, seed, K_MAIN, Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict))
        for bits in BITS:
            for method in ('zoqo_fixed', 'zoqo_safe', 'rc'):
                results.append(run_case(method, bits, seed, K_MAIN, Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict))
    for seed in SEEDS:
        for K in (5, 10, 23):
            results.append(run_case('rc', 4, seed, K, Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict))

    raw = [{k: v for k, v in row.items() if k != 'history'} for row in results]
    write_csv(CSV / 'mnist_runs_raw.csv', raw)
    for row in results:
        write_json(LOG / f"{row['tag']}.json", {k: v for k, v in row.items()})

    primary = [row for row in raw if row['method'] == 'fp'
               or (row['bits'] == 4 and (row['method'] != 'rc' or row['K'] == K_MAIN))]
    summary_main = aggregate([r for r in primary if r['method'] in METHODS], ['method', 'bits'])
    write_csv(CSV / 'table_J_mnist_4bit_summary.csv', summary_main)
    summary_8 = aggregate([r for r in raw if r['bits'] == 8 or r['method'] == 'fp'], ['method', 'bits'])
    write_csv(CSV / 'table_K_mnist_8bit_summary.csv', summary_8)
    summary_k = aggregate([r for r in raw if r['method'] == 'rc' and r['bits'] == 4], ['K'])
    write_csv(CSV / 'table_L_audit_K_sensitivity.csv', summary_k)
    sweep_rows, sweep_summary = support_sweep(Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict)

    # Figure 17: query/accuracy trade-off and post-floor query cost (4-bit).
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    colours = {'fp': 'C0', 'zoqo_fixed': 'C1', 'zoqo_safe': 'C2', 'rc': 'C3'}
    labels = {'fp': 'FP two-point ZO', 'zoqo_fixed': 'ZOQO fixed-resolution',
              'zoqo_safe': 'ZOQO clipping-safe', 'rc': 'RC-ZOQO (K=15)'}
    for method in METHODS:
        rows = [r for r in summary_main if r['method'] == method and (r['bits'] == 4 or method == 'fp')]
        if not rows:
            continue
        ax = axes[0]
        ax.errorbar([r['queries_total_mean'] for r in rows], [r['accuracy_end_mean'] for r in rows],
                    yerr=[r['accuracy_end_std'] for r in rows], marker='o', ms=4, capsize=3,
                    color=colours[method], label=labels[method])
        axes[1].bar(labels[method], [r['queries_post_floor_mean'] or 0.0 for r in rows],
                    color=colours[method], alpha=0.85)
    axes[0].set(xlabel='total forward queries', ylabel='test accuracy', title='4-bit quantized ZO adaptation')
    axes[0].grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set(ylabel='queries after one-grid entry', title='Post-floor query cost')
    axes[1].tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(FIG / 'fig17_mnist_query_savings.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig17_mnist_query_savings.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    # Figure 18: audit budget K sensitivity.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    rows = sorted(summary_k, key=lambda r: r['K'])
    axes[0].errorbar([r['K'] for r in rows], [r['accuracy_end_mean'] for r in rows],
                     yerr=[r['accuracy_end_std'] for r in rows], marker='o', ms=4, capsize=3, color='C3')
    axes[0].set(xlabel='audit budget K', ylabel='test accuracy', title='Accuracy vs audit budget')
    axes[0].grid(alpha=.25)
    axes[1].errorbar([r['K'] for r in rows], [r['queries_audit_mean'] for r in rows],
                     yerr=[r['queries_audit_std'] for r in rows], marker='s', ms=4, capsize=3, color='C0',
                     label='audit queries')
    axes[1].set(xlabel='audit budget K', ylabel='audit query overhead', title='Audit cost vs budget')
    axes[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / 'fig18_audit_budget_sensitivity.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig18_audit_budget_sensitivity.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    baseline = {r['method']: r for r in summary_main if r['method'] == 'zoqo_fixed'}
    rc_row = {r['method']: r for r in summary_main if r['method'] == 'rc'}
    savings = None
    if baseline and rc_row:
        base_post = baseline['zoqo_fixed']['queries_post_floor_mean']
        rc_post = rc_row['rc']['queries_post_floor_mean']
        if base_post:
            savings = 1.0 - (rc_post or 0.0) / base_post
    status = {
        'checkpoint_test_accuracy': checkpoint_accuracy,
        'runs': len(raw),
        'delta_4bit': next((r['delta'] for r in raw if r['bits'] == 4), None),
        'delta_8bit': next((r['delta'] for r in raw if r['bits'] == 8), None),
        'post_floor_query_saving_vs_fixed': savings,
        'rc_accuracy_delta_vs_fixed': (rc_row['rc']['accuracy_end_mean'] - baseline['zoqo_fixed']['accuracy_end_mean'])
        if baseline and rc_row else None,
        'rc_accuracy_delta_vs_fp': (rc_row['rc']['accuracy_end_mean'] -
                                    next(r['accuracy_end_mean'] for r in summary_main if r['method'] == 'fp'))
        if rc_row else None,
        'audit_accept_rate_4bit': sum(r['accepted_post_floor'] for r in raw if r['method'] == 'rc' and r['bits'] == 4
                                      and r['K'] == K_MAIN) /
        max(1, sum(r['accepted_post_floor'] + r['rejected_audits'] for r in raw if r['method'] == 'rc'
                   and r['bits'] == 4 and r['K'] == K_MAIN)),
        'stopped_early_rate_4bit': sum(r['stopped_early'] for r in raw if r['method'] == 'rc' and r['bits'] == 4
                                       and r['K'] == K_MAIN) / len(SEEDS),
        'audit_support_sweep_thresholds': {row['audit_support_m']: row['audit_threshold_mean']
                                           for row in sweep_summary},
        'audit_support_sweep_accepted_mean': {row['audit_support_m']: row['accepted_post_floor_mean']
                                              for row in sweep_summary},
    }
    write_json(LOG / 'practical_status.json', status)
    print(json.dumps(status, indent=2))
    print('done')


if __name__ == '__main__':
    main(smoke='smoke' in sys.argv)
