"""Exp17: full-model gamma sweep on the MNIST MLP (4-bit).

Goal (revision plan Step 2): find one gamma whose gate behaviour differs by
*state* rather than by schedule. Exp15 already shows the head-only block
continues at gamma=0.05 (21.7 accepted, 94.99% @ 256 q). This script tests
whether the full model, at its floor, stops under the SAME gamma values.

Expected per revision plan:
  gamma=0    : any strict decrease is accepted -> gate never stops (floor is noisy)
  gamma=0.05 : full-model floor has mean one-grid decrease ~5e-4 << threshold -> stop at entry
  gamma=0.25 : stop at entry (default used in exp14)
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP14 = ROOT / 'experiments' / 'exp14_mnist_mlp.py'
spec = importlib.util.spec_from_file_location('exp14_mnist', EXP14)
exp14 = importlib.util.module_from_spec(spec)
sys.modules['exp14_mnist'] = exp14
spec.loader.exec_module(exp14)

import csv
import torch

GAMMAS = (0.0, 0.05, 0.25)
BITS = 4


def main():
    exp14.dirs()
    Xtr, ytr, Xte, yte = exp14.get_data()
    state_dict, checkpoint_accuracy = exp14.get_checkpoint(Xtr, ytr, Xte, yte)
    torch.manual_seed(12345)
    subset_index = torch.randperm(Xtr.shape[0], device=exp14.DEVICE)[:exp14.SUBSET]
    X_sub, y_sub = Xtr[subset_index], ytr[subset_index]
    print(f'checkpoint test accuracy: {checkpoint_accuracy:.4f}')

    rows = []
    for gamma in GAMMAS:
        exp14.GAMMA = gamma
        for seed in exp14.SEEDS:
            r = exp14.run_case('rc', BITS, seed, exp14.K_MAIN,
                               Xtr, ytr, Xte, yte, X_sub, y_sub, state_dict)
            rows.append({
                'gamma': gamma, 'seed': seed,
                'stopped_early': r['stopped_early'], 'stop_iter': r['stop_iter'],
                'entry_iter': r['entry_iter'],
                'accepted_post_floor': r['accepted_post_floor'],
                'rejected_audits': r['rejected_audits'],
                'queries_total': r['queries_total'],
                'queries_post_floor': r['queries_post_floor'],
                'queries_audit': r['queries_audit'],
                'audit_threshold': r['audit_threshold'],
                'audit_mean_decrease': r['audit_mean_decrease'],
                'accuracy_at_entry': r['accuracy_at_entry'],
                'accuracy_end': r['accuracy_end'],
                'loss_end': r['loss_end'],
                'wall_clock_seconds': r['wall_clock_seconds'],
            })
            print(f"gamma={gamma:>5} seed={seed} stop={r['stopped_early']!s:5} "
                  f"stop_iter={r['stop_iter']!s:5} acc_ep={r['accuracy_at_entry']:.4f} "
                  f"acc_end={r['accuracy_end']:.4f} accepts={r['accepted_post_floor']} "
                  f"post_q={r['queries_post_floor']} tot_q={r['queries_total']} "
                  f"mean_dec={r['audit_mean_decrease']:.2e} thr={r['audit_threshold']:.2e}")

    csv_path = exp14.CSV / 'table_T_gamma_full_model.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {csv_path}')

    # summary per gamma
    import json
    summary = {}
    for gamma in GAMMAS:
        g = [r for r in rows if r['gamma'] == gamma]
        summary[str(gamma)] = {
            'runs': len(g),
            'stopped_early': sum(r['stopped_early'] for r in g),
            'accepted_mean': sum(r['accepted_post_floor'] for r in g) / len(g),
            'queries_total_mean': sum(r['queries_total'] for r in g) / len(g),
            'queries_post_floor_mean': sum(r['queries_post_floor'] for r in g) / len(g),
            'accuracy_end_mean': sum(r['accuracy_end'] for r in g) / len(g),
        }
    out = exp14.LOG / 'gamma_full_status.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
