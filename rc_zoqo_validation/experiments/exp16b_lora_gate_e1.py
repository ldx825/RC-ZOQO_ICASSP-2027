#!/usr/bin/env python3
"""Exp16b: LoRA gate on a 1-epoch (under-trained) checkpoint.

Same protocol as exp16, but the backbone is trained for a single epoch so the
adapter block starts further from its floor and the audit has resolvable signal
at the one-grid entry. This is the "accepting side at a realistic block scale".
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

import exp14_mnist_mlp as m14
import exp16_lora_gate as e16
from src.utils import write_csv, write_json

CKPT_E1 = m14.CACHE / 'mlp_mnist_e1.pt'
CSV = e16.CSV
LOG = e16.LOG


def train_e1(Xtr, ytr, Xte, yte):
    if CKPT_E1.exists():
        payload = torch.load(CKPT_E1, map_location=m14.DEVICE)
        return payload['state_dict'], payload['test_accuracy']
    torch.manual_seed(0)
    model = m14.MLP().to(m14.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    perm = torch.randperm(Xtr.shape[0], device=m14.DEVICE)
    for start in range(0, Xtr.shape[0], 256):
        index = perm[start:start + 256]
        optimizer.zero_grad()
        loss = F.cross_entropy(model(Xtr[index]), ytr[index])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = float((model(Xte).argmax(1) == yte).float().mean())
    torch.save({'state_dict': model.state_dict(), 'test_accuracy': accuracy}, CKPT_E1)
    return model.state_dict(), accuracy


def main():
    for path in (e16.FIG, CSV, LOG):
        path.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = m14.get_data()
    state_dict, accuracy = train_e1(Xtr, ytr, Xte, yte)
    print('1-epoch checkpoint test accuracy:', round(accuracy, 4))
    torch.manual_seed(12345)
    subset = torch.randperm(Xtr.shape[0], device=m14.DEVICE)[:m14.SUBSET]
    X_sub, y_sub = Xtr[subset], ytr[subset]

    rows, references = [], []
    for bits in e16.BITS:
        for seed in e16.SEEDS:
            for policy in ('stop_at_entry', 'forced', 'fp'):
                row = e16.run_lora(policy, bits, 0.0, seed, Xte, yte, X_sub, y_sub, state_dict)
                row['ckpt'] = 'e1'
                references.append(row)
        for gamma in e16.GAMMAS:
            for seed in e16.SEEDS:
                row = e16.run_lora('rc', bits, gamma, seed, Xte, yte, X_sub, y_sub, state_dict)
                row['ckpt'] = 'e1'
                rows.append(row)

    write_csv(CSV / 'lora_gate_e1_runs_raw.csv', rows)
    summary = e16.aggregate(rows, ['bits', 'gamma'])
    write_csv(CSV / 'table_R_lora_gate_e1_summary.csv', summary)
    reference_summary = e16.aggregate(references, ['bits', 'policy'])
    write_csv(CSV / 'table_S_lora_gate_e1_reference.csv', reference_summary)
    write_json(LOG / 'lora_gate_e1_status.json', {
        'checkpoint_test_accuracy': accuracy, 'rank': e16.RANK,
        'adapter_dimension': 16640, 'entry_iter': e16.ENTRY_ITER, 'iterations': e16.ITERS,
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
