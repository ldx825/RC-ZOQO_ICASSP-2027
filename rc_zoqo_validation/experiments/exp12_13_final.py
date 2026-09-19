#!/usr/bin/env python3
"""RC-ZOQO final theory validation.

Experiment 12: fixed-state optimal audit support (support-size law with fixed gradient).
Experiment 13: bit-invariant audit calibration (Delta-free audit decision in grid-normalized coordinates).
Experiment 13B: floor / audit scale matching.
"""
from pathlib import Path
import sys
import math
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import ROOT, write_csv, write_json

OUT = ROOT / 'outputs' / 'final'
FIG = OUT / 'figures'
CSV = OUT / 'csv'
LOG = OUT / 'logs'

L_SMOOTH = 1.0
GAMMA = 0.25
A_STEP = 0.01
C_SCALE = GAMMA + L_SMOOTH / 2.0
TRIALS = 100000

HESSIANS = {
    'isotropic': lambda d: np.ones(d),
    'logspace_0.1_10': lambda d: np.geomspace(0.1, 10.0, d),
    'logspace_0.01_100': lambda d: np.geomspace(0.01, 100.0, d),
}


def dirs():
    for path in (OUT, FIG, CSV, LOG):
        path.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------- Experiment 12

def lower_bound(m, q, nu, beta):
    m = float(m)
    denominator = nu + 3.0 * (m - 1.0) / (q - 1.0) * (1.0 - nu)
    return (1.0 - beta * m) ** 2 * (m / q) / denominator


def lower_bound_vec(m, q, nu, beta):
    m = np.asarray(m, dtype=np.float64)
    denominator = nu + 3.0 * (m - 1.0) / (q - 1.0) * (1.0 - nu)
    return (1.0 - beta * m) ** 2 * (m / q) / denominator


def lower_bound_log_derivative(m, q, nu, beta):
    # Sign of d/dm log p_LB; equals d/dm log p_LB times m(1-beta m)(nu + B(m-1)) > 0.
    B = 3.0 * (1.0 - nu) / (q - 1.0)
    A = nu - B
    return -2.0 * beta * B * m * m - 3.0 * beta * A * m + A


def continuous_m0(q, nu, beta):
    B = 3.0 * (1.0 - nu) / (q - 1.0)
    A = nu - B
    if B == 0.0:
        return 1.0 / (3.0 * beta)
    if A <= 0.0:
        return None
    return (-3.0 * A + math.sqrt(9.0 * A * A + 8.0 * A * B / beta)) / (4.0 * B)


def integer_star(q, nu, beta):
    m0 = continuous_m0(q, nu, beta)
    if m0 is None:
        return 1, None
    candidates = {int(min(q, max(1, math.floor(m0)))), int(min(q, max(1, math.ceil(m0))))}
    best = max(candidates, key=lambda m: lower_bound(m, q, nu, beta))
    return best, m0


def support_success(q, k, m, norm, rng, trials=TRIALS):
    overlap = rng.hypergeometric(k, q - k, m, size=trials)
    signed = 2.0 * rng.binomial(overlap, 0.5) - overlap
    threshold = A_STEP * m * C_SCALE
    return float(np.mean(np.abs(signed) * norm / math.sqrt(k) >= threshold))


def exp12(rng):
    regimes = [0.05, 0.2, 0.5, 0.8]
    qs = [128, 512]
    rows = []
    figure_data = {}
    for q in qs:
        powers = [2 ** i for i in range(int(math.log2(q)) + 1)]
        for k in [1, 4, 16, 64, q]:
            nu = 1.0 / k
            for r in regimes:
                beta = r / q
                norm = A_STEP * q * C_SCALE / math.sqrt(r)
                m_star, m0 = integer_star(q, nu, beta)
                tested = set(powers) | {m_star}
                if m0 is not None:
                    tested |= {math.floor(m0), math.ceil(m0)}
                tested = sorted({int(min(q, max(1, mv))) for mv in tested})
                probabilities = {}
                for m in tested:
                    p = support_success(q, k, m, norm, rng)
                    probabilities[m] = p
                best_m = max(tested, key=lambda m: probabilities[m])
                for m in tested:
                    p = probabilities[m]
                    se = math.sqrt(max(p * (1.0 - p), 1e-18) / TRIALS)
                    lb = lower_bound(m, q, nu, beta)
                    rows.append({
                        'q': q, 'k': k, 'nu': nu, 'effective_dimension': k, 'r': r,
                        'beta': beta, 'gradient_norm': norm, 'm': m,
                        'empirical_success_probability': p,
                        'empirical_standard_error': se,
                        'theoretical_lower_bound': lb,
                        'predicted_continuous_m0': '' if m0 is None else m0,
                        'predicted_integer_m_star': m_star,
                        'empirical_best_m': best_m,
                        'theorem_violation': int(p + 3.0 * se < lb),
                    })
                if q == 512 and r == 0.2:
                    figure_data[k] = (tested, probabilities, m_star)
    write_csv(CSV / 'table_H_fixed_state_optimal_support.csv', rows)

    checks = []
    for q in qs:
        axis = np.linspace(1.0, float(q), 200001)
        resolution = float(axis[1] - axis[0])
        for k in [1, 4, 16, 64, q]:
            nu = 1.0 / k
            for r in regimes:
                beta = r / q
                sub = sorted([row for row in rows if row['q'] == q and row['k'] == k and row['r'] == r],
                             key=lambda row: row['m'])
                m_star, m0 = integer_star(q, nu, beta)
                B = 3.0 * (1.0 - nu) / (q - 1.0)
                A = nu - B
                values = lower_bound_vec(axis, q, nu, beta)
                observed_argmax = float(axis[int(np.argmax(values))])
                if m0 is None:
                    predicted = 'decreasing'
                    feasible_argmax = 1.0
                    unimodal = int(all(sub[i]['theoretical_lower_bound'] > sub[i + 1]['theoretical_lower_bound']
                                       for i in range(len(sub) - 1)))
                else:
                    feasible_argmax = min(float(q), max(1.0, m0))
                    if feasible_argmax >= float(q) - 1e-9:
                        # Maximizer is clipped to the feasible boundary m = q.
                        predicted = 'boundary-max-at-q'
                        grid = np.linspace(1.0, float(q), 5000)
                        unimodal = int(all(lower_bound_log_derivative(m, q, nu, beta) > 0.0 for m in grid))
                    else:
                        predicted = 'interior-max'
                        probe = max(1e-6, 1e-4 * feasible_argmax)
                        unimodal = int(lower_bound_log_derivative(feasible_argmax - probe, q, nu, beta) > 0.0 >
                                       lower_bound_log_derivative(feasible_argmax + probe, q, nu, beta))
                gap = abs(observed_argmax - feasible_argmax)
                continuous_ok = int(bool(unimodal) and gap <= max(0.01 * feasible_argmax, 2.0 * resolution))
                integer_argmax = sub[int(np.argmax([row['theoretical_lower_bound'] for row in sub]))]['m']
                lo = max(1, math.floor(feasible_argmax) - 1)
                hi = min(q, math.ceil(feasible_argmax) + 1)
                checks.append({
                    'q': q, 'k': k, 'nu': nu, 'r': r, 'beta': beta, 'A': A, 'B': B,
                    'predicted_regime': predicted,
                    'predicted_continuous_m0': '' if m0 is None else m0,
                    'feasible_argmax': feasible_argmax,
                    'observed_continuous_argmax': observed_argmax,
                    'continuous_gap': gap,
                    'predicted_integer_m_star': m_star,
                    'observed_integer_argmax': integer_argmax,
                    'empirical_best_m': sub[0]['empirical_best_m'],
                    'unimodal_or_monotone_ok': unimodal,
                    'continuous_argmax_ok': continuous_ok,
                    'integer_argmax_matches': int(lo <= integer_argmax <= hi),
                })
    write_csv(CSV / 'exp12_theory_checks.csv', checks)

    violations = [row for row in rows if row['theorem_violation']]
    monotone_ok = all(check['unimodal_or_monotone_ok'] for check in checks)
    gap_ok = all(check['continuous_argmax_ok'] for check in checks)
    match_ok = all(check['integer_argmax_matches'] for check in checks)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    labels = {512: 'diffuse  ($k=q=512$)', 16: 'intermediate  ($k=16$)', 1: 'concentrated  ($k=1$)'}
    for ax, k in zip(axes, [512, 16, 1]):
        tested, probabilities, m_star = figure_data[k]
        nu = 1.0 / k
        beta = 0.2 / 512
        ax.semilogx(tested, [probabilities[m] for m in tested], 'o-', ms=3.4, color='C0', label='empirical')
        ax.semilogx(tested, [lower_bound(m, 512, nu, beta) for m in tested], 's--', ms=3.0, color='C1', label='lower bound')
        ax.axvline(m_star, color='0.45', ls=':', lw=1.1)
        ax.annotate(rf'$m^\star={m_star}$', xy=(m_star, 0.06), xytext=(m_star, 0.06),
                    ha='center', fontsize=8, color='0.35')
        ax.set(xlabel='support size $m$', title=labels[k])
        ax.grid(alpha=.25, which='both')
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel('audit success probability')
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, 'fig15_fixed_state_optimal_support')

    passed = not violations and monotone_ok and gap_ok and match_ok
    write_json(LOG / 'exp12_status.json', {
        'passed': bool(passed),
        'configurations': len({(row['q'], row['k'], row['r']) for row in rows}),
        'rows': len(rows),
        'trials_per_m': TRIALS,
        'theorem_violations': len(violations),
        'violation_examples': violations[:3],
        'unimodal_or_monotone_all_ok': bool(monotone_ok),
        'continuous_argmax_all_ok': bool(gap_ok),
        'integer_argmax_all_match': bool(match_ok),
        'regime_counts': {
            regime: sum(1 for check in checks if check['predicted_regime'] == regime)
            for regime in ('decreasing', 'interior-max', 'boundary-max-at-q')
        },
        'empirical_best_m_examples': [
            {'q': row['q'], 'k': row['k'], 'r': row['r'],
             'm_star': row['predicted_integer_m_star'], 'empirical_best_m': row['empirical_best_m']}
            for row in rows[::40]
        ],
    })
    print('Experiment 12:', passed, '| violations:', len(violations), '| rows:', len(rows))
    return passed


# ----------------------------------------------------------------- Experiment 13

def exp13(rng):
    dims = [64, 256]
    deltas = [0.005, 0.01, 0.03, 0.1, 0.3]
    state_bounds = [0.5, 1.0, 2.0, 4.0]
    ms_by_dim = {64: [1, 8, 64], 256: [1, 8, 256]}
    ndirs = 50000
    chunk = 10000
    rows = []
    figure_data = {}
    for d in dims:
        states = [rng.uniform(-b, b, size=d) for b in state_bounds]
        z_floor = rng.uniform(-0.5, 0.5, size=(20000, d))
        for hessian_name, maker in HESSIANS.items():
            h = maker(d)
            frobenius = float(math.sqrt(np.sum(h * h)))
            floors = {}
            for delta in deltas:
                empirical = float(math.sqrt(np.mean(np.sum((delta * h * z_floor) ** 2, axis=1))))
                floors[delta] = (empirical, delta * frobenius / math.sqrt(12.0))
            for z_id, (b, z) in enumerate(zip(state_bounds, states)):
                for m in ms_by_dim[d]:
                    generator = np.random.default_rng([20270910, d, z_id, m])
                    decisions = {delta: np.zeros(ndirs, dtype=bool) for delta in deltas}
                    for start in range(0, ndirs, chunk):
                        size = min(chunk, ndirs - start)
                        support = np.argsort(generator.random((size, d)), axis=1)[:, :m]
                        signs = generator.choice(np.array([-1.0, 1.0]), size=(size, m))
                        direction = np.zeros((size, d))
                        np.put_along_axis(direction, support, signs, axis=1)
                        for delta in deltas:
                            x = delta * z
                            f0 = 0.5 * float(np.sum(h * x * x))
                            plus = x + delta * direction
                            minus = x - delta * direction
                            fp = 0.5 * (plus * plus) @ h
                            fm = 0.5 * (minus * minus) @ h
                            decisions[delta][start:start + size] = np.minimum(fp, fm) <= f0 - GAMMA * delta * delta * m
                    reference = decisions[deltas[0]]
                    for delta in deltas:
                        success = decisions[delta]
                        p = float(np.mean(success))
                        se = math.sqrt(max(p * (1.0 - p), 1e-18) / ndirs)
                        mismatch = float(np.mean(success != reference))
                        empirical_floor, theoretical_floor = floors[delta]
                        rows.append({
                            'Delta': delta, 'q': d, 'm': m, 'hessian': hessian_name, 'z_id': z_id,
                            'z_bound': b, 'directions': ndirs,
                            'empirical_audit_success_probability': p,
                            'empirical_standard_error': se,
                            'normalized_decision_mismatch_rate': mismatch,
                            'rms_gradient_floor': empirical_floor,
                            'theoretical_rms_floor': theoretical_floor,
                        })
                    if d == 256 and z_id == 2:
                        figure_data[(hessian_name, m)] = [float(np.mean(decisions[delta])) for delta in deltas]
    write_csv(CSV / 'table_I_bit_invariant_audit.csv', rows)

    spreads = {}
    for row in rows:
        key = (row['q'], row['hessian'], row['z_id'], row['m'])
        spreads.setdefault(key, []).append(row['empirical_audit_success_probability'])
    max_spread = max(max(values) - min(values) for values in spreads.values())
    max_mismatch = max(row['normalized_decision_mismatch_rate'] for row in rows)
    max_floor_relative_error = max(abs(row['rms_gradient_floor'] / row['theoretical_rms_floor'] - 1.0)
                                   for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for ax, hessian_name in zip(axes, ['isotropic', 'logspace_0.1_10']):
        for m, style in zip([1, 8, 256], ['o-', 's--', '^:']):
            ax.semilogx(deltas, figure_data[(hessian_name, m)], style, ms=4, label=f'$m={m}$')
        ax.set(xlabel=r'grid spacing $\Delta$', ylabel='audit success probability',
               title=hessian_name.replace('_', ' '))
        ax.grid(alpha=.25, which='both')
        ax.set_ylim(-0.03, max(0.1, ax.get_ylim()[1]))
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, 'fig16_bit_invariant_audit')

    passed = max_mismatch <= 1e-9 and max_spread <= 0.01 and max_floor_relative_error <= 0.05
    write_json(LOG / 'exp13_status.json', {
        'passed': bool(passed),
        'configurations': len(spreads),
        'rows': len(rows),
        'directions_per_configuration': 50000,
        'deltas': deltas,
        'max_success_probability_spread_across_delta': max_spread,
        'max_normalized_decision_mismatch_rate': max_mismatch,
        'max_rms_floor_relative_error': max_floor_relative_error,
    })
    print('Experiment 13:', passed, '| max spread:', max_spread, '| max mismatch:', max_mismatch)
    return passed


# ----------------------------------------------------------------- Experiment 13B

def exp13b():
    deltas = [0.005, 0.01, 0.03, 0.1, 0.3]
    rows = []
    for q in [64, 256]:
        for hessian_name, maker in HESSIANS.items():
            h = maker(q)
            L = float(h.max())
            frobenius = float(math.sqrt(np.sum(h * h)))
            r_H = frobenius ** 2 / (L * L)
            for gamma in (GAMMA, 0.0):
                for m in [1, 8, q]:
                    entries = []
                    for delta in deltas:
                        F = delta * frobenius / math.sqrt(12.0)
                        T = delta * (gamma + L / 2.0) * math.sqrt(m * q)
                        ratio = T / F
                        formula = math.sqrt(12.0) * (gamma / L + 0.5) * math.sqrt(m * q / r_H)
                        entries.append((delta, F, T, ratio, formula))
                    spread = max(entry[3] for entry in entries) - min(entry[3] for entry in entries)
                    independent = int(spread <= 1e-12 * max(entry[3] for entry in entries))
                    for delta, F, T, ratio, formula in entries:
                        rows.append({
                            'hessian': hessian_name, 'q': q, 'm': m, 'gamma': gamma, 'Delta': delta,
                            'L': L, 'r_H': r_H, 'F_Delta': F, 'T_m': T,
                            'T_m_over_F_Delta': ratio, 'formula_ratio': formula,
                            'relative_error': abs(ratio / formula - 1.0),
                            'delta_independent': independent,
                        })
    write_csv(CSV / 'table_F_floor_audit_matching.csv', rows)

    special = [row for row in rows if row['gamma'] == 0.0 and row['m'] == 1 and row['hessian'] == 'isotropic']
    sqrt3_gap = max(abs(row['T_m_over_F_Delta'] - math.sqrt(3.0)) for row in special)
    max_relative_error = max(row['relative_error'] for row in rows)
    all_independent = all(row['delta_independent'] for row in rows)
    passed = all_independent and max_relative_error < 1e-12 and sqrt3_gap < 1e-12
    write_json(LOG / 'exp13b_status.json', {
        'passed': bool(passed),
        'rows': len(rows),
        'all_delta_independent': bool(all_independent),
        'max_relative_error': max_relative_error,
        'special_case_gamma0_isotropic_m1_T1_over_FDelta': special[0]['T_m_over_F_Delta'],
        'special_case_sqrt3_gap': sqrt3_gap,
    })
    print('Experiment 13B:', passed, '| special case T1/F:', special[0]['T_m_over_F_Delta'])
    return passed


def main():
    dirs()
    rng = np.random.default_rng(20270910)
    p12 = exp12(rng)
    p13 = exp13(rng)
    p13b = exp13b()
    print('Final theory validation:', p12 and p13 and p13b)


if __name__ == '__main__':
    main()
