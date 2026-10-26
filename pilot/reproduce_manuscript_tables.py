"""Regenerate the pilot contrasts of the preprint (version 1.2: Tables 5, 7, 8 and 9, Figure 2) from the result files.

Usage:  python pilot/reproduce_manuscript_tables.py
Needs only the Python standard library and NumPy. It reads results.json, results_iter2.json,
results_iter3.json, results_iter4.json and results_iter5_runs.jsonl, recomputes every paired contrast
from the per-run test scores, and checks the SHA-256 of the frozen iteration-4 and iteration-5 configurations.
"""
import hashlib
import json
import math
import os
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05 / 3          # confirmatory family of three tests
SESOI = 0.05


def t_quantile(p, df):
    """Quantile of Student's t by bisection on a numerically integrated CDF."""
    xs = np.linspace(0.0, 60.0, 600001)
    pdf = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
    cdf = 0.5 + np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(xs))])
    return float(np.interp(p, cdf, xs))


def summarize(diffs):
    n, m, sd = len(diffs), st.mean(diffs), st.stdev(diffs)
    se = sd / math.sqrt(n)
    ci95 = (m - t_quantile(0.975, n - 1) * se, m + t_quantile(0.975, n - 1) * se)
    ci12a = (m - t_quantile(1 - ALPHA, n - 1) * se, m + t_quantile(1 - ALPHA, n - 1) * se)
    if ci12a[1] < SESOI and ci12a[0] > -SESOI:
        reading = 'equivalent within ±SESOI'
    elif ci12a[1] < SESOI:
        reading = 'below +SESOI'
    elif ci95[0] > 0 and m >= SESOI:
        reading = 'support'
    else:
        reading = 'inconclusive'
    return m, sd, ci95, ci12a, reading


def load(name):
    with open(os.path.join(HERE, name), encoding='utf-8') as fh:
        return json.load(fh)


def scores(block):
    return {s: run['test']['bss'] for s, run in block.items()}


def report(label, D, F, N=None, ceiling=None):
    seeds = sorted(D, key=int)
    d = [D[s] - F[s] for s in seeds]
    m, sd, ci95, ci12a, reading = summarize(d)
    line = (f'{label:<34} n={len(seeds):<2} D {st.mean(D.values()):.3f}  F {st.mean(F.values()):.3f}  '
            f'D-F {m:+.3f} [{ci95[0]:+.3f}, {ci95[1]:+.3f}]  SD {sd:.3f}  (1-2a) [{ci12a[0]:+.3f}, {ci12a[1]:+.3f}]  {reading}')
    if N is not None:
        line += f'  F-Fnoz {st.mean([F[s] - N[s] for s in seeds if s in N]):+.3f}'
    if ceiling is not None:
        line += f'  F share {st.mean([F[s] / ceiling[s] for s in seeds]):.2f}  D share {st.mean([D[s] / ceiling[s] for s in seeds]):.2f}'
    print(line)


def main():
    print('Iterations 1-3 (main seeds 1-10, confirmatory budget 8 n0)')
    for it, name in [(1, 'results.json'), (2, 'results_iter2.json'), (3, 'results_iter3.json')]:
        r = load(name)
        R = r['runs']
        ceil = dict(zip([str(i) for i in range(1, 11)], r['checks']['M2']['full_bss']['values']))
        report(f'Iteration {it}', scores(R['D_lam0']), scores(R['F_lam0']), scores(R['Fnoz']), ceil)
    r3 = load('results_iter3.json')['runs']
    for b in ['1', '2', '4']:
        report(f'Iteration 3 at {b} n0 (5 seeds)', scores(r3['D_n' + b]), scores(r3['F_n' + b]))
    first5 = {s: v for s, v in scores(r3['D_lam0']).items() if int(s) <= 5}
    report('Iteration 3 at 8 n0 (5 seeds)', first5, {s: v for s, v in scores(r3['F_lam0']).items() if int(s) <= 5})

    print('\nIteration 4 (fresh seeds 11-20)')
    freeze = load('config_iter4.freeze.json')
    with open(os.path.join(HERE, 'config_iter4.yaml'), 'rb') as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(f'config_iter4.yaml SHA-256 {digest} -> {"MATCHES" if digest == freeze["sha256"] else "DOES NOT MATCH"} '
          f'the value frozen at {freeze["timestamp"]}')
    r4 = load('results_iter4.json')['runs']
    for cond in ['R', 'G8', 'T32', 'G32']:
        C = r4[cond]
        for b in ['8', '1']:
            kd, kf, kn = f'D|{b}|0', f'F|{b}|0', f'Fnoz|{b}|0'
            if kd in C:
                report(f'{cond} at {b} n0', scores(C[kd]), scores(C[kf]), scores(C[kn]) if kn in C else None)

    runs5 = os.path.join(HERE, 'results_iter5_runs.jsonl')
    if os.path.exists(runs5):
        print('\nIteration 5 (seeds 11-20; T32 and G32 seeds 11-13 from iteration 4)')
        freeze5 = load('config_iter5.freeze.json')
        with open(os.path.join(HERE, 'config_iter5.yaml'), 'rb') as fh:
            digest5 = hashlib.sha256(fh.read()).hexdigest()
        print(f'config_iter5.yaml SHA-256 {digest5} -> {"MATCHES" if digest5 == freeze5["sha256"] else "DOES NOT MATCH"} '
              f'the value frozen at {freeze5["timestamp"]}')
        r5 = {}
        with open(runs5, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    cond, arch, seed = r['tag'].split('|')
                    r5.setdefault(cond, {}).setdefault(arch, {})[seed] = r['test']['bss']

        def merged(cond5, cond4, arch):
            out = dict(r5.get(cond5, {}).get(arch, {}))
            out.update(scores(r4[cond4].get(f'{arch}|8|0', {})))
            return out

        T32 = {a: merged('T32_ext', 'T32', a) for a in ('D', 'F', 'Fnoz')}
        G32 = {a: merged('G32_ext', 'G32', a) for a in ('D', 'F')}
        report('T32, ten seeds', T32['D'], T32['F'], T32['Fnoz'])
        report('G32, ten seeds', G32['D'], G32['F'])
        if 'T32_pre' in r5:
            report('T32_pre D vs T32 F', r5['T32_pre']['D'], T32['F'])
        if 'T32_R' in r5:
            report('T32_R', r5['T32_R']['D'], r5['T32_R']['F'])
        if 'T32_R_pre' in r5 and 'T32_R' in r5:
            report('T32_R_pre D vs T32_R F', r5['T32_R_pre']['D'], r5['T32_R']['F'])
        for cond in ('N32', 'T32_long', 'G32_long', 'R_audit'):
            for arch, vals in r5.get(cond, {}).items():
                print(f'{cond:<10} {arch:<4} ' + '  '.join(f'{s}: {v:+.3f}' for s, v in sorted(vals.items(), key=lambda t: int(t[0]))))


if __name__ == '__main__':
    main()
