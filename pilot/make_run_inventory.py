"""Write runs_inventory.csv: one row per trained self-model run in the four pilot iterations.

Usage:  python pilot/make_run_inventory.py
Standard library only. Walks results.json, results_iter2.json, results_iter3.json and
results_iter4.json and lists, for every run that has a test score: iteration, condition key,
architecture, misalignment lambda, training budget (multiples of n0), seed, and test BSS.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [(1, 'results.json'), (2, 'results_iter2.json'), (3, 'results_iter3.json'), (4, 'results_iter4.json')]


def walk(node, path, out, iteration):
    if isinstance(node, dict) and isinstance(node.get('test'), dict) and 'bss' in node['test']:
        out.append({
            'iteration': iteration,
            'condition': '/'.join(path[:-1]),
            'arch': node.get('arch', ''),
            'lambda': node.get('lam', ''),
            'budget_n0': node.get('budget', ''),
            'seed': node.get('seed', path[-1]),
            'test_bss': f"{node['test']['bss']:.6f}",
            'n_test_episodes': node['test'].get('n', ''),
        })
        return
    if isinstance(node, dict):
        for key, value in node.items():
            walk(value, path + [str(key)], out, iteration)


def main():
    rows = []
    for iteration, name in FILES:
        with open(os.path.join(HERE, name), encoding='utf-8') as fh:
            walk(json.load(fh)['runs'], [], rows, iteration)
    runs5 = os.path.join(HERE, 'results_iter5_runs.jsonl')   # iteration 5: one finished run per line
    if os.path.exists(runs5):
        with open(runs5, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    cond, arch, seed = r['tag'].split('|')
                    rows.append({'iteration': 5, 'condition': cond, 'arch': arch, 'lambda': r.get('lam', ''),
                                 'budget_n0': r.get('budget', ''), 'seed': seed, 'test_bss': f"{r['test']['bss']:.6f}",
                                 'n_test_episodes': r['test'].get('n', '')})
    target = os.path.join(HERE, 'runs_inventory.csv')
    with open(target, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    by_iter = {}
    for row in rows:
        by_iter[row['iteration']] = by_iter.get(row['iteration'], 0) + 1
    print(f'{len(rows)} runs written to runs_inventory.csv; by iteration: {by_iter}')


if __name__ == '__main__':
    main()
