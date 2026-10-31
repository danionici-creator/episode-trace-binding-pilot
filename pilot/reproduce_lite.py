"""One command for the three levels of verification described in README.md.

    python pilot/reproduce_lite.py             # all three levels (levels 2 and 3 need PyTorch)
    python pilot/reproduce_lite.py --tables    # level 1 only: Python and NumPy

Level 1 recomputes every contrast of the reported tables from the saved per-run results and checks the
SHA-256 of the frozen configurations (reproduce_manuscript_tables.py). Level 2 runs the 20 automated checks
of the pipeline on a small life of an unused seed (audit_checks.py, about a minute). Level 3 trains D and F
with tiny budgets on two seeds of iteration 1 (run_pilot.py --quick, a few minutes, output in quick/). No
result file of the study is written. The environment is fixed by requirements.lock:

    pip install --require-hashes -r pilot/requirements.lock

The exit code is the number of levels that failed. These levels check internal consistency, the logic of the
pipeline, and that training runs end to end; they do not reproduce the reported scores, which needs the full
runs (hours of CPU time) and the thread settings recorded in README.md.
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = [
    ("1  tables from saved results", ["reproduce_manuscript_tables.py"], False),
    ("2  audit checks", ["audit_checks.py"], True),
    ("3  smoke training of D and F", ["run_pilot.py", "--quick"], True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true", help="level 1 only (no PyTorch needed)")
    args = ap.parse_args()
    has_torch = importlib.util.find_spec("torch") is not None
    failed = 0
    for name, cmd, needs_torch in LEVELS[:1] if args.tables else LEVELS:
        if needs_torch and not has_torch:
            print(f"SKIP  level {name}: PyTorch is not installed (pip install --require-hashes -r pilot/requirements.lock)")
            failed += 1
            continue
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], cwd=HERE.parent)
        ok = proc.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  level {name}  ({time.time() - t0:.0f} s, exit code {proc.returncode})", flush=True)
        failed += not ok
    sys.exit(failed)


if __name__ == "__main__":
    main()
