"""Iteration 5, full re-run: per-episode Brier scores and a reproduction check of all 100 runs.

    python pilot/iter5_rerun_episodes.py

The first run of iteration 5 stored per-run scores only. This script re-runs every run of the frozen
configuration with the same lives, seeds, and thread settings, saves the per-episode Brier scores to
cache/per_episode_brier_iter5.npz (as for iterations 1-4), and compares every re-run test score with the
recorded one in results_iter5_runs.jsonl -> results_iter5_rerun_check.json. It changes no recorded result.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iter4_common import load_iter4  # noqa: E402
from iter5_run import CONF, check_freezes, cond_archs, cond_cfg, cond_seeds, dims_of, hp_of, load_iter5, read_runs  # noqa: E402
from run_pilot import _init_worker  # noqa: E402
from self_models import matched_widths  # noqa: E402
from sim_env import window_layout  # noqa: E402
from train_sm import run_job  # noqa: E402


def main():
    c4, c5 = load_iter4(), load_iter5()
    check_freezes(c5)
    recorded = read_runs(HERE / c5["meta"]["output_runs"])
    lives = json.loads((HERE / "results_iter5_lives.json").read_text(encoding="utf-8"))
    jobs = []
    for name in c5["conditions"]:
        cfg = cond_cfg(c4, c5, name)
        widths = matched_widths(dims_of(cfg), cfg["self_model"], window_layout(cfg))
        k = "k32" if cfg["environment"]["episodes_per_window"] == 32 else "k8"
        for arch in cond_archs(c5, name):
            for s in cond_seeds(c5, name, arch):
                path = HERE / lives[f"{k}|{s}"]["path"]
                assert path.exists(), f"missing life {path}; run iter5_run.py first to re-simulate it"
                jobs.append(({"seed": s, "arch": arch, "lam": 0.0, "budget": CONF, "tag": f"{name}|{arch}|{s}",
                              "hp": hp_of(c4, c5, name, arch), "data_path": str(path)}, cfg, widths))
    nw, nt = c5["meta"]["n_workers"], c5["meta"]["threads_per_worker"]
    items, rows, t0 = {}, [], time.time()
    with ProcessPoolExecutor(nw, initializer=_init_worker, initargs=(nt,)) as ex:
        futs = [ex.submit(run_job, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            items[r["tag"].replace("|", "__")] = r["brier_items"]
            rec = recorded[r["tag"]]["test"]["bss"]
            rows.append({"tag": r["tag"], "rerun": r["test"]["bss"], "recorded": rec, "abs_diff": abs(r["test"]["bss"] - rec)})
            print(f"  [rerun] {i}/{len(jobs)} {r['tag']} |diff| {rows[-1]['abs_diff']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(HERE / "cache" / "per_episode_brier_iter5.npz", **items)
    out = {"n_runs": len(rows), "max_abs_diff": max(r["abs_diff"] for r in rows),
           "n_exact": sum(r["abs_diff"] == 0 for r in rows), "seconds": time.time() - t0,
           "rows": sorted(rows, key=lambda r: r["tag"])}
    (HERE / "results_iter5_rerun_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{out['n_runs']} runs; exact {out['n_exact']}; max |diff| {out['max_abs_diff']:.3e}")


if __name__ == "__main__":
    main()
