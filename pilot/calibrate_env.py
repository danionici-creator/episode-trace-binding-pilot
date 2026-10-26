"""Environment calibration on the tuning seed only (no D/F model is trained here).

Prints M1, the M2 information ceiling and the leakage audit for the configuration in
config.yaml, optionally with dotted overrides, e.g.
    python pilot/calibrate_env.py measurement.sigma_z=0.2 environment.own.perceptual_noise=0.4
Used once, before the confirmatory pilot run, to place the M2 ceiling in a range that
is neither at chance nor saturated. Iterations are logged in REPORT.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checks import run_m2_and_leakage  # noqa: E402
from sim_env import simulate_life  # noqa: E402


def main():
    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8"))
    for arg in sys.argv[1:]:
        key, val = arg.split("=")
        node = cfg
        *path, leaf = key.split(".")
        for p in path:
            node = node[p]
        node[leaf] = yaml.safe_load(val)
    seed = cfg["seeds"]["tuning"]
    t0 = time.time()
    data, m1 = simulate_life(cfg, seed)
    t1 = time.time()
    m2 = run_m2_and_leakage(data, cfg, seed)
    print(f"simulate {t1 - t0:.1f}s, decoders {time.time() - t1:.1f}s")
    print("M1", json.dumps({k: m1[k] for k in ("delta_c", "median_permuted", "ratio", "pass")}))
    for k in ("full", "s_only", "no_z", "control", "misaligned"):
        r = m2[k]
        print(f"{k:11s} bss={r['bss']:+.3f} acc={r['accuracy']:.3f} ll={r['log_loss']:.3f}")
    print("route balance", np.bincount(data["y"].ravel(), minlength=3) / data["y"].size)


if __name__ == "__main__":
    main()
