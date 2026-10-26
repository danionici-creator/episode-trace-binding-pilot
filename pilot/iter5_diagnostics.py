"""Iteration 5, construct-validity diagnostics -> results_iter5_diagnostics.json.

    python pilot/iter5_diagnostics.py

Decoder-only analyses (no self-model is trained). On the iteration-4 lives with eight episodes per
window (seeds 11-20, re-simulated deterministically), the decoding-ceiling decoder of M2 is trained on
feature sets that remove parts of the source signal:

  full          z_t of every step, s_k, record, timestamp            (the M2 ceiling, as before)
  s_only        s_k, record, timestamp                               (what Phi retains)
  norm3         |mean_t z_t|, mean_t |z_t|, sd_t |z_t|               (magnitude only: three scalars)
  scalar5       norm3 plus the scalar mean and SD of all entries of z over the episode
  standardized  z_t of every step after subtracting the episode mean vector and dividing by the
                episode's scalar SD                                  (first- and second-order level removed)
  direction     z_t / |z_t| for every step                           (direction only, magnitude removed)

and, on a second life per seed in which the slow state is reset to its post-burn-in value before every
episode (removing all dependence on earlier episodes), the feature sets full and s_only.
Record, content kind and timestamp are appended to every set, as in M2; they carry no source information.
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

from checks import _time_feats, mlp_decoder  # noqa: E402
from iter4_common import condition_cfg, load_iter4  # noqa: E402
from run_pilot import _init_worker, describe  # noqa: E402
from scoring import score  # noqa: E402
from sim_env import Agent, Measurement, World, records_with_kind, role_sequence, simulate_life  # noqa: E402

SEEDS = list(range(11, 21))
SETS = ["full", "s_only", "norm3", "scalar5", "standardized", "direction"]


def simulate_reset_life(cfg, seed):
    """As sim_env.simulate_life (without M1), but h is reset to its post-burn-in value before every episode."""
    E, P = cfg["environment"], cfg["plasticity"]
    K, T, nk = E["episodes_per_window"], E["steps_per_episode"], E["n_kinds"]
    rng_env = np.random.default_rng([seed, 1])
    rng_xi = np.random.default_rng([seed, 2])
    rng_eta = np.random.default_rng([seed, 3])
    world = World(E, rng_env)
    agent = Agent(P, E["d_out"], E["d_in"])
    meas = Measurement(cfg["measurement"], 2 * E["d_out"] * E["d_in"])
    g = E["inferred"]["infer_weight"]
    for _ in range(E["burn_in_episodes"]):
        ep = world.episode(rng_env.choice(3, p=E["route_probs"]), rng_env.integers(nk), rng_env)
        agent.run_episode(ep, g, rng_xi)
    h0 = agent.state()
    roles = role_sequence(cfg, np.random.default_rng([seed, 5]))
    nw, dz = len(roles), cfg["measurement"]["d_z"]
    out = {"role": roles.astype(np.int8), "y": np.empty((nw, K), np.int8), "kind": np.empty((nw, K), np.int8),
           "rec": np.empty((nw, K, E["d_record"]), np.float32), "q": np.empty((nw, K, E["d_record"]), np.float32),
           "z": np.empty((nw, K * T, dz), np.float32), "s": np.empty((nw, K, 2 * dz), np.float32)}
    for w in range(nw):
        for k in range(K):
            route = rng_env.choice(3, p=E["route_probs"])
            kind = rng_env.integers(nk)
            ep = world.episode(route, kind, rng_env)
            agent.set_state(h0)
            z = meas.M(agent.run_episode(ep, g, rng_xi), P["eta"], rng_eta)
            out["y"][w, k], out["kind"][w, k] = route, kind
            out["rec"][w, k], out["q"][w, k] = ep["record"], ep["query"]
            out["z"][w, k * T:(k + 1) * T] = z
            out["s"][w, k] = meas.Phi(z)
    return out


def features(data, idx, cfg, fs):
    E = cfg["environment"]
    K, T, nk = E["episodes_per_window"], E["steps_per_episode"], E["n_kinds"]
    n = len(idx)
    z = data["z"][idx].reshape(n, K, T, -1).astype(np.float64)
    s = data["s"][idx]
    rec = records_with_kind(data["rec"][idx], data["kind"][idx], nk)
    pos = np.broadcast_to(_time_feats(np.arange(K) * T, cfg["self_model"]["time_freqs"], K * T),
                          (n, K, 2 * cfg["self_model"]["time_freqs"]))
    step_norm = np.linalg.norm(z, axis=-1)                                   # (n, K, T)
    norm3 = np.stack([np.linalg.norm(z.mean(2), axis=-1), step_norm.mean(-1), step_norm.std(-1)], -1)
    if fs == "full":
        parts = [z.reshape(n, K, -1), s]
    elif fs == "s_only":
        parts = [s]
    elif fs == "norm3":
        parts = [norm3]
    elif fs == "scalar5":
        flat = z.reshape(n, K, -1)
        parts = [norm3, flat.mean(-1, keepdims=True), flat.std(-1, keepdims=True)]
    elif fs == "standardized":
        zc = z - z.mean(2, keepdims=True)
        zc = zc / (zc.reshape(n, K, -1).std(-1)[..., None, None] + 1e-9)
        parts = [zc.reshape(n, K, -1)]
    elif fs == "direction":
        parts = [(z / (step_norm[..., None] + 1e-9)).reshape(n, K, -1)]
    else:
        raise ValueError(fs)
    X = np.concatenate(parts + [rec, pos], -1).reshape(n * K, -1).astype(np.float32)
    return X, data["y"][idx].reshape(-1).astype(np.int64)


def decode_all(data, cfg, seed, sets):
    tr, te = np.flatnonzero(data["role"] == 3), np.flatnonzero(data["role"] == 2)
    base = np.bincount(data["y"][tr].ravel(), minlength=3) / data["y"][tr].size
    out = {}
    for fs in sets:
        Xtr, ytr = features(data, tr, cfg, fs)
        Xte, yte = features(data, te, cfg, fs)
        out[fs] = score(mlp_decoder(Xtr, ytr, Xte, cfg, seed), yte, base, cfg["checks"]["ece_bins"])
    return out


def task(cfg, seed):
    t0 = time.time()
    data, _ = simulate_life(cfg, seed, with_m1=False)
    normal = decode_all(data, cfg, seed, SETS)
    reset = decode_all(simulate_reset_life(cfg, seed), cfg, seed, ["full", "s_only"])
    return {"seed": seed, "normal": normal, "reset": reset, "seconds": time.time() - t0}


def main():
    c4 = load_iter4()
    cfg = condition_cfg(c4, "R")
    nw = int(os.environ.get("DIAG_WORKERS", "2"))
    out_path = HERE / "results_iter5_diagnostics.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"seeds": {}}
    todo = [s for s in SEEDS if str(s) not in res["seeds"]]
    t0 = time.time()
    with ProcessPoolExecutor(nw, initializer=_init_worker, initargs=(1,)) as ex:
        futs = [ex.submit(task, cfg, s) for s in todo]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res["seeds"][str(r["seed"])] = r
            out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(f"  [diag] {i}/{len(todo)} seed {r['seed']} full {r['normal']['full']['bss']:+.3f} "
                  f"norm3 {r['normal']['norm3']['bss']:+.3f} reset {r['reset']['full']['bss']:+.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    seeds = sorted(res["seeds"], key=int)
    summary = {}
    for cond in ("normal", "reset"):
        for fs in res["seeds"][seeds[0]][cond]:
            v = [res["seeds"][s][cond][fs]["bss"] for s in seeds]
            full = [res["seeds"][s]["normal"]["full"]["bss"] for s in seeds]
            summary[f"{cond}|{fs}"] = {"bss": describe(v), "share_of_full": describe(np.array(v) / np.array(full))}
    res["summary"] = summary
    res["recorded_ceiling_iter4_k8"] = json.loads((HERE / "results_iter4.json").read_text(encoding="utf-8"))["checks"]["k8"]["M2_ceiling"]
    out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in summary.items():
        print(f"{k:22s} BSS {v['bss']['mean']:+.3f} ± {v['bss']['sd']:.3f}   share of full {v['share_of_full']['mean']:.2f}")


if __name__ == "__main__":
    main()
