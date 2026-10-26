"""Iteration 2: M4 candidate search on the tuning seed only.

    python pilot/iter2_candidates.py          # stop at the first candidate that passes M4
    python pilot/iter2_candidates.py --all    # also screen the later candidates (information only)

Candidates are tried in the preregistered order of config_iter2.yaml (a, a+b, a+b+c, a+b+c+d).
Every change is applied identically to D and F (and F_noz). M4 on the tuning seed:
BSS(F) - BSS(F_noz) >= m4_margin on the tuning seed's held-out test episodes. No main seed is
simulated or touched here. Results are written to candidates_iter2.json after every candidate.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_pilot import run_parallel, simulate_task  # noqa: E402
from self_models import matched_widths  # noqa: E402
from sim_env import window_layout  # noqa: E402
from train_sm import run_job  # noqa: E402


def candidate_cfg(base, name, steps):
    """Cumulative application of the candidate settings named in `name` (e.g. 'a+b+c')."""
    cfg = copy.deepcopy(base)
    C = base["candidates"]
    parts = name.split("+")
    cfg["training"]["steps"] = steps                      # a
    if "b" in parts:
        cfg["training"]["curriculum"] = C["b"]["curriculum"]
    if "c" in parts:
        cfg["training"]["aux_cotimed_weight"] = C["c"]["aux_cotimed_weight"]
        cfg["self_model"]["aux_head"] = True
    if "d" in parts:
        cfg["self_model"].update({k: C["d"][k] for k in ("d_model", "n_heads", "enc_layers", "d_ff_D")})
    return cfg


def dims_of(cfg):
    E = cfg["environment"]
    return {"rec": E["d_record"] + E["n_kinds"], "z": cfg["measurement"]["d_z"], "s": 2 * cfg["measurement"]["d_z"]}


def jobs_for(cfg, name, steps, data_path):
    hp = cfg["candidates"]["screening_hp"]
    conf = cfg["data"]["confirmatory_budget"]
    return [{"seed": cfg["seeds"]["tuning"], "arch": a, "lam": 0.0, "budget": conf, "steps": steps,
             "tag": f"{name}|{steps}|{a}", "hp": hp["F" if a == "Fnoz" else a], "data_path": data_path}
            for a in ("D", "F", "Fnoz")]


def summarise(res, cfg):
    by = {r["arch"]: r for r in res}
    key = cfg["candidates"]["m4_set"]
    f_test, n_test = by["F"]["test"]["bss"], by["Fnoz"]["test"]["bss"]
    return {
        "val_bss": {a: by[a]["val_bss"] for a in by}, "test_bss": {a: by[a]["test"]["bss"] for a in by},
        "m4_test": f_test - n_test, "m4_val": by["F"]["val_bss"] - by["Fnoz"]["val_bss"],
        "m4": (f_test - n_test) if key == "test" else by["F"]["val_bss"] - by["Fnoz"]["val_bss"],
        "D_minus_F_test": by["D"]["test"]["bss"] - f_test,
        "best_step": {a: max(by[a]["curve"], key=lambda c: c[2])[0] for a in by},
        "curves": {a: [(s, round(v, 4)) for s, _, v in by[a]["curve"]] for a in ("D", "F")},
        "seconds": {a: by[a]["seconds"] for a in by},
    }


def reproduce_iteration1(cfg1, nw, nt):
    """D on the iteration-1 tuning-seed life (config.yaml) must reproduce results.json exactly."""
    tseed = cfg1["seeds"]["tuning"]
    tmp = HERE / "cache" / "iter1"
    tmp.mkdir(parents=True, exist_ok=True)
    path = simulate_task(cfg1, tseed, str(tmp), False)["path"]
    w1 = matched_widths(dims_of(cfg1), cfg1["self_model"], window_layout(cfg1))
    R1 = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    hp1 = R1["tuning"]["selected"]["D"]
    job = {"seed": tseed, "arch": "D", "lam": 0.0, "budget": cfg1["data"]["confirmatory_budget"],
           "tag": "repro_iter1", "hp": hp1, "data_path": path}
    r = run_parallel(run_job, [(job, cfg1, w1)], 1, nt, "repro")[0]
    ref = next(t for t in R1["tuning"]["grid_results"] if t["arch"] == "D" and t["hp"] == hp1)
    Path(path).unlink(missing_ok=True)
    return {"val_bss": r["val_bss"], "reference_val_bss": ref["val_bss"], "test_bss": r["test"]["bss"],
            "reference_test_bss": ref["test_bss_tuning_seed"], "match": bool(abs(r["val_bss"] - ref["val_bss"]) < 1e-6),
            "note": "iteration-1 life simulated with config.yaml; iteration-2 lives differ because the window-role "
                    "sequence depends on the largest learning-curve budget (16 n0 in iteration 1, 8 n0 here)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--repro-only", action="store_true", help="redo the iteration-1 reproduction check")
    ap.add_argument("--screen-rest", action="store_true",
                    help="screen the candidates after the selected one (information only)")
    args = ap.parse_args()
    base = yaml.safe_load((HERE / "config_iter2.yaml").read_text(encoding="utf-8"))
    cfg1 = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    nw, nt = base["meta"]["n_workers"], base["meta"]["threads_per_worker"]
    margin = base["checks"]["m4_margin"]
    tseed = base["seeds"]["tuning"]
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    out_path = HERE / base["meta"]["candidates_file"]
    log = {"seed": tseed, "m4_margin": margin, "order": base["candidates"]["order"], "candidates": [],
           "generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save():
        out_path.write_text(json.dumps(log, indent=1, default=float), encoding="utf-8")

    if args.repro_only or args.screen_rest:
        log = json.loads(out_path.read_text(encoding="utf-8"))
        if args.repro_only:
            log["iteration1_reproduction"] = reproduce_iteration1(cfg1, nw, nt)
            save()
            print("reproduction:", log["iteration1_reproduction"], flush=True)
        if args.screen_rest:
            data_path = simulate_task(base, tseed, str(cache), False)["path"]
            s_star = log["candidates"][0]["steps"]
            done = {c["name"] for c in log["candidates"]}
            todo = [n for n in base["candidates"]["order"] if n not in done]
            arg_list, widths = [], {}
            for name in todo:
                cfg = candidate_cfg(base, name, s_star)
                widths[name] = matched_widths(dims_of(cfg), cfg["self_model"], window_layout(cfg))
                assert abs(widths[name]["rel_mismatch"]) <= cfg["self_model"]["max_param_mismatch"]
                arg_list += [(j, cfg, widths[name]) for j in jobs_for(cfg, name, s_star, data_path)]
            res = run_parallel(run_job, arg_list, nw, nt, "screen")
            for name in todo:
                cfg = candidate_cfg(base, name, s_star)
                summ = summarise([r for r in res if r["tag"].startswith(f"{name}|")], cfg)
                log["candidates"].append({"name": name, "steps": s_star, **summ,
                                          "pass": bool(summ["m4"] >= margin), "params": widths[name],
                                          "screened_after_selection": True})
                print(f"{name}: M4={summ['m4']:+.3f} (information only)", flush=True)
            log["best_m4"] = max(c["m4"] for c in log["candidates"])
            save()
        return

    log["iteration1_reproduction"] = reproduce_iteration1(cfg1, nw, nt)
    sim = simulate_task(base, tseed, str(cache), False)
    data_path = sim["path"]

    # candidate a: step ladder
    lad = base["candidates"]["a"]["step_ladder"]
    tol = base["candidates"]["a"]["plateau_tolerance"]
    wa = matched_widths(dims_of(base), base["self_model"], window_layout(base))
    jobs = [(j, candidate_cfg(base, "a", S), wa) for S in lad for j in jobs_for(base, "a", S, data_path)]
    jobs.sort(key=lambda t: -t[0]["steps"])
    t0 = time.time()
    res = run_parallel(run_job, jobs, nw, nt, "candidate a")
    rungs = []
    for S in lad:
        summ = summarise([r for r in res if r["tag"].startswith(f"a|{S}|")], base)
        rungs.append({"steps": S, **summ})
    s_star, plateau = lad[-1], False
    for lo, hi in zip(rungs[:-1], rungs[1:]):
        if all(hi["val_bss"][a] - lo["val_bss"][a] < tol for a in ("D", "F")):
            s_star, plateau = lo["steps"], True
            break
    sel = next(r for r in rungs if r["steps"] == s_star)
    log["candidates"].append({"name": "a", "rungs": rungs, "plateau_steps": s_star, "plateau_reached": plateau,
                              "steps": s_star, "m4": sel["m4"], "m4_test": sel["m4_test"], "m4_val": sel["m4_val"],
                              "pass": bool(sel["m4"] >= margin), "params": wa, "wall_s": time.time() - t0})
    save()
    print(f"a: S*={s_star} (plateau {plateau}) M4={sel['m4']:+.3f} pass={sel['m4'] >= margin}", flush=True)
    selected = "a" if sel["m4"] >= margin else None

    for name in base["candidates"]["order"][1:]:
        if selected and not args.all:
            break
        cfg = candidate_cfg(base, name, s_star)
        wc = matched_widths(dims_of(cfg), cfg["self_model"], window_layout(cfg))
        assert abs(wc["rel_mismatch"]) <= cfg["self_model"]["max_param_mismatch"]
        t0 = time.time()
        res = run_parallel(run_job, [(j, cfg, wc) for j in jobs_for(cfg, name, s_star, data_path)], nw, nt, name)
        summ = summarise(res, cfg)
        ok = summ["m4"] >= margin
        log["candidates"].append({"name": name, "steps": s_star, **summ, "pass": bool(ok), "params": wc,
                                  "screened_after_selection": selected is not None, "wall_s": time.time() - t0})
        save()
        print(f"{name}: M4={summ['m4']:+.3f} pass={ok}", flush=True)
        if ok and selected is None:
            selected = name
    log["selected"] = selected
    log["best_m4"] = max(c["m4"] for c in log["candidates"])
    save()
    print("selected:", selected, flush=True)


if __name__ == "__main__":
    main()
