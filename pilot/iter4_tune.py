"""Iteration 4: tune G8, T32 and G32 on the tuning seed (1000) only -> tuning_iter4.json.

    python pilot/iter4_tune.py

1. tuning-seed lives for K = 8 and K = 32, with the M2 information ceiling (validation and test);
2. step ladder with the screening hyperparameters for D and F, extended rung by rung until the
   plateau rule holds or the cap is reached -> S* per condition;
3. identical grid for D and F at S* (learning rate x auxiliary weight, weight decay fixed); the
   ladder run at S* is reused as the grid point with the screening values; F_noz at S*;
4. M4 on validation with the selected F: F >= 0.5 x ceiling and F - F_noz >= 0.05.
No main seed is simulated or touched.
"""
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iter4_common import condition_cfg, hp_for, load_iter4  # noqa: E402
from run_pilot import run_parallel, simulate_task  # noqa: E402
from self_models import matched_widths  # noqa: E402
from sim_env import window_layout  # noqa: E402
from train_sm import run_job  # noqa: E402

TUNED = ["G8", "T32", "G32"]


def dims_of(cfg):
    E = cfg["environment"]
    return {"rec": E["d_record"] + E["n_kinds"], "z": cfg["measurement"]["d_z"], "s": 2 * cfg["measurement"]["d_z"]}


def main():
    c4 = load_iter4()
    tu = c4["tuning"]
    tseed = c4["seeds"]["tuning"]
    t_start = time.time()
    base = condition_cfg(c4, "R")
    nw, nt = base["meta"]["n_workers"], base["meta"]["threads_per_worker"]
    out = {"seed": tseed, "conditions": {}, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path = HERE / c4["meta"]["tuning_file"]

    # 1. tuning-seed lives (K = 8, K = 32) with the M2 ceiling on the same seed and protocol
    lives = {}
    for k_name, cond in (("k8", "G8"), ("k32", "T32")):
        cache = HERE / "cache" / f"iter4_tune_{k_name}"
        cache.mkdir(parents=True, exist_ok=True)
        lives[k_name] = (condition_cfg(c4, cond, 1200), str(cache))
    sims = run_parallel(simulate_task, [(cfg, tseed, cache, True) for cfg, cache in lives.values()], 2, nt, "lives")
    by_k = {}
    for r in sims:
        k_name = "k32" if "k32" in r["path"] else "k8"
        by_k[k_name] = r
    out["ceiling"] = {k: {"val": r["m2"]["full_val"]["bss"], "test": r["m2"]["full"]["bss"]} for k, r in by_k.items()}
    out["m1_ratio"] = {k: r["m1"]["ratio"] for k, r in by_k.items()}
    path_of = {c: by_k["k32" if c4["conditions"][c]["k32"] else "k8"]["path"] for c in TUNED}
    cfgs = {c: condition_cfg(c4, c, 1200) for c in TUNED}
    widths = {c: matched_widths(dims_of(cfg), cfg["self_model"], window_layout(cfg)) for c, cfg in cfgs.items()}
    for c, w_ in widths.items():
        assert abs(w_["rel_mismatch"]) <= cfgs[c]["self_model"]["max_param_mismatch"], c

    def job(c, arch, steps, hp):
        return ({"seed": tseed, "arch": arch, "lam": 0.0, "budget": cfgs[c]["data"]["confirmatory_budget"],
                 "steps": steps, "tag": f"{c}|{steps}|{arch}|{json.dumps(hp, sort_keys=True)}", "hp": hp,
                 "data_path": path_of[c]}, condition_cfg(c4, c, steps), widths[c])

    def parse(r):
        c, S, a, hp = r["tag"].split("|", 3)
        return c, int(S), a, json.loads(hp)

    # 2. step ladder with the screening values
    scr = {c: hp_for(c4, c, tu["screening_hp"]["lr"], tu["screening_hp"]["aux"]) for c in TUNED}
    lad, tol = tu["step_ladder"], tu["plateau_tolerance"]
    ladder = {c: {} for c in TUNED}
    s_star, pending = {}, {c: lad[:2] for c in TUNED}
    t0 = time.time()
    while pending:
        args = [job(c, a, S, scr[c]) for c, rungs in pending.items() for S in rungs for a in ("D", "F")]
        args.sort(key=lambda t: -t[0]["steps"])
        for r in run_parallel(run_job, args, nw, nt, "ladder"):
            c, S, a, _ = parse(r)
            ladder[c].setdefault(S, {})[a] = r
        pending = {}
        for c in TUNED:
            if c in s_star:
                continue
            done = sorted(ladder[c])
            for lo, hi in zip(done[:-1], done[1:]):
                if all(ladder[c][hi][a]["val_bss"] - ladder[c][lo][a]["val_bss"] < tol for a in ("D", "F")):
                    s_star[c] = (lo, True)
                    break
            else:
                nxt = [S for S in lad if S > done[-1]]
                if nxt:
                    pending[c] = nxt[:1]
                else:
                    s_star[c] = (done[-1], False)
    t_ladder = time.time() - t0

    # 3. grid at S* (the screening point is reused from the ladder) and F_noz at S*
    args = []
    for c in TUNED:
        S = s_star[c][0]
        for lr, aux in itertools.product(tu["grid"]["lr"], tu["grid"]["aux"]):
            hp = hp_for(c4, c, lr, aux)
            if hp != scr[c]:
                args += [job(c, a, S, hp) for a in ("D", "F")]
        args.append(job(c, "Fnoz", S, scr[c]))
    args.sort(key=lambda t: -t[0]["steps"] * (2 if c4["conditions"][t[0]["tag"].split("|")[0]]["masked"] else 1))
    t0 = time.time()
    res = run_parallel(run_job, args, nw, nt, "grid")
    t_grid = time.time() - t0
    rows = {c: {"D": [], "F": []} for c in TUNED}
    fnoz = {}
    for c in TUNED:
        S = s_star[c][0]
        for a in ("D", "F"):
            r = ladder[c][S][a]
            rows[c][a].append({"hp": scr[c], "val_bss": r["val_bss"], "test_bss": r["test"]["bss"], "seconds": r["seconds"]})
    for r in res:
        c, S, a, hp = parse(r)
        if a == "Fnoz":
            fnoz[c] = r
        else:
            rows[c][a].append({"hp": hp, "val_bss": r["val_bss"], "test_bss": r["test"]["bss"], "seconds": r["seconds"]})

    # 4. selection and M4 (validation)
    for c in TUNED:
        k = "k32" if c4["conditions"][c]["k32"] else "k8"
        sel = {a: max(rows[c][a], key=lambda x: x["val_bss"]) for a in ("D", "F")}
        ce = out["ceiling"][k]
        f_val, n_val = sel["F"]["val_bss"], fnoz[c]["val_bss"]
        m4 = {"F_share_of_ceiling_val": f_val / ce["val"], "F_minus_Fnoz_val": f_val - n_val,
              "F_share_of_ceiling_test": sel["F"]["test_bss"] / ce["test"],
              "F_minus_Fnoz_test": sel["F"]["test_bss"] - fnoz[c]["test"]["bss"]}
        m4["pass"] = bool(m4["F_share_of_ceiling_val"] >= tu["m4_ceiling_fraction"]
                          and m4["F_minus_Fnoz_val"] >= tu["m4_margin"])
        out["conditions"][c] = {
            "k": k, "steps": s_star[c][0], "plateau_reached": s_star[c][1], "params": widths[c],
            "ladder": [{"steps": S, **{f"{a}_val": ladder[c][S][a]["val_bss"] for a in ("D", "F")},
                        **{f"{a}_test": ladder[c][S][a]["test"]["bss"] for a in ("D", "F")},
                        "seconds": {a: ladder[c][S][a]["seconds"] for a in ("D", "F")}} for S in sorted(ladder[c])],
            "grid": rows[c], "selected_hp": {a: sel[a]["hp"] for a in ("D", "F")},
            "selected_val": {a: sel[a]["val_bss"] for a in ("D", "F")},
            "selected_test": {a: sel[a]["test_bss"] for a in ("D", "F")},
            "Fnoz": {"val": n_val, "test": fnoz[c]["test"]["bss"], "seconds": fnoz[c]["seconds"]},
            "m4": m4}
    out["timing_s"] = {"ladder": t_ladder, "grid": t_grid, "total": time.time() - t_start}
    out_path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for c in TUNED:
        o = out["conditions"][c]
        print(f"{c}: S*={o['steps']} plateau={o['plateau_reached']} hp D {o['selected_hp']['D']} F {o['selected_hp']['F']} "
              f"val D {o['selected_val']['D']:+.3f} F {o['selected_val']['F']:+.3f} Fnoz {o['Fnoz']['val']:+.3f} "
              f"share {o['m4']['F_share_of_ceiling_val']:.2f} M4 {o['m4']['pass']}", flush=True)
    print("ceiling", out["ceiling"], "timing", {k: round(v / 60, 1) for k, v in out["timing_s"].items()}, flush=True)


if __name__ == "__main__":
    main()
