"""Iteration 4, tuning amendment (tuning seed only): the plateau rule requires that learning has begun.

    python pilot/iter4_extend.py

The literal plateau rule (best validation BSS gains < 0.01 for both D and F when steps double) also
fires when D and F are both still at the floor, before any learning (T32: both ~0.007 at 1200 and
2400 steps). Amended rule, applied to every tuned condition: a rung pair counts as a plateau only if D
or F already reaches validation BSS >= SESOI at the lower rung; otherwise the ladder continues to its
cap. For conditions whose S* changes, the grid and F_noz are rerun at the new S* and M4 is re-judged.
The literal-rule result is kept in tuning_iter4.json under "literal_rule".
"""
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iter4_common import condition_cfg, hp_for, load_iter4  # noqa: E402
from iter4_tune import TUNED, dims_of  # noqa: E402
from run_pilot import run_parallel  # noqa: E402
from self_models import matched_widths  # noqa: E402
from sim_env import window_layout  # noqa: E402
from train_sm import run_job  # noqa: E402


def amended_s_star(ladder, tol, onset, cap):
    rungs = ladder
    for lo, hi in zip(rungs[:-1], rungs[1:]):
        learned = max(lo["D_val"], lo["F_val"]) >= onset
        if learned and all(hi[f"{a}_val"] - lo[f"{a}_val"] < tol for a in ("D", "F")):
            return lo["steps"], True
    return cap, False


def main():
    c4 = load_iter4()
    tu, tseed = c4["tuning"], c4["seeds"]["tuning"]
    t_path = HERE / c4["meta"]["tuning_file"]
    T = json.loads(t_path.read_text(encoding="utf-8"))
    base = condition_cfg(c4, "R")
    nw, nt = base["meta"]["n_workers"], base["meta"]["threads_per_worker"]
    onset, cap = base["meta"]["sesoi_bss"], tu["step_ladder"][-1]
    t0 = time.time()
    T["amendment"] = {"rule": "plateau only once D or F has validation BSS >= SESOI at the lower rung; else continue to the cap",
                      "onset_bss": onset, "cap": cap, "changed": []}
    redo = {}
    for c in TUNED:
        o = T["conditions"][c]
        S, plateau = amended_s_star(o["ladder"], tu["plateau_tolerance"], onset, cap)
        if S != o["steps"]:
            redo[c] = S
            T["amendment"]["changed"].append({"condition": c, "literal_S*": o["steps"], "amended_S*": S})
    for c, S in redo.items():
        cfg = condition_cfg(c4, c, S)
        w = matched_widths(dims_of(cfg), cfg["self_model"], window_layout(cfg))
        path = str(HERE / "cache" / f"iter4_tune_{'k32' if c4['conditions'][c]['k32'] else 'k8'}" / f"seed_{tseed}.npz")
        scr = hp_for(c4, c, tu["screening_hp"]["lr"], tu["screening_hp"]["aux"])
        args = []
        for lr, aux in itertools.product(tu["grid"]["lr"], tu["grid"]["aux"]):
            hp = hp_for(c4, c, lr, aux)
            for a in ("D", "F"):
                args.append(({"seed": tseed, "arch": a, "lam": 0.0, "budget": cfg["data"]["confirmatory_budget"],
                              "steps": S, "tag": f"{a}|{json.dumps(hp, sort_keys=True)}", "hp": hp, "data_path": path}, cfg, w))
        args.append(({"seed": tseed, "arch": "Fnoz", "lam": 0.0, "budget": cfg["data"]["confirmatory_budget"],
                      "steps": S, "tag": f"Fnoz|{json.dumps(scr, sort_keys=True)}", "hp": scr, "data_path": path}, cfg, w))
        res = run_parallel(run_job, args, nw, nt, f"{c} at {S}")
        o = T["conditions"][c]
        o["literal_rule"] = {k: o[k] for k in ("steps", "plateau_reached", "grid", "selected_hp", "selected_val",
                                               "selected_test", "Fnoz", "m4")}
        rows = {"D": [], "F": []}
        fn = None
        for r in res:
            a, hp = r["tag"].split("|", 1)
            if a == "Fnoz":
                fn = r
            else:
                rows[a].append({"hp": json.loads(hp), "val_bss": r["val_bss"], "test_bss": r["test"]["bss"],
                                "seconds": r["seconds"]})
        scr_rows = {a: next(x for x in rows[a] if x["hp"] == scr) for a in ("D", "F")}
        o["ladder"].append({"steps": S, **{f"{a}_val": scr_rows[a]["val_bss"] for a in ("D", "F")},
                            **{f"{a}_test": scr_rows[a]["test_bss"] for a in ("D", "F")},
                            "seconds": {a: scr_rows[a]["seconds"] for a in ("D", "F")}})
        sel = {a: max(rows[a], key=lambda x: x["val_bss"]) for a in ("D", "F")}
        ce = T["ceiling"][o["k"]]
        m4 = {"F_share_of_ceiling_val": sel["F"]["val_bss"] / ce["val"],
              "F_minus_Fnoz_val": sel["F"]["val_bss"] - fn["val_bss"],
              "F_share_of_ceiling_test": sel["F"]["test_bss"] / ce["test"],
              "F_minus_Fnoz_test": sel["F"]["test_bss"] - fn["test"]["bss"]}
        m4["pass"] = bool(m4["F_share_of_ceiling_val"] >= tu["m4_ceiling_fraction"] and m4["F_minus_Fnoz_val"] >= tu["m4_margin"])
        o.update(steps=S, plateau_reached=False, grid=rows, selected_hp={a: sel[a]["hp"] for a in ("D", "F")},
                 selected_val={a: sel[a]["val_bss"] for a in ("D", "F")},
                 selected_test={a: sel[a]["test_bss"] for a in ("D", "F")},
                 Fnoz={"val": fn["val_bss"], "test": fn["test"]["bss"], "seconds": fn["seconds"]}, m4=m4)
        print(f"{c}: amended S*={S} hp D {o['selected_hp']['D']} F {o['selected_hp']['F']} val D {o['selected_val']['D']:+.3f} "
              f"F {o['selected_val']['F']:+.3f} Fnoz {fn['val_bss']:+.3f} share {m4['F_share_of_ceiling_val']:.2f} "
              f"M4 {m4['pass']}", flush=True)
    T["amendment"]["seconds"] = time.time() - t0
    T["timing_s"]["total"] += T["amendment"]["seconds"]
    t_path.write_text(json.dumps(T, indent=1, default=float), encoding="utf-8")
    print("changed:", T["amendment"]["changed"], flush=True)


if __name__ == "__main__":
    main()
