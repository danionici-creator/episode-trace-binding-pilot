"""Iteration 3: screen the admissible candidates on the tuning seed and pick the strongest F.

    python pilot/iter3_candidates.py

Candidates (config_iter3.yaml, all with the label-free co-timed objective, applied identically to D,
F and F_noz): a+c, a+b+c, a+b+c+d. For each, the step budget S* comes from the plateau rule on a
doubling ladder (best validation BSS gains < tol for both D and F); F_noz is trained at S*.
M4 (strengthened): F >= m4_ceiling_fraction x the M2 ceiling AND F - F_noz >= m4_margin, judged on
the validation windows, where selection also happens; test values are reported, never used.
Selection: highest F validation BSS among the candidates that meet M4 (the strongest otherwise, with
M4 reported as failed). No main seed is simulated or touched. Output: candidates_iter3.json.
"""
import json
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iter2_candidates import candidate_cfg, dims_of  # noqa: E402
from run_pilot import run_parallel, simulate_task  # noqa: E402
from self_models import matched_widths  # noqa: E402
from sim_env import window_layout  # noqa: E402
from train_sm import run_job  # noqa: E402


def job(cfg, name, arch, steps, data_path):
    hp = cfg["candidates"]["screening_hp"]["F" if arch == "Fnoz" else arch]
    return {"seed": cfg["seeds"]["tuning"], "arch": arch, "lam": 0.0, "budget": cfg["data"]["confirmatory_budget"],
            "steps": steps, "tag": f"{name}|{steps}|{arch}", "hp": hp, "data_path": data_path}


def main():
    base = yaml.safe_load((HERE / "config_iter3.yaml").read_text(encoding="utf-8"))
    nw, nt = base["meta"]["n_workers"], base["meta"]["threads_per_worker"]
    C, K = base["candidates"], base["checks"]
    tseed, lad, tol = base["seeds"]["tuning"], C["step_ladder"], C["plateau_tolerance"]
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    out_path = HERE / base["meta"]["candidates_file"]
    t_start = time.time()

    # tuning-seed life with M1 / M2: the information ceiling on the same seed and protocol
    sim = simulate_task(base, tseed, str(cache), True)
    m2 = sim["m2"]
    ceiling = {"val": m2["full_val"]["bss"], "test": m2["full"]["bss"]}
    log = {"format": "iter3", "seed": tseed, "order": C["order"], "ceiling": ceiling,
           "m4_margin": K["m4_margin"], "m4_ceiling_fraction": K["m4_ceiling_fraction"],
           "m4_set": "val", "plateau_tolerance": tol, "candidates": [], "m1": sim["m1"],
           "generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    cfgs = {n: candidate_cfg(base, n, lad[0]) for n in C["order"]}
    widths = {n: matched_widths(dims_of(c), c["self_model"], window_layout(c)) for n, c in cfgs.items()}
    for n, w_ in widths.items():
        assert abs(w_["rel_mismatch"]) <= base["self_model"]["max_param_mismatch"], n

    # step ladder, two rungs at a time, extended only for candidates that have not plateaued
    results = {n: {} for n in C["order"]}            # name -> steps -> arch -> result
    s_star = {}
    pending = {n: lad[:2] for n in C["order"]}
    while pending:
        arg_list = [(job(cfgs[n], n, a, S, sim["path"]), candidate_cfg(base, n, S), widths[n])
                    for n, rungs in pending.items() for S in rungs for a in ("D", "F")]
        arg_list.sort(key=lambda t: -t[0]["steps"])
        for r in run_parallel(run_job, arg_list, nw, nt, "ladder"):
            n, S, a = r["tag"].split("|")
            results[n].setdefault(int(S), {})[a] = r
        pending = {}
        for n in C["order"]:
            if n in s_star:
                continue
            done = sorted(results[n])
            for lo, hi in zip(done[:-1], done[1:]):
                if all(results[n][hi][a]["val_bss"] - results[n][lo][a]["val_bss"] < tol for a in ("D", "F")):
                    s_star[n] = (lo, True)
                    break
            else:
                nxt = [S for S in lad if S > done[-1]]
                if nxt:
                    pending[n] = nxt[:1]
                else:
                    s_star[n] = (done[-1], False)

    # F_noz at each candidate's S*, then M4 and selection
    arg_list = [(job(cfgs[n], n, "Fnoz", s_star[n][0], sim["path"]), candidate_cfg(base, n, s_star[n][0]), widths[n])
                for n in C["order"]]
    for r in run_parallel(run_job, arg_list, nw, nt, "F_noz"):
        n, S, a = r["tag"].split("|")
        results[n][int(S)][a] = r

    for n in C["order"]:
        S, plateau = s_star[n]
        rr = results[n][S]
        ent = {"name": n, "steps": S, "plateau_reached": plateau, "params": widths[n],
               "rungs": [{"steps": s_, **{f"{a}_val": results[n][s_][a]["val_bss"] for a in ("D", "F")},
                          **{f"{a}_test": results[n][s_][a]["test"]["bss"] for a in ("D", "F")}}
                         for s_ in sorted(results[n])]}
        for split in ("val", "test"):
            get = (lambda a: rr[a]["val_bss"]) if split == "val" else (lambda a: rr[a]["test"]["bss"])
            ent[split] = {a: get(a) for a in ("D", "F", "Fnoz")}
            ent[split]["F_minus_Fnoz"] = get("F") - get("Fnoz")
            ent[split]["F_share_of_ceiling"] = get("F") / ceiling[split]
            ent[split]["D_share_of_ceiling"] = get("D") / ceiling[split]
            ent[split]["m4_pass"] = bool(ent[split]["F_minus_Fnoz"] >= K["m4_margin"]
                                         and ent[split]["F_share_of_ceiling"] >= K["m4_ceiling_fraction"])
        ent["m4_pass"] = ent["val"]["m4_pass"]
        ent["best_step"] = {a: max(rr[a]["curve"], key=lambda c: c[2])[0] for a in rr}
        ent["curves"] = {a: [(s_, round(v, 4)) for s_, _, v in rr[a]["curve"]] for a in ("D", "F")}
        log["candidates"].append(ent)

    admitted = [c for c in log["candidates"] if c["m4_pass"]]
    pool = admitted or log["candidates"]
    best = max(pool, key=lambda c: c["val"]["F"])
    log["selected"] = best["name"]
    log["selected_steps"] = best["steps"]
    log["m4_met_by_selected"] = bool(best["m4_pass"])
    log["selection_rule"] = ("highest F validation BSS among candidates meeting M4" if admitted else
                             "no candidate meets M4: strongest F by validation BSS, M4 reported as failed")
    log["wall_s"] = time.time() - t_start
    out_path.write_text(json.dumps(log, indent=1, default=float), encoding="utf-8")
    Path(sim["path"]).unlink(missing_ok=True)
    for c in log["candidates"]:
        print(f"{c['name']:8s} S*={c['steps']:5d} plateau={c['plateau_reached']}  "
              f"val D {c['val']['D']:+.3f} F {c['val']['F']:+.3f} Fnoz {c['val']['Fnoz']:+.3f} "
              f"share {c['val']['F_share_of_ceiling']:.2f} M4 {c['m4_pass']} | "
              f"test D {c['test']['D']:+.3f} F {c['test']['F']:+.3f}", flush=True)
    print("ceiling", ceiling, "selected", log["selected"], log["selection_rule"], flush=True)


if __name__ == "__main__":
    main()
