"""Supplementary analyses of the existing pilot results -> results_supplementary.json (no model is trained).

    python pilot/supplementary_analyses.py

Needs only the Python standard library and NumPy. Reports, from the per-run results of iterations 1-4:
  A  full statistics for Table 7 (SDs, 95% and (1 - 2 alpha) intervals, df, shares of the ceiling);
  B  the misalignment conditions of iteration 3 with SDs and intervals;
  C  the classification of the K = 8 contrasts under margins from 0.05 down to 0.02 BSS (descriptive;
     the pre-specified margin remains 0.05);
  D  control competence (M4) at 0.50, 0.75 and 0.90 of the decoding ceiling, per seed;
  E  the binomial false-positive rate of the per-seed leakage accuracy criterion;
  F  the onset of learning (first evaluation step with validation BSS >= 0.05) and the step at which
     90% of the final validation score is reached, for D and F;
  G  the iteration-1 data range, parameter counts, and the baseline offsets of Table 5.
"""
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reproduce_manuscript_tables import ALPHA, t_quantile  # noqa: E402
MARGINS = [0.05, 0.04, 0.03, 0.025, 0.02]


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return json.load(fh)


def bss(block):
    return {int(s): r["test"]["bss"] for s, r in block.items()}


def stats(x):
    x = list(x)
    n, m = len(x), st.mean(x)
    sd = st.stdev(x) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    return {"n": n, "df": n - 1, "mean": m, "sd": sd,
            "ci95": [m - t_quantile(0.975, n - 1) * se, m + t_quantile(0.975, n - 1) * se],
            "ci_1m2alpha": [m - t_quantile(1 - ALPHA, n - 1) * se, m + t_quantile(1 - ALPHA, n - 1) * se]}


def contrast(D, F, ceil=None):
    seeds = sorted(set(D) & set(F))
    out = {"D": stats(D[s] for s in seeds), "F": stats(F[s] for s in seeds),
           "D_minus_F": stats(D[s] - F[s] for s in seeds)}
    if ceil:
        out["F_share"] = stats(F[s] / ceil[s] for s in seeds)
        out["D_share"] = stats(D[s] / ceil[s] for s in seeds)
    return out


def classify(ci, m):
    lo, hi = ci
    if lo > -m and hi < m:
        return "equivalent" + (" (excludes 0)" if lo > 0 or hi < 0 else "")
    if hi < -m:
        return "relevant disadvantage for D"
    if lo > m:
        return "relevant advantage for D"
    return "not equivalent (interval crosses the margin)"


def onset(curve, final, thr=0.05):
    first = next((s for s, _, v in curve if v >= thr), None)
    reach = next((s for s, _, v in curve if v >= 0.9 * final), None) if final > thr else None
    return first, reach


def main():
    out = {}
    r1, r3, r4 = load("results.json"), load("results_iter3.json"), load("results_iter4.json")
    R4, chk4 = r4["runs"], r4["checks"]
    ceil8 = dict(zip(chk4["k8"]["seeds"], chk4["k8"]["M2_ceiling"]["values"]))
    ceil32 = dict(zip(chk4["k32"]["seeds"], chk4["k32"]["M2_ceiling"]["values"]))
    ceil3 = dict(zip(range(1, 11), r3["checks"]["M2"]["full_bss"]["values"]))

    # A. Table 7, complete
    A = {}
    for cond, ceil in (("R", ceil8), ("G8", ceil8), ("T32", ceil32), ("G32", ceil32)):
        for b in ("8", "1"):
            if f"D|{b}|0" in R4[cond]:
                A[f"{cond} at {b} n0"] = contrast(bss(R4[cond][f"D|{b}|0"]), bss(R4[cond][f"F|{b}|0"]), ceil)
                if cond in ("T32", "G32"):
                    A[f"{cond} at {b} n0"]["per_seed"] = {s: {"D": bss(R4[cond]["D|8|0"])[s], "F": bss(R4[cond]["F|8|0"])[s]}
                                                         for s in sorted(bss(R4[cond]["D|8|0"]))}
    out["A_table7"] = A

    # B. iteration-3 misalignment conditions
    R3 = r3["runs"]
    B = {k: stats(bss(R3[k]).values()) for k in ("D_lam0", "D_lam0.5", "D_lam1", "F_lam0", "F_lam0.5", "F_lam1", "Fnoz")}
    B["D_test_misbound"] = stats(r["test_misbound"]["bss"] for r in R3["D_lam0"].values())
    B["note"] = ("D_lambda: the permutation is drawn once per training window and stays attached to each episode "
                 "when windows are recomposed (not redrawn at every presentation); record descriptors are redrawn.")
    out["B_misalignment_iter3"] = B

    # C. margin sensitivity (descriptive)
    C = {}
    sets = {"iteration 3 at 8 n0": (bss(R3["D_lam0"]), bss(R3["F_lam0"])),
            "R at 8 n0": (bss(R4["R"]["D|8|0"]), bss(R4["R"]["F|8|0"])),
            "G8 at 8 n0": (bss(R4["G8"]["D|8|0"]), bss(R4["G8"]["F|8|0"])),
            "R at 1 n0": (bss(R4["R"]["D|1|0"]), bss(R4["R"]["F|1|0"])),
            "G8 at 1 n0": (bss(R4["G8"]["D|1|0"]), bss(R4["G8"]["F|1|0"]))}
    for name, (D, F) in sets.items():
        s = stats(D[k] - F[k] for k in sorted(D))
        C[name] = {"mean": s["mean"], "ci_1m2alpha": s["ci_1m2alpha"], "ci_1m2alpha_4dp": [round(v, 4) for v in s["ci_1m2alpha"]],
                   **{f"margin {m}": classify(s["ci_1m2alpha"], m) for m in MARGINS}}
    out["C_margin_sensitivity"] = C

    # D. M4 at several fractions of the ceiling, per seed
    Dm = {}
    for name, (Dd, Ff, ceil) in {"iteration 3": (bss(R3["D_lam0"]), bss(R3["F_lam0"]), ceil3),
                                 "R": (bss(R4["R"]["D|8|0"]), bss(R4["R"]["F|8|0"]), ceil8),
                                 "G8": (bss(R4["G8"]["D|8|0"]), bss(R4["G8"]["F|8|0"]), ceil8)}.items():
        fs = {s: Ff[s] / ceil[s] for s in Ff}
        ds = {s: Dd[s] / ceil[s] for s in Dd}
        Dm[name] = {"F_share": stats(fs.values()), "D_share": stats(ds.values()),
                    "F_min_share": min(fs.values()), "D_min_share": min(ds.values()),
                    **{f"seeds passing {fr}": {"F": sum(v >= fr for v in fs.values()), "D": sum(v >= fr for v in ds.values()),
                                               "n": len(fs)} for fr in (0.5, 0.75, 0.9)}}
    out["D_m4_sensitivity"] = Dm

    # E. leakage criterion: binomial false-positive rate under no leakage (chance 1/3, 2,048 test episodes)
    n_test = 2048
    thr = 1 / 3 + 0.02
    k0 = math.floor(thr * n_test) + 1
    p = 1 / 3
    tail = sum(math.exp(math.lgamma(n_test + 1) - math.lgamma(k + 1) - math.lgamma(n_test - k + 1)
                        + k * math.log(p) + (n_test - k) * math.log(1 - p)) for k in range(k0, n_test + 1))
    out["E_leakage_binomial"] = {"n_test": n_test, "threshold_accuracy": thr, "min_correct_to_exceed": k0,
                                 "per_seed_false_positive": tail, "familywise_10_seeds": 1 - (1 - tail) ** 10,
                                 "seed13_correct": round(0.356 * n_test),
                                 "binomial_se": math.sqrt(p * (1 - p) / n_test)}

    # F. onset of learning and time to 90% of the final validation score
    Fo = {}
    for name, blk in {"iteration 1 D": R3 and r1["runs"]["D_lam0"], "iteration 1 F": r1["runs"]["F_lam0"],
                      "iteration 3 D": R3["D_lam0"], "iteration 3 F": R3["F_lam0"],
                      "R D": R4["R"]["D|8|0"], "R F": R4["R"]["F|8|0"],
                      "G8 D": R4["G8"]["D|8|0"], "G8 F": R4["G8"]["F|8|0"]}.items():
        firsts, reaches, finals = [], [], []
        for r in blk.values():
            fin = r.get("val_bss", max(v for _, _, v in r["curve"]))
            a, b = onset(r["curve"], fin)
            firsts.append(a)
            reaches.append(b)
            finals.append(fin)
        ok = [x for x in firsts if x is not None]
        okr = [x for x in reaches if x is not None]
        Fo[name] = {"n": len(firsts), "n_onset": len(ok), "onset_step_median": st.median(ok) if ok else None,
                    "onset_steps": firsts, "step_to_90pct_median": st.median(okr) if okr else None,
                    "final_val_mean": st.mean(finals), "steps_total": max(s for s, _, _ in next(iter(blk.values()))["curve"])}
    out["F_onset_k8"] = Fo

    # G. iteration-1 data range, parameter counts, baseline offsets
    R1 = r1["runs"]
    out["G_misc"] = {
        "iteration1_learning_curve": {k: stats(bss(R1[k]).values())["mean"] for k in ("D_n1", "D_n4", "D_lam0", "D_n16",
                                                                                      "F_n1", "F_n4", "F_lam0", "F_n16", "F_long")},
        "iteration1_budgets_n0": "n1 = 1 n0, n4 = 4 n0, lam0 = 8 n0, n16 = 16 n0; F_long = 4x the steps at 8 n0 (5 seeds)",
        "params": {c: r4["conditions"][c]["params"] if "params" in r4["conditions"][c] else None for c in r4["conditions"]},
        "iteration3_no_z_decoder_bss": r3["checks"]["M2"].get("no_z_bss", {}).get("mean") if isinstance(r3["checks"]["M2"].get("no_z_bss"), dict) else None,
        "iteration3_Fnoz_mean": stats(bss(R3["Fnoz"]).values())["mean"],
    }
    with open(os.path.join(HERE, "results_supplementary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("A. Table 7")
    for k, v in A.items():
        d = v["D_minus_F"]
        print(f"  {k:12s} D {v['D']['mean']:.3f}±{v['D']['sd']:.3f}  F {v['F']['mean']:.3f}±{v['F']['sd']:.3f}  "
              f"D-F {d['mean']:+.3f} SD {d['sd']:.3f} df {d['df']} 95% [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
              f"(1-2a) [{d['ci_1m2alpha'][0]:+.3f}, {d['ci_1m2alpha'][1]:+.3f}]  F share {v['F_share']['mean']:.2f}  D share {v['D_share']['mean']:.2f}")
    print("B. Iteration-3 misalignment")
    for k, v in B.items():
        if isinstance(v, dict):
            print(f"  {k:16s} {v['mean']:+.3f} ± {v['sd']:.3f}  95% [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("C. Margin sensitivity ((1-2a) interval, 4 dp)")
    for k, v in C.items():
        print(f"  {k:20s} {v['mean']:+.3f} {v['ci_1m2alpha_4dp']}  " + "; ".join(f"{m}: {v[f'margin {m}']}" for m in MARGINS))
    print("D. M4 sensitivity")
    for k, v in Dm.items():
        print(f"  {k:12s} F share {v['F_share']['mean']:.3f} (min {v['F_min_share']:.3f})  D share {v['D_share']['mean']:.3f} (min {v['D_min_share']:.3f})  "
              + "  ".join(f"{fr}: F {v[f'seeds passing {fr}']['F']}/{v[f'seeds passing {fr}']['n']} D {v[f'seeds passing {fr}']['D']}"
                          for fr in (0.5, 0.75, 0.9)))
    e = out["E_leakage_binomial"]
    print(f"E. Leakage: P(exceed | no leakage) per seed {e['per_seed_false_positive']:.4f}; over 10 seeds {e['familywise_10_seeds']:.3f}; "
          f"needs >= {e['min_correct_to_exceed']} correct of {n_test}; seed 13 ~{e['seed13_correct']}")
    print("F. Onset (first val BSS >= 0.05) and step to 90% of final, medians")
    for k, v in Fo.items():
        print(f"  {k:14s} onset {v['n_onset']}/{v['n']} median step {v['onset_step_median']}  90% at {v['step_to_90pct_median']}  "
              f"final val {v['final_val_mean']:.3f}  (of {v['steps_total']} steps)")
    print("G.", json.dumps(out["G_misc"], indent=None)[:900])


if __name__ == "__main__":
    main()
