"""Iteration 4 main run on fresh seeds 11-20 -> results_iter4.json, REPORT_iter4.md.

    python pilot/iter4_run.py                # checks the freeze record, then runs every allocated condition
    python pilot/iter4_run.py --report-only  # re-render REPORT_iter4.md from results_iter4.json

The run refuses to start unless config_iter4.yaml still has the SHA-256 recorded at freezing.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iter4_common import condition_cfg, load_iter4, sha256  # noqa: E402
from run_pilot import _ci, _ms, decision, describe, paired, run_parallel, simulate_task, within_seed_se  # noqa: E402
from self_models import matched_widths  # noqa: E402
from sim_env import window_layout  # noqa: E402
from train_sm import run_job  # noqa: E402

COST = {(False, False): 1.0, (False, True): 1.8, (True, False): 1.55, (True, True): 2.6}   # measured per-step ratios


def dims_of(cfg):
    E = cfg["environment"]
    return {"rec": E["d_record"] + E["n_kinds"], "z": cfg["measurement"]["d_z"], "s": 2 * cfg["measurement"]["d_z"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    c4 = load_iter4()
    res_path, rep_path = HERE / c4["meta"]["output_results"], HERE / c4["meta"]["output_report"]
    if args.report_only:
        R = json.loads(res_path.read_text(encoding="utf-8"))
        rep_path.write_text(render(R), encoding="utf-8")
        return
    freeze = json.loads((HERE / "config_iter4.freeze.json").read_text(encoding="utf-8"))
    assert sha256(HERE / "config_iter4.yaml") == freeze["sha256"], "config_iter4.yaml changed after freezing"
    t_start = time.time()
    alloc = c4["allocation"]
    conds = [c for c in c4["meta"]["priority"] if alloc[c]["seeds"] > 0]
    cfgs = {c: condition_cfg(c4, c) for c in conds}
    widths = {c: matched_widths(dims_of(cfg), cfg["self_model"], window_layout(cfg)) for c, cfg in cfgs.items()}
    seeds = {c: c4["seeds"]["main"][:alloc[c]["seeds"]] for c in conds}
    nw, nt = cfgs[conds[0]]["meta"]["n_workers"], cfgs[conds[0]]["meta"]["threads_per_worker"]
    conf = cfgs[conds[0]]["data"]["confirmatory_budget"]

    # 1. fresh lives, one per seed and window length, with M1, M2 and the decoder leakage audit
    lives = {}
    for k32, rep in ((False, "R"), (True, "T32")):
        need = sorted({s for c in conds if c4["conditions"][c]["k32"] == k32 for s in seeds[c]})
        if need:
            cache = HERE / "cache" / f"iter4_{'k32' if k32 else 'k8'}"
            cache.mkdir(parents=True, exist_ok=True)
            lives[k32] = (condition_cfg(c4, rep), str(cache), need)
    t0 = time.time()
    sims = run_parallel(simulate_task, [(cfg, s, cache, True) for cfg, cache, need in lives.values() for s in need],
                        nw, nt, "lives")
    sim = {(Path(r["path"]).parent.name == "iter4_k32", r["seed"]): r for r in sims}
    t_sim = time.time() - t0

    # 2. self-model runs
    jobs = []
    for c in conds:
        k32 = c4["conditions"][c]["k32"]
        hp = c4["conditions"][c]["hp"]
        for s in seeds[c]:
            base = {"seed": s, "lam": 0.0, "data_path": sim[(k32, s)]["path"]}
            for arch, budget, lam in ([("D", conf, 0.0), ("F", conf, 0.0), ("Fnoz", conf, 0.0)]
                                      + [(a, b, 0.0) for b in alloc[c].get("budgets", c4["outputs"]["budgets"])
                                         if b != conf for a in ("D", "F")]
                                      + ([(a, conf, 1.0) for a in ("D", "F")] if c == "T32" and alloc[c].get("lambda") else [])):
                jobs.append(({**base, "arch": arch, "budget": budget, "lam": lam, "tag": f"{c}|{arch}|{budget}|{lam:g}",
                              "hp": hp["F" if arch == "Fnoz" else arch]}, cfgs[c], widths[c]))
    jobs.sort(key=lambda t: -t[1]["training"]["steps"] * COST[(t[1]["environment"]["episodes_per_window"] > 8,
                                                                t[1]["training"].get("aux_objective") == "masked")])
    print(f"{len(jobs)} self-model runs over {conds}", flush=True)
    t0 = time.time()
    res = run_parallel(run_job, jobs, nw, nt, "main")
    t_main = time.time() - t0
    runs = {}
    for r in res:
        c, arch, budget, lam = r["tag"].split("|")
        runs.setdefault(c, {}).setdefault(f"{arch}|{budget}|{lam}", {})[r["seed"]] = r
    np.savez(HERE / "cache" / "per_episode_brier_iter4.npz",
             **{f"{c}__{k}__seed{s}": r["brier_items"] for c, rr in runs.items() for k, ss in rr.items() for s, r in ss.items()})

    # 3. analysis per condition
    sesoi = cfgs[conds[0]]["meta"]["sesoi_bss"]
    rng = np.random.default_rng(12345)
    out = {}
    for c in conds:
        cfg, rr, sd = cfgs[c], runs[c], seeds[c]
        k32 = c4["conditions"][c]["k32"]
        col = lambda key, m="bss": [rr[key][s]["test"][m] for s in sd]  # noqa: E731
        ceiling = np.array([sim[(k32, s)]["m2"]["full"]["bss"] for s in sd])
        D, F, N = np.array(col(f"D|{conf}|0")), np.array(col(f"F|{conf}|0")), np.array(col(f"Fnoz|{conf}|0"))
        share = F / ceiling
        m4m, frac = c4["tuning"]["m4_margin"], c4["tuning"]["m4_ceiling_fraction"]
        dmf = paired(D, F, cfg, rng)
        dmf["within_seed_test_se"] = within_seed_se([rr[f"D|{conf}|0"][s] for s in sd], [rr[f"F|{conf}|0"][s] for s in sd])
        o = {"seeds": sd, "steps": cfg["training"]["steps"], "hp": c4["conditions"][c]["hp"], "params": widths[c],
             "D": describe(D), "F": describe(F), "Fnoz": describe(N),
             "secondary": {m: {"D": describe(col(f"D|{conf}|0", m)), "F": describe(col(f"F|{conf}|0", m))}
                           for m in ("log_loss", "accuracy", "ece")},
             "M4": {"F_share_of_ceiling": describe(share), "F_minus_Fnoz": describe(F - N),
                    "n_pass_both": int(np.sum((share >= frac) & (F - N >= m4m))),
                    "pass": bool(share.mean() >= frac and (F - N).mean() >= m4m)},
             "ceiling": describe(ceiling), "D_minus_F": dmf, "decision": decision(dmf, sesoi),
             "pipeline_leakage_pass": bool(N.mean() <= cfg["checks"]["leakage_margin"])}
        for b in alloc[c].get("budgets", c4["outputs"]["budgets"]):
            if b != conf:
                Db, Fb = col(f"D|{b}|0"), col(f"F|{b}|0")
                pb = paired(Db, Fb, cfg, rng)
                o[f"n{b}"] = {"D": describe(Db), "F": describe(Fb), "D_minus_F": pb, "decision": decision(pb, sesoi)}
        if f"D|{conf}|1" in rr:
            D1, F1 = col(f"D|{conf}|1"), col(f"F|{conf}|1")
            o["lambda1"] = {"D": describe(D1), "F": describe(F1), "D1_minus_F0": paired(D1, F, cfg, rng),
                            "D1_minus_D0": paired(D1, D, cfg, rng), "F1_minus_F0": paired(F1, F, cfg, rng)}
        out[c] = o
    checks = {}
    for k32 in lives:
        ss = lives[k32][2]
        m = [sim[(k32, s)] for s in ss]
        checks["k32" if k32 else "k8"] = {
            "seeds": ss, "M1_ratio": describe([x["m1"]["ratio"] for x in m]), "M1_pass": all(x["m1"]["pass"] for x in m),
            "M2_ceiling": describe([x["m2"]["full"]["bss"] for x in m]),
            "M2_delta": describe([x["m2"]["delta_bss"] for x in m]),
            "M2_pass": all(x["m2"]["delta_bss"] >= sesoi for x in m),
            "decoder_no_z_bss": describe([x["m2"]["no_z"]["bss"] for x in m]),
            "decoder_no_z_accuracy": describe([x["m2"]["no_z"]["accuracy"] for x in m]),
            "decoder_leakage_pass": all(x["m2"]["no_z"]["bss"] <= 0.02 and x["m2"]["no_z"]["accuracy"] <= 1 / 3 + 0.02
                                        for x in m)}
    import torch
    R = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "freeze": freeze, "base_config": c4["base_config"],
         "base_config_sha256": c4["base_config_sha256"], "config": c4, "allocation": alloc, "conditions": out,
         "checks": checks, "tuning": json.loads((HERE / c4["meta"]["tuning_file"]).read_text(encoding="utf-8")),
         "timing_s": {"lives": t_sim, "main": t_main, "total": time.time() - t_start},
         "environment_info": {"python": platform.python_version(), "torch": torch.__version__, "n_workers": nw},
         "runs": {c: {k: {str(s): {kk: v for kk, v in r.items() if kk not in ("brier_items", "brier_ref_items")}
                          for s, r in ss.items()} for k, ss in rr.items()} for c, rr in runs.items()}}
    res_path.write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    R = json.loads(res_path.read_text(encoding="utf-8"))
    rep_path.write_text(render(R), encoding="utf-8")
    for k32, (cfg, cache, need) in lives.items():
        for s in need:
            Path(sim[(k32, s)]["path"]).unlink(missing_ok=True)
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {res_path}, {rep_path}", flush=True)


# -------------------------------------------------------------------------------- report
LABEL = {"R": "R: replication of iteration 3 (co-timed objective, K = 8)",
         "G8": "G8: generic masked-token objective, K = 8",
         "T32": "T32: co-timed objective, K = 32", "G32": "G32: generic objective, K = 32"}


def _reading(dec):
    return dec["decision"]


def render(R):
    c4, out, ck, tu = R["config"], R["conditions"], R["checks"], R["tuning"]
    sesoi = c4["outputs"]["equivalence_bound"]
    conds = [c for c in c4["meta"]["priority"] if c in out]
    dropped = [c for c in c4["meta"]["priority"] if c not in out]
    L = []
    w = L.append
    w("# Phase I simulation pilot (iteration 4, fresh seeds) — Study 1, *Binding Episodes to Their Internal Effects*\n")
    w(f"**Frozen configuration:** `config_iter4.yaml`, SHA-256 `{R['freeze']['sha256']}`, frozen "
      f"{R['freeze']['timestamp']}, before any of seeds {c4['seeds']['main'][0]}–{c4['seeds']['main'][-1]} was simulated. "
      f"Base (R) configuration: `{R['base_config']}`, SHA-256 `{R['base_config_sha256']}`.\n")
    w(f"Generated {R['generated']} by `python pilot/iter4_run.py` ({R['timing_s']['total'] / 60:.0f} min for the main run "
      f"on {R['environment_info']['n_workers']} CPU workers; tuning on seed 1000 took "
      f"{tu['timing_s']['total'] / 60:.0f} min; Python {R['environment_info']['python']}, torch "
      f"{R['environment_info']['torch']}). Means ± SD across seeds, 95% t-intervals; (1 − 2α) intervals with "
      f"α = 0.05/3 for the equivalence reading against ±{sesoi}. Full numbers: `results_iter4.json`.\n")
    alloc_txt = "; ".join(f"{c} {R['allocation'][c]['seeds']} seeds"
                          + (f" (budgets {', '.join(str(b) + ' n0' for b in R['allocation'][c]['budgets'])})"
                             if "budgets" in R["allocation"][c] else "") for c in c4["meta"]["priority"])
    w(f"Allocation: {alloc_txt}"
      + (f"; T32 λ ∈ {{0, 1}} {'included' if R['allocation']['T32'].get('lambda') else 'dropped'}" if "T32" in R["allocation"] else "")
      + (f"; dropped entirely: {', '.join(dropped)}" if dropped else "") + ".\n")

    w("## Bottom line\n")
    if "R" in out:
        o = out["R"]
        w(f"- **Replication (R, fresh seeds):** D − F = {o['D_minus_F']['mean']:+.3f} {_ci(o['D_minus_F'])}, (1 − 2α) CI "
          f"{_ci(o['D_minus_F'], 'ci_1m2alpha')} → **{_reading(o['decision'])}**. "
          + ("M4 fails, so the comparison is uninformative." if not o["M4"]["pass"] else
             "The iteration-3 equivalence replicates." if o["decision"]["equivalent"] else
             "The iteration-3 equivalence does NOT replicate as such.")
          + f" M4 {'passes' if o['M4']['pass'] else 'FAILS'} (F at {o['M4']['F_share_of_ceiling']['mean']:.2f} of the ceiling).")
    if "G8" in out:
        o = out["G8"]
        w(f"- **Generic objective (G8):** F reaches {o['M4']['F_share_of_ceiling']['mean']:.2f} of the ceiling, F − F_noz "
          f"{o['M4']['F_minus_Fnoz']['mean']:+.3f}; M4 {'passes' if o['M4']['pass'] else 'FAILS'}. D − F = "
          f"{o['D_minus_F']['mean']:+.3f} {_ci(o['D_minus_F'])}, (1 − 2α) {_ci(o['D_minus_F'], 'ci_1m2alpha')} → "
          f"**{_reading(o['decision'])}**.")
    for c in ("T32", "G32"):
        if c in out:
            o = out[c]
            n1 = o.get("n1")
            learned = lambda x: "learns" if x["mean"] >= sesoi else "stays at the floor"  # noqa: E731
            w(f"- **{c} (K = 32, {len(o['seeds'])} seeds):** D {o['D']['mean']:+.3f} ({learned(o['D'])}), F "
              f"{o['F']['mean']:+.3f} ({learned(o['F'])}); M4 {'passes' if o['M4']['pass'] else 'FAILS'} (F at "
              f"{o['M4']['F_share_of_ceiling']['mean']:.2f} of the ceiling). At 8 n0, D − F = {o['D_minus_F']['mean']:+.3f} "
              f"{_ci(o['D_minus_F'])} → **{_reading(o['decision'])}**"
              + (f"; at 1 n0, D − F = {n1['D_minus_F']['mean']:+.3f} {_ci(n1['D_minus_F'])} → **{_reading(n1['decision'])}**"
                 if n1 else "") + ".")
    n1s = [c for c in conds if "n1" in out[c]]
    if n1s:
        w("- **Sample efficiency (1 n0):** " + "; ".join(
            f"{c} D − F = {out[c]['n1']['D_minus_F']['mean']:+.3f} {_ci(out[c]['n1']['D_minus_F'])} "
            f"({_reading(out[c]['n1']['decision'])})" for c in n1s)
          + (". D has no small-data advantage; F is slightly better, within ±SESOI."
             if all(out[c]["n1"]["D_minus_F"]["mean"] < 0 and out[c]["n1"]["decision"]["equivalent"] for c in n1s) else "."))
    sup = [c for c in conds if out[c]["decision"]["decision"] == "support"
           or out[c].get("n1", {}).get("decision", {}).get("decision") == "support"]
    valid = [c for c in sup if out[c]["M4"]["pass"]]
    invalid = [c for c in sup if not out[c]["M4"]["pass"]]
    w(f"- **Does any condition restore an advantage for D against a competent control?** "
      + (f"Yes: {', '.join(valid)}." if valid else "No.")
      + (f" {', '.join(invalid)} reaches the support reading only because F fails M4 (F stays at the floor), so it is "
         f"not a test against a competent control." if invalid else ""))
    w("")

    w("## Results by condition (test BSS, novel episodes)\n")
    w("| Condition | Seeds | Steps | D | F | F_noz | F share of ceiling | D − F, 95% CI | SD of D − F | (1 − 2α) CI | Reading |\n"
      "|---|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        o = out[c]
        w(f"| {LABEL[c]} | {len(o['seeds'])} | {o['steps']} | {_ms(o['D'])} | {_ms(o['F'])} | {_ms(o['Fnoz'])} | "
          f"{o['M4']['F_share_of_ceiling']['mean']:.2f} | {o['D_minus_F']['mean']:+.3f} {_ci(o['D_minus_F'])} | "
          f"{o['D_minus_F']['sd']:.3f} | {_ci(o['D_minus_F'], 'ci_1m2alpha')} | {_reading(o['decision'])} |")
    w("\n**Sample efficiency, 1 n0 (1,024 labelled training episodes; same steps):**\n")
    w("| Condition | D | F | D − F, 95% CI | SD of D − F | (1 − 2α) CI | Reading |\n|---|---|---|---|---|---|---|")
    for c in conds:
        n1 = out[c].get("n1")
        if n1:
            w(f"| {c} | {_ms(n1['D'])} | {_ms(n1['F'])} | {n1['D_minus_F']['mean']:+.3f} {_ci(n1['D_minus_F'])} | "
              f"{n1['D_minus_F']['sd']:.3f} | {_ci(n1['D_minus_F'], 'ci_1m2alpha')} | {_reading(n1['decision'])} |")
    w("\n**M4 (strengthened) on the main seeds:**\n")
    w("| Condition | F share of ceiling | F − F_noz | Both met (seeds) | M4 |\n|---|---|---|---|---|")
    for c in conds:
        m = out[c]["M4"]
        w(f"| {c} | {_ms(m['F_share_of_ceiling'])} | {_ms(m['F_minus_Fnoz'])} | {m['n_pass_both']}/{len(out[c]['seeds'])} | "
          f"{'PASS' if m['pass'] else 'FAIL'} |")
    if any("lambda1" in out[c] for c in conds):
        w("\n**T32 misalignment (λ = 1 at training, correct pairing at test):**\n")
        lo = out["T32"]["lambda1"]
        w(f"D₁ {_ms(lo['D'])}, F₁ {_ms(lo['F'])}; D₁ − D₀ {lo['D1_minus_D0']['mean']:+.3f} {_ci(lo['D1_minus_D0'])}, "
          f"F₁ − F₀ {lo['F1_minus_F0']['mean']:+.3f} {_ci(lo['F1_minus_F0'])}, D₁ − F₀ {lo['D1_minus_F0']['mean']:+.3f} "
          f"{_ci(lo['D1_minus_F0'])}.")
    w("\nSecondary outcomes at 8 n0 (means): " + "; ".join(
        f"{c}: log loss D {out[c]['secondary']['log_loss']['D']['mean']:.3f} / F {out[c]['secondary']['log_loss']['F']['mean']:.3f}, "
        f"accuracy {out[c]['secondary']['accuracy']['D']['mean']:.3f} / {out[c]['secondary']['accuracy']['F']['mean']:.3f}, "
        f"ECE {out[c]['secondary']['ece']['D']['mean']:.3f} / {out[c]['secondary']['ece']['F']['mean']:.3f}" for c in conds) + ".\n")

    w("## Manipulation checks on the fresh seeds\n")
    w("| Lives | Seeds | M1 distance ratio | M2 ceiling | M2 Δ (with − without z) | Decoder leakage BSS / accuracy | "
      "Pipeline leakage (F_noz) |\n|---|---|---|---|---|---|---|")
    for k, v in ck.items():
        pipe = ", ".join(f"{c} {out[c]['Fnoz']['mean']:+.3f}" for c in conds if (c4["conditions"][c]["k32"]) == (k == "k32"))
        bad = [(sd, b, a) for sd, b, a in zip(v["seeds"], v["decoder_no_z_bss"]["values"], v["decoder_no_z_accuracy"]["values"])
               if b > 0.02 or a > 1 / 3 + 0.02]
        why = "" if not bad else " — " + "; ".join(
            f"seed {sd}: accuracy {a:.4f} > {1 / 3 + 0.02:.4f}, BSS {b:+.4f}" for sd, b, a in bad)
        w(f"| K = {32 if k == 'k32' else 8} | {len(v['seeds'])} | {_ms(v['M1_ratio'])} ({'PASS' if v['M1_pass'] else 'FAIL'}) | "
          f"{_ms(v['M2_ceiling'])} | {_ms(v['M2_delta'])} ({'PASS' if v['M2_pass'] else 'FAIL'}) | "
          f"{v['decoder_no_z_bss']['mean']:+.3f} / {v['decoder_no_z_accuracy']['mean']:.3f} "
          f"({'PASS' if v['decoder_leakage_pass'] else 'FAIL'}{why}) | {pipe} |")
    w("\nM3 (active parameters D / F, mismatch): " + "; ".join(
        f"{c} {out[c]['params']['params_D']:,} / {out[c]['params']['params_F']:,} ({100 * out[c]['params']['rel_mismatch']:+.2f}%)"
        for c in conds) + ".\n")

    w("## Tuning on seed 1000 (G8, T32, G32; R uses the iteration-3 values)\n")
    w("| Condition | S* (plateau) | Selected D | Selected F | D val | F val | F_noz val | F share of ceiling (val) | M4 (val) |\n"
      "|---|---|---|---|---|---|---|---|---|")
    for c, o in tu["conditions"].items():
        hp = lambda h: ", ".join(f"{k} {v:g}" for k, v in h.items())  # noqa: E731
        w(f"| {c} | {o['steps']} ({'yes' if o['plateau_reached'] else 'cap'}) | {hp(o['selected_hp']['D'])} | "
          f"{hp(o['selected_hp']['F'])} | {o['selected_val']['D']:+.3f} | {o['selected_val']['F']:+.3f} | "
          f"{o['Fnoz']['val']:+.3f} | {o['m4']['F_share_of_ceiling_val']:.2f} | {'pass' if o['m4']['pass'] else 'FAIL'} |")
    w("\nStep ladders (best validation BSS, D / F): " + "; ".join(
        f"{c}: " + ", ".join(f"{r['steps']} → {r['D_val']:+.3f} / {r['F_val']:+.3f}" for r in o["ladder"])
        for c, o in tu["conditions"].items()) + f". Tuning-seed ceilings: K = 8 val {tu['ceiling']['k8']['val']:.3f}, "
      f"K = 32 val {tu['ceiling']['k32']['val']:.3f}.\n")
    am = tu.get("amendment")
    if am and am["changed"]:
        for ch in am["changed"]:
            lit = tu["conditions"][ch["condition"]]["literal_rule"]
            w(f"**Plateau-rule amendment (tuning seed only, before freezing):** {am['rule']} (onset = SESOI = "
              f"{am['onset_bss']}). It changes {ch['condition']} only: the literal rule gave S* = {ch['literal_S*']} with D "
              f"and F both at the floor (validation BSS {lit['selected_val']['D']:+.3f} / {lit['selected_val']['F']:+.3f}; "
              f"M4 {'pass' if lit['m4']['pass'] else 'fail'}), the amended rule gives S* = {ch['amended_S*']}. Both "
              f"tuning results are in `tuning_iter4.json`.\n")
    notes = HERE / c4["meta"]["design_notes"]
    if notes.exists():
        w(notes.read_text(encoding="utf-8"))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
