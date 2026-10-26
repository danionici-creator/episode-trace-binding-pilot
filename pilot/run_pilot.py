"""Phase I simulation pilot for Study 1, "Binding Episodes to Their Internal Effects".

    python pilot/run_pilot.py            # full pilot -> pilot/results.json, pilot/REPORT.md
    python pilot/run_pilot.py --quick    # smoke test  -> pilot/quick/
    python pilot/run_pilot.py --config config_iter2.yaml   # iteration 2 -> results_iter2.json, REPORT_iter2.md

Phases
  1. one agent life per seed (persistent environment, plastic slow state, M and Phi);
     M1 (path dependence), M2 (information ceiling) and the decoder leakage audit
  2. identical hyperparameter-tuning budget for D and F on the tuning seed
  3. all self-model conditions on the main seeds (D_lambda, F_lambda, F_noz, F_long,
     learning curve), every job single-threaded, in parallel
  4. paired seed-level analysis, power calculation, report
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.stats as st  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from checks import run_m2_and_leakage  # noqa: E402
from self_models import matched_widths  # noqa: E402
from sim_env import ROUTES, simulate_life, window_layout  # noqa: E402
from train_sm import run_job  # noqa: E402


# ----------------------------------------------------------------------------- workers
def _init_worker(n_threads):
    import torch
    torch.set_num_threads(n_threads)


def simulate_task(cfg, seed, cache_dir, with_checks):
    t0 = time.time()
    data, m1 = simulate_life(cfg, seed, with_m1=with_checks)
    m2 = run_m2_and_leakage(data, cfg, seed) if with_checks else None
    counts = np.zeros((3, cfg["environment"]["n_kinds"]), int)
    np.add.at(counts, (data["y"].ravel(), data["kind"].ravel()), 1)
    keep = data["role"] != 3   # decoder-only windows are not needed by the self-models
    path = Path(cache_dir) / f"seed_{seed}.npz"
    np.savez(path, **{k: v[keep] for k, v in data.items()})
    return {"seed": seed, "m1": m1, "m2": m2, "route_kind_counts": counts.tolist(),
            "seconds": time.time() - t0, "path": str(path)}


def run_parallel(fn, arg_list, n_workers, n_threads, label):
    out, t0 = [], time.time()
    with ProcessPoolExecutor(n_workers, initializer=_init_worker, initargs=(n_threads,)) as ex:
        futs = [ex.submit(fn, *a) for a in arg_list]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            print(f"  [{label}] {i}/{len(futs)} done ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------- statistics
def describe(x):
    x = np.asarray(x, float)
    n, m = len(x), float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    h = st.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": m, "sd": sd, "n": n, "ci95": [m - h, m + h], "values": x.tolist()}


def paired(a, b, cfg, rng):
    """Seed-level paired contrast a - b, with the inputs needed for the power analysis."""
    meta = cfg["meta"]
    d = np.asarray(a, float) - np.asarray(b, float)
    out = describe(d)
    n, sd, m = out["n"], out["sd"], out["mean"]
    if n < 2:
        return out
    alpha = meta["alpha_two_sided"]
    h_adj = st.t.ppf(1 - alpha, n - 1) * sd / math.sqrt(n)   # (1 - 2 alpha) interval, as in Sec. 4.7
    out["ci_1m2alpha"] = [m - h_adj, m + h_adj]
    h_two = st.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n)   # (1 - alpha) two-sided interval
    out["ci_1malpha"] = [m - h_two, m + h_two]
    boots = rng.choice(d, (10000, n), replace=True).mean(1)
    out["ci95_bootstrap"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    out["dz"] = m / sd if sd > 0 else float("inf")
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(meta["power"])
    sd_ucl80 = sd * math.sqrt((n - 1) / st.chi2.ppf(0.20, n - 1))
    out["sd_ucl80"] = sd_ucl80
    out["n_runs_for_sesoi"] = math.ceil((z * sd / meta["sesoi_bss"]) ** 2)
    out["n_runs_for_sesoi_ucl80"] = math.ceil((z * sd_ucl80 / meta["sesoi_bss"]) ** 2)
    return out


def decision(pc, sesoi):
    """Decision rule of Sec. 4.7 applied to a paired contrast (pilot reading, not confirmatory).

    support: the (1 - alpha) interval excludes 0 and the estimate reaches SESOI;
    falsification: the (1 - 2 alpha) interval lies entirely below +SESOI (equivalence when it also
    lies above -SESOI); inconclusive otherwise.
    """
    if "ci_1malpha" not in pc:
        return {"decision": "n/a", "equivalent": False}
    lo2, hi2 = pc["ci_1m2alpha"]
    if pc["ci_1malpha"][0] > 0 and pc["mean"] >= sesoi:
        return {"decision": "support", "equivalent": False}
    if hi2 < sesoi:
        eq = lo2 > -sesoi
        return {"decision": "falsification" + (" (equivalent to zero within ±SESOI)" if eq else " (below SESOI)"),
                "equivalent": bool(eq)}
    return {"decision": "inconclusive", "equivalent": False}


def within_seed_se(runs_a, runs_b):
    """Test-set sampling SE of BSS_a - BSS_b inside one seed (same test episodes)."""
    ses = []
    for ra, rb in zip(runs_a, runs_b):
        diff = rb["brier_items"].astype(float) - ra["brier_items"].astype(float)
        ses.append(diff.std(ddof=1) / math.sqrt(len(diff)) / ra["brier_ref_items"].mean())
    return float(np.sqrt(np.mean(np.square(ses))))


# ------------------------------------------------------------------------------ main run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test with tiny budgets")
    ap.add_argument("--report-only", action="store_true", help="re-render REPORT.md from results.json")
    ap.add_argument("--config", default="config.yaml", help="configuration file inside pilot/")
    ap.add_argument("--keep-cache", action="store_true", help="keep the simulated lives in cache/")
    args = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load((HERE / args.config).read_text(encoding="utf-8"))
    sfx = cfg["meta"].get("output_suffix", "")
    out_dir = HERE / f"quick{sfx}" if args.quick else HERE
    res_path, rep_path = out_dir / f"results{sfx}.json", out_dir / f"REPORT{sfx}.md"
    if args.report_only:
        R = json.loads(res_path.read_text(encoding="utf-8"))
        rep_path.write_text(render_report(R), encoding="utf-8")
        return
    if args.quick:
        cfg["seeds"]["main"] = [1, 2]
        cfg["training"]["steps"] = 200
        cfg["data"].update(budgets=[1, 8], m2_multiplier=1)
        cfg["checks"]["decoder_epochs"] = 2
        cfg["conditions"]["diagnostic_seeds"] = 2
    out_dir.mkdir(exist_ok=True)
    cache = out_dir / "cache"
    cache.mkdir(exist_ok=True)
    nw = cfg["meta"]["n_workers"]
    n_workers = max(1, (os.cpu_count() or 4) - 2) if nw == "auto" else int(nw)
    n_threads = cfg["meta"].get("threads_per_worker", 1)
    seeds, tseed = cfg["seeds"]["main"], cfg["seeds"]["tuning"]
    dseeds = seeds[: cfg["conditions"]["diagnostic_seeds"]]
    conf = cfg["data"]["confirmatory_budget"]
    rng = np.random.default_rng(12345)
    timing = {}

    E = cfg["environment"]
    dims = {"rec": E["d_record"] + E["n_kinds"], "z": cfg["measurement"]["d_z"], "s": 2 * cfg["measurement"]["d_z"]}
    widths = matched_widths(dims, cfg["self_model"], window_layout(cfg))
    print("parameter matching:", widths, flush=True)
    assert abs(widths["rel_mismatch"]) <= cfg["self_model"]["max_param_mismatch"], "M3 parameter match failed"

    # 1. simulation, M1, M2, decoder leakage audit
    print(f"phase 1: simulating {len(seeds) + 1} agent lives on {n_workers} workers", flush=True)
    t0 = time.time()
    sims = run_parallel(simulate_task, [(cfg, s, str(cache), s != tseed) for s in [tseed] + seeds], n_workers, n_threads, "simulate")
    sims = {r["seed"]: r for r in sims}
    timing["simulate_and_checks_s"] = time.time() - t0

    # 2. tuning: identical grid for D and F on the tuning seed
    tg = cfg["training"]["tuning_grid"]   # full factorial over every listed hyperparameter, in file order
    grid = [dict(zip(tg, vals)) for vals in itertools.product(*tg.values())]
    print(f"phase 2: tuning, {len(grid)} configurations per architecture", flush=True)
    t0 = time.time()
    tjobs = [{"seed": tseed, "arch": a, "lam": 0.0, "budget": conf, "tag": f"tune_{a}_{i}", "hp": hp,
              "data_path": sims[tseed]["path"]} for a in ("D", "F") for i, hp in enumerate(grid)]
    tres = run_parallel(run_job, [(j, cfg, widths) for j in tjobs], n_workers, n_threads, "tune")
    hp_best, tuning = {}, []
    for a in ("D", "F"):
        rows = [r for r in tres if r["arch"] == a]
        best = max(rows, key=lambda r: r["val_bss"])
        hp_best[a] = best["hp"]
        tuning += [{"arch": a, "hp": r["hp"], "val_bss": r["val_bss"], "test_bss_tuning_seed": r["test"]["bss"]} for r in rows]
    print("selected:", hp_best, flush=True)
    timing["tuning_s"] = time.time() - t0

    # 3. main conditions
    jobs = []
    for s in seeds:
        base = {"seed": s, "data_path": sims[s]["path"]}
        if cfg["conditions"]["f_long_step_multiplier"] and s in dseeds:
            jobs.append({**base, "arch": "F", "lam": 0.0, "budget": conf, "tag": "F_long", "hp": hp_best["F"],
                         "steps": cfg["training"]["steps"] * cfg["conditions"]["f_long_step_multiplier"]})
        for lam in cfg["conditions"]["lambdas"]:
            for a in ("D", "F"):
                jobs.append({**base, "arch": a, "lam": lam, "budget": conf, "tag": f"{a}_lam{lam:g}", "hp": hp_best[a],
                             "test_misbinding": a == "D" and lam == 0 and cfg["conditions"]["test_misbinding"]})
        if cfg["conditions"]["f_noz"]:
            jobs.append({**base, "arch": "Fnoz", "lam": 0.0, "budget": conf, "tag": "Fnoz", "hp": hp_best["F"]})
        for b in cfg["data"]["budgets"]:
            if b != conf and s in dseeds:
                for a in ("D", "F"):
                    jobs.append({**base, "arch": a, "lam": 0.0, "budget": b, "tag": f"{a}_n{b}", "hp": hp_best[a]})
    jobs.sort(key=lambda j: -j.get("steps", cfg["training"]["steps"]))   # longest first for packing
    print(f"phase 3: {len(jobs)} self-model runs", flush=True)
    t0 = time.time()
    res = run_parallel(run_job, [(j, cfg, widths) for j in jobs], n_workers, n_threads, "main")
    timing["main_runs_s"] = time.time() - t0
    runs = {}
    for r in res:
        runs.setdefault(r["tag"], {})[r["seed"]] = r
    np.savez(cache / f"per_episode_brier{sfx}.npz",
             **{f"{tag}__seed{s}": r["brier_items"] for tag, rr in runs.items() for s, r in rr.items()},
             **{f"ref__seed{s}": runs["D_lam0"][s]["brier_ref_items"] for s in seeds})

    # 4. analysis
    def col(tag, key="bss", sub="test", on=None):
        return [runs[tag][s][sub][key] for s in (on or seeds)]

    def lc_tag(a, b):   # the confirmatory-budget point of the learning curve is the lambda = 0 run
        return f"{a}_lam0" if b == conf else f"{a}_n{b}"

    lams = cfg["conditions"]["lambdas"]
    sesoi, margin = cfg["meta"]["sesoi_bss"], cfg["checks"]["leakage_margin"]
    an = {}
    an["conditions"] = {tag: {k: describe(col(tag, k, on=sorted(runs[tag]))) for k in ("bss", "log_loss", "accuracy", "ece")}
                        for tag in sorted(runs)}
    an["D_minus_F"] = paired(col("D_lam0"), col("F_lam0"), cfg, rng)
    an["D_minus_F"]["within_seed_test_se"] = within_seed_se([runs["D_lam0"][s] for s in seeds],
                                                            [runs["F_lam0"][s] for s in seeds])
    an["D_minus_F"]["between_run_sd_est"] = float(math.sqrt(max(0.0, an["D_minus_F"]["sd"] ** 2
                                                                 - an["D_minus_F"]["within_seed_test_se"] ** 2)))
    an["D_minus_F_secondary"] = {k: paired(col("D_lam0", k), col("F_lam0", k), cfg, rng)
                                 for k in ("log_loss", "accuracy", "ece")}
    slopes = {a: [float(np.polyfit(lams, [runs[f"{a}_lam{l:g}"][s]["test"]["bss"] for l in lams], 1)[0]) for s in seeds]
              for a in ("D", "F")}
    adv0 = np.array(col("D_lam0")) - np.array(col("F_lam0"))
    adv1 = np.array(col("D_lam1")) - np.array(col("F_lam0"))
    an["lambda"] = {
        "D_slope": describe(slopes["D"]), "F_slope": describe(slopes["F"]),
        "D_by_lambda": {f"{l:g}": describe(col(f"D_lam{l:g}")) for l in lams},
        "F_by_lambda": {f"{l:g}": describe(col(f"F_lam{l:g}")) for l in lams},
        "D_lambda_minus_F0": {f"{l:g}": paired(col(f"D_lam{l:g}"), col("F_lam0"), cfg, rng) for l in lams},
        "D_lambda_minus_F_lambda": {f"{l:g}": paired(col(f"D_lam{l:g}"), col(f"F_lam{l:g}"), cfg, rng) for l in lams},
        "D1_minus_F0": paired(col("D_lam1"), col("F_lam0"), cfg, rng),
        "retained_fraction_at_lambda1": float(adv1.mean() / adv0.mean()) if adv0.mean() != 0 else float("nan"),
        "D_means_monotone_decreasing": bool(np.all(np.diff([np.mean(col(f"D_lam{l:g}")) for l in lams]) < 0)),
    }
    if cfg["conditions"]["test_misbinding"]:
        mis = [runs["D_lam0"][s]["test_misbound"]["bss"] for s in seeds]
        an["D_test_misbound"] = {"bss": describe(mis), "minus_F0": paired(mis, col("F_lam0"), cfg, rng)}
    if "F_long" in runs:
        an["F_long"] = {"seeds": dseeds, "bss": describe(col("F_long", on=dseeds)),
                        "D0_minus_F_long": paired(col("D_lam0", on=dseeds), col("F_long", on=dseeds), cfg, rng),
                        "F_long_minus_F0": paired(col("F_long", on=dseeds), col("F_lam0", on=dseeds), cfg, rng)}
    an["learning_curve"] = {str(b): {"seeds": dseeds, "D": describe(col(lc_tag("D", b), on=dseeds)),
                                     "F": describe(col(lc_tag("F", b), on=dseeds)),
                                     "D_minus_F": paired(col(lc_tag("D", b), on=dseeds), col(lc_tag("F", b), on=dseeds), cfg, rng),
                                     "n_train_episodes": b * cfg["data"]["n0_windows"] * E["episodes_per_window"]}
                            for b in cfg["data"]["budgets"]}
    an["H1_decision"] = decision(an["D_minus_F"], sesoi)
    for lc in an["learning_curve"].values():
        lc["decision"] = decision(lc["D_minus_F"], sesoi)

    # manipulation checks and audits
    m1 = {s: sims[s]["m1"] for s in seeds}
    m2 = {s: sims[s]["m2"] for s in seeds}
    ceiling = np.array([m2[s]["full"]["bss"] for s in seeds])
    checks = {
        "M1": {"per_seed": m1, "ratio": describe([m1[s]["ratio"] for s in seeds]),
               "n_pass": int(sum(m1[s]["pass"] for s in seeds)), "pass": all(m1[s]["pass"] for s in seeds)},
        "M2": {"per_seed": m2, "n_train_episodes": m2[seeds[0]]["n_train_episodes"],
               **{f"{k}_bss": describe([m2[s][k]["bss"] for s in seeds]) for k in ("full", "s_only", "no_z", "control", "misaligned")},
               "delta_bss": describe([m2[s]["delta_bss"] for s in seeds]),
               "n_pass": int(sum(m2[s]["delta_bss"] >= sesoi for s in seeds)),
               "pass": all(m2[s]["delta_bss"] >= sesoi for s in seeds),
               "fraction_of_ceiling_D": describe(np.array(col("D_lam0")) / ceiling),
               "fraction_of_ceiling_F": describe(np.array(col("F_lam0")) / ceiling)},
        "leakage": {"decoder_no_z_bss": describe([m2[s]["no_z"]["bss"] for s in seeds]),
                    "decoder_no_z_accuracy": describe([m2[s]["no_z"]["accuracy"] for s in seeds]),
                    "decoder_pass": all(m2[s]["no_z"]["bss"] <= margin and m2[s]["no_z"]["accuracy"] <= 1 / 3 + margin
                                        for s in seeds)},
        "M3_parameter_match": {**widths, "pass": abs(widths["rel_mismatch"]) <= cfg["self_model"]["max_param_mismatch"]},
        "route_kind_counts": {s: sims[s]["route_kind_counts"] for s in seeds},
    }
    if "Fnoz" in runs:
        checks["leakage"]["pipeline_F_noz_bss"] = describe(col("Fnoz"))
        checks["leakage"]["pipeline_pass"] = bool(np.mean(col("Fnoz")) <= margin)
    if "Fnoz" in runs and "m4_margin" in cfg["checks"]:   # iteration 2: control competence
        m4m = cfg["checks"]["m4_margin"]
        m4 = paired(col("F_lam0"), col("Fnoz"), cfg, rng)
        checks["M4_control_competence"] = {
            **m4, "margin": m4m, "pass": bool(m4["mean"] >= m4m),
            "n_pass": int(sum(f - n >= m4m for f, n in zip(col("F_lam0"), col("Fnoz"))))}
        frac = cfg["checks"].get("m4_ceiling_fraction")
        if frac is not None:   # iteration 3: F must also reach a fraction of the information ceiling
            share = np.array(col("F_lam0")) / ceiling
            above = np.array(col("F_lam0")) - np.array(col("Fnoz")) >= m4m
            checks["M4_control_competence"].update(
                ceiling_fraction_required=frac, F_share_of_ceiling=describe(share),
                n_pass_share=int(np.sum(share >= frac)), n_pass_both=int(np.sum((share >= frac) & above)),
                pass_margin=bool(m4["mean"] >= m4m), pass_share=bool(share.mean() >= frac),
                **{"pass": bool(m4["mean"] >= m4m and share.mean() >= frac)})

    timing["total_s"] = time.time() - t_start
    results = {
        "study": "Binding Episodes to Their Internal Effects (Study 1) - Phase I simulation pilot",
        "config_file": args.config,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment_info": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": __import__("torch").__version__, "platform": platform.platform(),
                             "n_workers": n_workers},
        "timing": timing, "config": cfg, "tuning": {"grid_results": tuning, "selected": hp_best},
        "checks": checks, "analysis": an,
        "runs": {tag: {str(s): {k: v for k, v in r.items() if k not in ("brier_items", "brier_ref_items")}
                       for s, r in rr.items()} for tag, rr in runs.items()},
    }
    res_path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    R = json.loads(res_path.read_text(encoding="utf-8"))   # render from what was saved
    rep_path.write_text(render_report(R), encoding="utf-8")
    if not args.keep_cache:   # simulated lives are deterministic given config + seed; ~10 MB each
        for s in [tseed] + seeds:
            Path(sims[s]["path"]).unlink(missing_ok=True)
    print(f"done in {timing['total_s'] / 60:.1f} min -> {res_path}, {rep_path}", flush=True)


# -------------------------------------------------------------------------------- report
def render_candidates_iter3(cd, cfg, w):
    """Iteration 3: every admissible candidate on the tuning seed, its plateau budget, M4 and the selection."""
    ce = cd["ceiling"]
    w(f"### Candidates screened on the tuning seed (seed {cd['seed']} only; all with the co-timed objective)\n")
    w(f"Information ceiling on the tuning seed (M2 decoder, same protocol): validation BSS {ce['val']:.3f}, test "
      f"{ce['test']:.3f}. M4 = F ≥ {cd['m4_ceiling_fraction']} × ceiling and F − F_noz ≥ {cd['m4_margin']}, judged "
      f"on validation (where selection happens); test values are reported only.\n")
    w("| Candidate | S* (steps) | D val / test | F val / test | F_noz val / test | F share of ceiling, val / test "
      "| F − F_noz, val / test | M4 (val) |\n|---|---|---|---|---|---|---|---|")
    for c in cd["candidates"]:
        v, t = c["val"], c["test"]
        w(f"| {c['name']}{' **(selected)**' if c['name'] == cd['selected'] else ''} | {c['steps']}"
          f"{'' if c['plateau_reached'] else ' (cap)'} | {v['D']:+.3f} / {t['D']:+.3f} | {v['F']:+.3f} / {t['F']:+.3f} | "
          f"{v['Fnoz']:+.3f} / {t['Fnoz']:+.3f} | {v['F_share_of_ceiling']:.2f} / {t['F_share_of_ceiling']:.2f} | "
          f"{v['F_minus_Fnoz']:+.3f} / {t['F_minus_Fnoz']:+.3f} | {'pass' if c['m4_pass'] else 'fail'} |")
    w("\nStep ladders (best validation BSS; S* = smallest budget after which doubling gains < "
      f"{cd['plateau_tolerance']} for both D and F):\n")
    w("| Candidate | " + " | ".join(f"{r['steps']} steps: D / F" for r in max((c['rungs'] for c in cd['candidates']), key=len))
      + " |\n|---|" + "---|" * max(len(c["rungs"]) for c in cd["candidates"]))
    for c in cd["candidates"]:
        w(f"| {c['name']} | " + " | ".join(f"{r['D_val']:+.3f} / {r['F_val']:+.3f}" for r in c["rungs"]) + " |")
    w(f"\n**Selected:** {cd['selected']} at {cd['selected_steps']} steps. Rule: {cd['selection_rule']}"
      f"{'' if cd['m4_met_by_selected'] else ' (M4 FAILED on the tuning seed)'}.\n")


def render_candidates(cd, cfg, w):
    """Iteration 2: M4 for every candidate tried on the tuning seed."""
    if cd.get("format") == "iter3":
        return render_candidates_iter3(cd, cfg, w)
    w(f"### M4 candidates tried on the tuning seed (seed {cd['seed']} only; preregistered order "
      f"{' → '.join(cd['order'])})\n")
    rep = cd.get("iteration1_reproduction")
    if rep:
        w(f"Iteration-1 reproduction check (D on the tuning seed with config.yaml): val BSS {rep['val_bss']:.4f} vs "
          f"{rep['reference_val_bss']:.4f} in results.json ({'identical' if rep['match'] else 'DIFFERENT'}).\n")
    w("| Candidate | Steps | D val / test | F val / test | F_noz test | M4 = F − F_noz (test) | M4 |\n|---|---|---|---|---|---|---|")
    rows = []
    for c in cd["candidates"]:
        for r in (c["rungs"] if "rungs" in c else [c]):
            tag = c["name"] + (" (plateau budget)" if "rungs" in c and r["steps"] == c["plateau_steps"] else
                               " (ladder)" if "rungs" in c else "") + (" — screened after selection" if c.get("screened_after_selection") else "")
            rows.append(f"| {tag} | {r['steps']} | {r['val_bss']['D']:+.3f} / {r['test_bss']['D']:+.3f} | "
                        f"{r['val_bss']['F']:+.3f} / {r['test_bss']['F']:+.3f} | {r['test_bss']['Fnoz']:+.3f} | "
                        f"{r['m4_test']:+.3f} | {'pass' if r['m4_test'] >= cd['m4_margin'] else 'fail'} |")
    w("\n".join(rows))
    a = next((c for c in cd["candidates"] if c["name"] == "a"), None)
    if a:
        w(f"\nCandidate a: plateau budget S* = {a['plateau_steps']} steps "
          f"({'plateau reached' if a['plateau_reached'] else 'no plateau within the ladder; S* capped at its last rung'}; "
          f"rule: best validation BSS improves by < {cfg['candidates']['a']['plateau_tolerance']} for both D and F when "
          f"the steps are doubled). M4 for candidate a is judged at S*.")
    sel = cd.get("selected")
    w(f"\n**Selected:** {sel + ', the first candidate in the preregistered order to pass M4 on the tuning seed' if sel else 'none. No candidate passed M4 on the tuning seed; best F − F_noz = ' + format(cd['best_m4'], '+.3f')}.\n")


def _ci(d, k="ci95"):
    return f"[{d[k][0]:+.3f}, {d[k][1]:+.3f}]"


def _ms(d):
    return f"{d['mean']:+.3f} ± {d['sd']:.3f}"


def render_report(R):
    cfg, an, ck = R["config"], R["analysis"], R["checks"]
    sesoi = cfg["meta"]["sesoi_bss"]
    seeds = cfg["seeds"]["main"]
    conf = cfg["data"]["confirmatory_budget"]
    lams = cfg["conditions"]["lambdas"]
    dmf = an["D_minus_F"]
    C = an["conditions"]
    m1, m2, lk, lam = ck["M1"], ck["M2"], ck["leakage"], an["lambda"]
    f_above_floor = C["F_lam0"]["bss"]["mean"] - C["Fnoz"]["bss"]["mean"] if "Fnoz" in C else C["F_lam0"]["bss"]["mean"]
    m4 = ck.get("M4_control_competence")
    checks_ok = (m1["pass"] and m2["pass"] and lk["decoder_pass"] and lk.get("pipeline_pass", True)
                 and ck["M3_parameter_match"]["pass"] and (m4 is None or m4["pass"]))
    sfx = cfg["meta"].get("output_suffix", "")
    it = cfg["meta"].get("iteration", 1)
    sampling_dominated = dmf["within_seed_test_se"] >= 0.8 * dmf["sd"]
    L = []
    w = L.append
    w(f"# Phase I simulation pilot{f' (iteration {it})' if it > 1 else ''} — Study 1, "
      f"*Binding Episodes to Their Internal Effects*\n")
    w(f"Generated {R['generated']} by `python pilot/run_pilot.py"
      f"{'' if R.get('config_file', 'config.yaml') == 'config.yaml' else ' --config ' + R['config_file']}` "
      f"({R['timing']['total_s'] / 60:.1f} min on {R['environment_info']['n_workers']} CPU workers; "
      f"Python {R['environment_info']['python']}, torch {R['environment_info']['torch']}). "
      f"{len(seeds)} seeds per condition; every seed is a new environment instance and agent life. "
      f"All numbers are means ± SD across seeds with 95% t-intervals unless stated otherwise. "
      f"Full numbers: `results{sfx}.json`. Preregistered settings: `{R.get('config_file', 'config.yaml')}`.\n")

    w("## Bottom line\n")
    w(f"- **Feasibility:** {'all manipulation checks pass' if checks_ok else 'at least one manipulation check FAILS (see below)'}. "
      f"The trace carries source information that the log does not: the M2 ceiling is BSS {m2['full_bss']['mean']:.2f} with z "
      f"vs {m2['no_z_bss']['mean']:+.3f} without. The slow state is path-dependent: permuted-order distance is "
      f"{m1['ratio']['mean']:.1f}× the test–retest threshold. No source information reaches the self-models except "
      f"through the trace.")
    if m4 is not None and "F_share_of_ceiling" in m4:
        sh = m4["F_share_of_ceiling"]
        w(f"- **M4 (control competence, strengthened):** F reaches {sh['mean']:.2f} {_ci(sh)} of the information "
          f"ceiling (required ≥ {m4['ceiling_fraction_required']}; {m4['n_pass_share']}/{m4['n']} seeds) and "
          f"F − F_noz = {m4['mean']:+.3f} {_ci(m4)} BSS (required ≥ {m4['margin']}; {m4['n_pass']}/{m4['n']} seeds); "
          f"both criteria are met in {m4['n_pass_both']}/{m4['n']} seeds; **{'PASS' if m4['pass'] else 'FAIL'}**.")
    elif m4 is not None:
        w(f"- **M4 (control competence):** F − F_noz = {m4['mean']:+.3f} {_ci(m4)} BSS; at least SESOI in "
          f"{m4['n_pass']}/{m4['n']} seeds; {'PASS' if m4['pass'] else 'FAIL'}. F exploits "
          f"{100 * m2['fraction_of_ceiling_F']['mean']:.0f}% of the information ceiling.")
    above = ("remains above SESOI; the whole 95% CI lies above it" if dmf["ci95"][0] > sesoi else
             "has a point estimate above SESOI, but the CI includes it" if dmf["mean"] >= sesoi else "falls below SESOI")
    if it == 1:   # iteration-1 wording kept so that REPORT.md reproduces byte for byte
        w(f"- **H1:** D exploits {100 * m2['fraction_of_ceiling_D']['mean']:.0f}% of the information ceiling, F "
          f"{100 * m2['fraction_of_ceiling_F']['mean']:.0f}%. D − F = {dmf['mean']:+.3f} BSS {_ci(dmf)}; "
          f"the SD of the paired difference over {dmf['n']} seeds is {dmf['sd']:.3f}.")
    else:
        w(f"- **H1:** D exploits {100 * m2['fraction_of_ceiling_D']['mean']:.0f}% of the information ceiling, F "
          f"{100 * m2['fraction_of_ceiling_F']['mean']:.0f}%. D − F = {dmf['mean']:+.3f} BSS {_ci(dmf)}, which "
          f"{above}. The SD of the paired difference over {dmf['n']} seeds is {dmf['sd']:.3f}.")
    adv0 = lam["D_lambda_minus_F0"][f"{lams[0]:g}"]["mean"]
    if it >= 3:
        dec = an["H1_decision"]
        w(f"- **Decision rule (Sec. 4.7, pilot reading):** D − F = {dmf['mean']:+.3f}; (1 − α) CI "
          f"{_ci(dmf, 'ci_1malpha')}, (1 − 2α) CI {_ci(dmf, 'ci_1m2alpha')} with α = 0.05/3, equivalence bounds "
          f"±{sesoi}. **{dec['decision'][0].upper() + dec['decision'][1:]}**: " + {
              "support": "the interval excludes zero and the estimate reaches SESOI.",
              "inconclusive": "the interval neither lies entirely below +SESOI nor supports an effect of SESOI.",
          }.get(dec["decision"], "the (1 − 2α) interval lies entirely below +SESOI"
                + (" and inside ±SESOI: D and F are equivalent within the SESOI." if dec["equivalent"] else ".")))
        lcs = {int(b): an["learning_curve"][b] for b in an["learning_curve"]}
        parts = [f"{b} n0: {v['D_minus_F']['mean']:+.3f} {_ci(v['D_minus_F'])} ({v['decision']['decision'].split(' (')[0]})"
                 for b, v in sorted(lcs.items())]
        sup = [b for b, v in sorted(lcs.items()) if v["decision"]["decision"] == "support"]
        where = ("D's advantage reaches SESOI at every budget" if len(sup) == len(lcs) else
                 f"D's advantage reaches SESOI only at {', '.join(str(b) + ' n0' for b in sup)}" if sup else
                 "D's advantage does not reach SESOI at any budget")
        w(f"- **Learning curve ({len(lcs[conf]['seeds'])} seeds), D − F by training budget:** " + "; ".join(parts)
          + f". {where}.")
    if it >= 3 and adv0 < sesoi:
        w(f"- **H2:** with no D − F advantage of SESOI at λ = 0 there is little for misalignment to remove, so H2 is "
          f"not informative here. D's own BSS falls from {lam['D_by_lambda'][f'{lams[0]:g}']['mean']:+.3f} to "
          f"{lam['D_by_lambda'][f'{lams[-1]:g}']['mean']:+.3f} as λ goes from {lams[0]:g} to {lams[-1]:g}, and F's from "
          f"{lam['F_by_lambda'][f'{lams[0]:g}']['mean']:+.3f} to {lam['F_by_lambda'][f'{lams[-1]:g}']['mean']:+.3f}; "
          f"D₁ − F₀ = {lam['D1_minus_F0']['mean']:+.3f} {_ci(lam['D1_minus_F0'])}. "
          + (f"D barely depends on the bound signatures. Trained with all of them misbound, it learns the pairing from "
             f"the unbound streams instead, while F, whose only pairing is the permuted stream, falls to the floor"
             + (f". Yet with correct training, misbound test signatures drive D to "
                f"{an['D_test_misbound']['bss']['mean']:+.3f}: D uses B when it is present but does not need it."
                if "D_test_misbound" in an else ".")
             if lam['D_by_lambda'][f'{lams[0]:g}']['mean'] - lam['D_by_lambda'][f'{lams[-1]:g}']['mean'] < sesoi else
             "Both use the trace only when it is correctly paired."))
    else:
        w(f"- **H2:** D's BSS {'declines monotonically' if lam['D_means_monotone_decreasing'] else 'does not decline monotonically'} "
        f"with training-time misalignment (slope {lam['D_slope']['mean']:+.3f} per unit λ) and reaches D₁ − F₀ = "
        f"{lam['D1_minus_F0']['mean']:+.3f} {_ci(lam['D1_minus_F0'])} at λ = 1, the pattern H2 predicts"
        + (f"; D with correct training but misbound test signatures falls far below F "
           f"({an['D_test_misbound']['minus_F0']['mean']:+.3f})." if "D_test_misbound" in an else "."))
    cand_path = HERE / cfg["meta"].get("candidates_file", "") if cfg["meta"].get("candidates_file") else None
    if cand_path and cand_path.exists():
        cd = json.loads(cand_path.read_text(encoding="utf-8"))
        later = [c for c in cd["candidates"] if c.get("screened_after_selection")]
        best = max(later, key=lambda c: c["m4_test"], default=None)
        sel = next((c for c in cd["candidates"] if c["name"] == cd.get("selected")), None)
        if best and sel and best["m4_test"] > sel["m4"] + 2 * sesoi:
            w(f"- **Post-selection screen (tuning seed only, information only):** candidate {best['name']} raises "
              f"F − F_noz from {sel['m4']:+.3f} (selected {sel['name']}) to {best['m4_test']:+.3f}. On that seed F reaches "
              f"{best['test_bss']['F']:+.3f} against D's {best['test_bss']['D']:+.3f} under the same training, so "
              f"D − F = {best['test_bss']['D'] - best['test_bss']['F']:+.3f}. The selected F passes M4 but is far from "
              f"the strongest admissible control, and D's advantage over it says little about D against a competent "
              f"F. M4 as preregistered is too lenient (design notes, points 3–4). This is a single untuned run, "
              f"not a main-seed result.")
    if sampling_dominated:
        w(f"- **Variance:** the across-seed SD of D − F ({dmf['sd']:.3f}) is accounted for by test-set sampling "
          f"(within-seed SE {dmf['within_seed_test_se']:.3f} with {R['runs']['D_lam0'][str(seeds[0])]['test']['n']:,} test "
          f"episodes per run). Between-run variance is negligible here, so the confirmatory SD is set mainly by the number "
          f"of test episodes per run, not by the number of runs.")
    if f_above_floor < sesoi:
        w(f"- **Main caveat:** F barely uses the trace. F − F_noz = {f_above_floor:+.3f}, below SESOI"
          + (f"; with {cfg['conditions']['f_long_step_multiplier']}× the steps F reaches only "
             f"{an['F_long']['bss']['mean']:+.3f}" if "F_long" in an else "")
          + f"; over a {max(cfg['data']['budgets']) // min(cfg['data']['budgets'])}-fold range of training data F moves only "
            f"from {an['learning_curve'][str(min(cfg['data']['budgets']))]['F']['mean']:+.3f} to "
            f"{an['learning_curve'][str(max(cfg['data']['budgets']))]['F']['mean']:+.3f}, while D moves from "
            f"{an['learning_curve'][str(min(cfg['data']['budgets']))]['D']['mean']:+.3f} to "
            f"{an['learning_curve'][str(max(cfg['data']['budgets']))]['D']['mean']:+.3f}. The effect therefore measures the two-hop "
            f"episode–trace join that a generic transformer does not discover within budget. That is the "
            f"inductive-bias claim in its strongest form, but it also means the H1 effect size, and with it the power "
            f"analysis, says nothing yet about a *competent* control. A control-competence check and a stronger "
            f"generic F are needed before confirmatory data collection "
            f"({'design note 8' if it == 1 else 'see design notes'}).")
    w("")

    w("## Instantiation in one paragraph\n")
    E = cfg["environment"]
    w(f"A persistent environment (world map o = tanh(Ac), {E['n_kinds']} content kinds, a scripted other agent) produces "
      f"fixed-length episodes ({E['steps_per_episode']} steps) with a latent label y_k in {{own, imported, inferred}}, "
      f"drawn i.i.d. and independent of content kind. **Own**: repeated action–outcome trials with sensory and outcome noise. "
      f"**Imported**: the other agent's message (cue through a {E['imported']['comm_dim']}-d shared code; outcome from the "
      f"sender's own, imperfect world model). **Inferred**: rehearsal that recombines the item with an earlier retained cue, "
      f"its target partly regenerated from the slow memory U. Every route drives the same delta-rule update of the slow "
      f"state h = (W, U) (fast synapse consolidating into a slow variable), with the same gain. "
      f"M: z_t = R vec(h_t − h_{{t−1}})/η + N(0, {cfg['measurement']['sigma_z']}²), R a fixed Gaussian projection to "
      f"{cfg['measurement']['d_z']} dimensions; Φ: s_k = (mean, SD) of z over the episode. "
      f"Log records and queries are route-independent random descriptors (no provenance flag exists to strip). "
      f"Self-models see windows of {E['episodes_per_window']} episodes: record tokens, per-step trace tokens (z_t, timestamp, "
      f"boundary flag) and an unbound signature stream (s_k, stamped with its episode's timestamp). "
      f"**F** is a {cfg['self_model']['enc_layers']}+{cfg['self_model']['readout_layers']}-layer memory-augmented transformer "
      f"over these streams; **D** is the same network plus the binding B (s_k written into the record token of e_k). "
      f"Active parameters: D {ck['M3_parameter_match']['params_D']:,}, F {ck['M3_parameter_match']['params_F']:,} "
      f"(F's feed-forward width {ck['M3_parameter_match']['d_ff_F']} vs {ck['M3_parameter_match']['d_ff_D']}; "
      f"mismatch {100 * ck['M3_parameter_match']['rel_mismatch']:+.2f}%). "
      + (f"Both are trained with a window-length curriculum (K = "
         f"{' → '.join(str(k) for _, k in cfg['training']['curriculum'])} episodes). " if cfg["training"].get("curriculum") else "")
      + (f"Both carry a label-free auxiliary loss: each encoded record token reconstructs its co-timed signature "
         + (f"(weight tuned per architecture from {cfg['training']['tuning_grid']['aux_cotimed_weight']}; selected D "
            f"{R['tuning']['selected']['D']['aux_cotimed_weight']}, F {R['tuning']['selected']['F']['aux_cotimed_weight']}). "
            if "aux_cotimed_weight" in cfg["training"]["tuning_grid"] else
            f"(weight {cfg['training']['aux_cotimed_weight']}). ") if cfg["training"].get("aux_cotimed_weight") else "")
      + f"Identical steps ({cfg['training']['steps']}), "
      f"batch, schedule, data order and tuning budget ({len(R['tuning']['grid_results']) // 2} configurations each; "
      f"selected D {R['tuning']['selected']['D']}, F {R['tuning']['selected']['F']}). Confirmatory budget "
      f"{conf} n0 = {an['learning_curve'][str(conf)]['n_train_episodes']:,} labelled training episodes; "
      f"{R['runs']['D_lam0'][str(seeds[0])]['test']['n']:,} novel test episodes per run.\n")

    w("## Manipulation checks and leakage audit\n")
    w("| Check | Criterion | Result | Verdict |\n|---|---|---|---|")
    w(f"| M1 path dependence | median d(h_σ, h_id) > δ_C (95th pct. test–retest) | distance ratio {_ms(m1['ratio'])} "
      f"(min {min(m1['ratio']['values']):.2f}) | {'PASS' if m1['pass'] else 'FAIL'} ({m1['n_pass']}/{len(seeds)} seeds) |")
    w(f"| M2 information ceiling | BSS(decoder with z) − BSS(without z) ≥ {sesoi} | Δ = {_ms(m2['delta_bss'])} "
      f"(min {min(m2['delta_bss']['values']):+.3f}) | {'PASS' if m2['pass'] else 'FAIL'} ({m2['n_pass']}/{len(seeds)} seeds) |")
    w(f"| Leakage, decoder | non-trace inputs: BSS ≤ 0.02 and accuracy ≤ 1/3 + 0.02 | BSS {_ms(lk['decoder_no_z_bss'])}, "
      f"accuracy {lk['decoder_no_z_accuracy']['mean']:.3f} (max {max(lk['decoder_no_z_accuracy']['values']):.3f}) | "
      f"{'PASS' if lk['decoder_pass'] else 'FAIL'} |")
    if "pipeline_F_noz_bss" in lk:
        w(f"| Leakage, pipeline | F trained and tested without z and s: BSS ≤ 0.02 | BSS {_ms(lk['pipeline_F_noz_bss'])} | "
          f"{'PASS' if lk['pipeline_pass'] else 'FAIL'} |")
    w(f"| M3 capacity match | active parameters within ±2% | {100 * ck['M3_parameter_match']['rel_mismatch']:+.2f}% | "
      f"{'PASS' if ck['M3_parameter_match']['pass'] else 'FAIL'} |")
    if m4 is not None and "F_share_of_ceiling" in m4:
        w(f"| M4 control competence | BSS(F) ≥ {m4['ceiling_fraction_required']} × ceiling and BSS(F) − BSS(F_noz) ≥ "
          f"{m4['margin']} | share {_ms(m4['F_share_of_ceiling'])}; F − F_noz {_ms(m4)} {_ci(m4)} | "
          f"{'PASS' if m4['pass'] else 'FAIL'} ({m4['n_pass_both']}/{m4['n']} seeds meet both individually) |")
    elif m4 is not None:
        w(f"| M4 control competence | BSS(F) − BSS(F_noz) ≥ {m4['margin']} | {_ms(m4)} {_ci(m4)} | "
          f"{'PASS' if m4['pass'] else 'FAIL'} ({m4['n_pass']}/{m4['n']} seeds individually) |")
    w("")
    if cfg["meta"].get("candidates_file") and (HERE / cfg["meta"]["candidates_file"]).exists():
        render_candidates(json.loads((HERE / cfg["meta"]["candidates_file"]).read_text(encoding="utf-8")), cfg, w)
    w(f"Information ceiling (independent decoder, correct pairing, {m2['n_train_episodes']:,} "
      f"training episodes = 10 × the confirmatory budget), test BSS:\n")
    w("| Decoder features | BSS |\n|---|---|")
    for k, lab in (("full", "z for every step + s_k + record + timestamp (ceiling with z)"),
                   ("s_only", "s_k + record + timestamp (what Φ retains)"),
                   ("no_z", "record + query + timestamp + window log context (ceiling without z)"),
                   ("control", "control task: full features, shuffled training labels"),
                   ("misaligned", "full features with z and s_k taken from another episode of the window")):
        w(f"| {lab} | {_ms(m2[k + '_bss'])} |")
    w(f"\nD exploits {100 * m2['fraction_of_ceiling_D']['mean']:.0f}% of the ceiling, "
      f"F {100 * m2['fraction_of_ceiling_F']['mean']:.0f}%.\n")

    w("## Primary outcome: D vs F (H1)\n")
    w(f"Novel test episodes, confirmatory budget {conf} n0.\n")
    w("| Condition | BSS | log loss | accuracy | ECE |\n|---|---|---|---|---|")
    for tag, lab in (("D_lam0", "D (explicit binding)"), ("F_lam0", "F (information control)"),
                     ("F_long", f"F_long ({cfg['conditions']['f_long_step_multiplier']}× steps, diagnostic)"),
                     ("Fnoz", "F without z and s (floor)")):
        if tag in C:
            c = C[tag]
            w(f"| {lab} | {_ms(c['bss'])} {_ci(c['bss'])} | {c['log_loss']['mean']:.3f} | "
              f"{c['accuracy']['mean']:.3f} | {c['ece']['mean']:.3f} |")
    w("")
    w(f"**Paired D − F (BSS):** mean {dmf['mean']:+.3f}, **SD of the paired difference {dmf['sd']:.3f}** "
      f"(80% upper confidence limit {dmf['sd_ucl80']:.3f}), 95% CI {_ci(dmf)}, bootstrap 95% CI {_ci(dmf, 'ci95_bootstrap')}, "
      f"(1 − 2α) CI with α = 0.05/3 {_ci(dmf, 'ci_1m2alpha')}"
      + (f", (1 − α) CI {_ci(dmf, 'ci_1malpha')}" if it >= 3 else "") + f"; d_z = {dmf['dz']:.1f}. "
      f"Test-set sampling SE within a seed {dmf['within_seed_test_se']:.3f}; implied between-run SD "
      f"{dmf['between_run_sd_est']:.3f}"
      + (" (the across-seed SD is accounted for by test-set sampling)." if sampling_dominated else ".") + "\n")
    sec = an["D_minus_F_secondary"]
    w(f"Secondary outcomes, paired D − F: log loss {sec['log_loss']['mean']:+.3f} {_ci(sec['log_loss'])}, "
      f"accuracy {sec['accuracy']['mean']:+.3f} {_ci(sec['accuracy'])}, ECE {sec['ece']['mean']:+.3f} {_ci(sec['ece'])}.\n")
    if "F_long" in an:
        fl = an["F_long"]
        w(f"F with {cfg['conditions']['f_long_step_multiplier']}× the optimisation steps ({len(fl['seeds'])} diagnostic seeds): "
          f"BSS {_ms(fl['bss'])}; "
          f"F_long − F {fl['F_long_minus_F0']['mean']:+.3f} {_ci(fl['F_long_minus_F0'])}; "
          f"D − F_long {fl['D0_minus_F_long']['mean']:+.3f} {_ci(fl['D0_minus_F_long'])}.\n")

    w("## Misalignment λ (H2)\n")
    w("D_λ: a fraction λ of the bound signatures is permuted across episodes of the same window during training "
      "(unbound streams intact). F_λ: the same permutation applied to F's trace stream and signature stream. "
      "Testing always uses correct pairing.\n")
    w("| λ | D_λ BSS | F_λ BSS | D_λ − F₀ | D_λ − F_λ |\n|---|---|---|---|---|")
    for l in lams:
        k = f"{l:g}"
        a, b = lam["D_lambda_minus_F0"][k], lam["D_lambda_minus_F_lambda"][k]
        w(f"| {k} | {_ms(lam['D_by_lambda'][k])} | {_ms(lam['F_by_lambda'][k])} | {a['mean']:+.3f} {_ci(a)} | "
          f"{b['mean']:+.3f} {_ci(b)} |")
    w(f"\nPer-seed slope of BSS on λ: D {lam['D_slope']['mean']:+.3f} {_ci(lam['D_slope'])}, "
      f"F {lam['F_slope']['mean']:+.3f} {_ci(lam['F_slope'])}. D₁ − F₀ = {lam['D1_minus_F0']['mean']:+.3f} "
      f"{_ci(lam['D1_minus_F0'])}; "
      + ("the retained fraction of the D − F advantage is not meaningful, as there is no advantage at λ = 0"
         if it >= 3 and adv0 < sesoi else
         f"D₁ retains {100 * lam['retained_fraction_at_lambda1']:.0f}% of the D − F advantage (H2 requires < 50%)")
      + f". D means decrease monotonically in λ: {'yes' if lam['D_means_monotone_decreasing'] else 'no'}.\n")
    if "D_test_misbound" in an:
        tm = an["D_test_misbound"]
        w(f"D_λ^test (correct training, all test signatures permuted): BSS {_ms(tm['bss'])}; minus F "
          f"{tm['minus_F0']['mean']:+.3f} {_ci(tm['minus_F0'])} (expected below F).\n")

    w(f"## Learning curve (D₀, F₀; {len(an['learning_curve'][str(conf)]['seeds'])} diagnostic seeds)\n")
    if it >= 3:
        w("| budget | training episodes | D BSS | F BSS | D − F, 95% CI | (1 − 2α) CI | reading |\n|---|---|---|---|---|---|---|")
    else:
        w("| budget | training episodes | D BSS | F BSS | D − F |\n|---|---|---|---|---|")
    for b in cfg["data"]["budgets"]:
        lc = an["learning_curve"][str(b)]
        row = (f"| {b} n0 | {lc['n_train_episodes']:,} | {_ms(lc['D'])} | {_ms(lc['F'])} | "
               f"{lc['D_minus_F']['mean']:+.3f} {_ci(lc['D_minus_F'])} |")
        if it >= 3:
            row += f" {_ci(lc['D_minus_F'], 'ci_1m2alpha')} | {lc['decision']['decision']} |"
        w(row)
    w("")

    w("## Inputs for the power analysis\n")
    txt = (f"With the observed SD of the paired difference ({dmf['sd']:.3f} BSS), two-sided α = 0.05/3 and power 0.90, "
           f"detecting an effect of size SESOI = {sesoi} needs ((z₁₋α/₂ + z₀.₉₀) · SD / SESOI)² = "
           f"**{dmf['n_runs_for_sesoi']} paired run(s)** ({dmf['n_runs_for_sesoi_ucl80']} with the 80% upper confidence "
           f"limit of SD). The manuscript's illustration assumed SD = 0.08 (≈ 35 runs). "
           f"For H2, the per-seed λ slope of D has SD {an['lambda']['D_slope']['sd']:.3f} BSS per unit λ; the D₁ − F₀ "
           f"contrast has SD {an['lambda']['D1_minus_F0']['sd']:.3f}.")
    if it >= 3:
        alpha = cfg["meta"]["alpha_two_sided"]
        z_eq = st.norm.ppf(1 - alpha) + st.norm.ppf(1 - (1 - cfg["meta"]["power"]) / 2)
        n_eq = math.ceil((z_eq * dmf["sd"] / sesoi) ** 2)
        n_eq_u = math.ceil((z_eq * dmf["sd_ucl80"] / sesoi) ** 2)
        txt += (f" For the equivalence test (two one-sided tests at α = 0.05/3 against ±{sesoi}, power 0.90 when the "
                f"true difference is 0) the requirement is ((z₁₋α + z₀.₉₅) · SD / SESOI)² = **{n_eq} paired run(s)** "
                f"({n_eq_u} with the 80% upper confidence limit of SD).")
    if abs(dmf["mean"]) > 3 * sesoi:
        txt += (f" The observed D − F effect ({dmf['mean']:+.3f}) is more than three times SESOI, so in this instantiation "
                f"H1 does not drive the run count. The planned 40 runs are set by the equivalence and λ contrasts and by H5, "
                f"and above all by whether a stronger F changes the effect size.")
    w(txt + "\n")

    w("## Verdicts and automatically detected problems\n")
    flags = []
    if not m1["pass"]:
        flags.append("M1 failed in some seeds: the slow state is not path-dependent beyond test–retest noise.")
    if not m2["pass"]:
        flags.append("M2 failed in some seeds: the trace carries too little source information; a D–F null would be uninformative.")
    if not lk["decoder_pass"] or not lk.get("pipeline_pass", True):
        flags.append("Leakage audit failed: source is recoverable without the trace.")
    if f_above_floor < sesoi:
        flags.append(f"F is within SESOI of the no-trace floor (F − F_noz = {f_above_floor:+.3f}). The control uses the trace "
                     f"only marginally, so D − F measures the episode–trace join barrier rather than a graded advantage "
                     f"over a competent control. A stronger generic F, or many more steps or data, is needed before "
                     f"the H1 contrast can be called a fair test (design notes 5 and 8).")
    sd0 = lam["D_by_lambda"][f"{lams[0]:g}"]["sd"]
    mid = [l for l in lams if 0 < l < 1]
    sd_mid = max((lam["D_by_lambda"][f"{l:g}"]["sd"] for l in mid), default=0.0)
    if mid and sd_mid > 3 * sd0:
        flags.append(f"Intermediate misalignment is unstable: the between-seed SD of D_λ reaches {sd_mid:.3f} vs {sd0:.3f} at "
                     f"λ = 0. Some seeds discover the partly wrong binding late in training and others not at all. The λ "
                     f"slope therefore mixes reliance on the binding with learning speed, and the intermediate λ points "
                     f"dominate the variance of the H2 test.")
    steps, every = cfg["training"]["steps"], cfg["training"]["eval_every"]
    late = sum(max(r["curve"], key=lambda c: c[2])[0] >= steps - every for r in R["runs"]["D_lam0"].values())
    if late >= len(seeds) / 2:
        flags.append(f"D is still improving at the end of training: its best checkpoint is in the last two evaluations "
                     f"in {late}/{len(seeds)} seeds. The step budget truncates D (conservative for H1) and makes the "
                     f"intermediate λ points sensitive to it.")
    h2_moot = it >= 3 and adv0 < sesoi
    if lam["retained_fraction_at_lambda1"] >= 0.5 and not h2_moot:
        flags.append("D₁ retains at least half of the D − F advantage: H2 pattern not reproduced.")
    if lam["D_slope"]["ci95"][1] >= 0:
        flags.append("The λ slope for D is not reliably negative.")
    if m2["fraction_of_ceiling_D"]["mean"] > 0.95:
        flags.append("D is at the information ceiling; differences above D are not measurable.")
    big = str(max(cfg["data"]["budgets"]))
    if an["learning_curve"][big]["D_minus_F"]["mean"] < sesoi and it < 3:
        flags.append(f"F catches up with D at {big} n0: the advantage is one of sample efficiency only.")
    if it >= 3:
        w(f"- **H1 (pilot reading, not confirmatory):** {an['H1_decision']['decision']}; D − F = {dmf['mean']:+.3f} BSS, "
          f"(1 − 2α) CI {_ci(dmf, 'ci_1m2alpha')}.")
        w(f"- **H2 (pilot reading):** " + ("not informative: no D − F advantage at λ = 0." if h2_moot else
          f"D slope {lam['D_slope']['mean']:+.3f} per unit λ; D₁ retains {100 * lam['retained_fraction_at_lambda1']:.0f}% "
          f"of the advantage."))
    else:
        w(f"- **H1 (pilot reading, not confirmatory):** D − F = {dmf['mean']:+.3f} BSS, (1 − 2α) CI {_ci(dmf, 'ci_1m2alpha')}; "
          f"{'the interval excludes 0 and the estimate exceeds SESOI' if dmf['ci_1m2alpha'][0] > 0 and dmf['mean'] >= sesoi else 'support criterion not met'}.")
        w(f"- **H2 (pilot reading):** D slope {lam['D_slope']['mean']:+.3f} per unit λ; D₁ retains "
          f"{100 * lam['retained_fraction_at_lambda1']:.0f}% of the advantage.")
    for f in flags:
        w(f"- ⚠ {f}")
    if not flags:
        w("- No automatic flags.")
    w("")
    notes = HERE / cfg["meta"].get("design_notes", "DESIGN_NOTES.md")
    if notes.exists():
        w(notes.read_text(encoding="utf-8"))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
