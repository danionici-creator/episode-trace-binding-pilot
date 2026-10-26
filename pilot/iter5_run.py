"""Iteration 5: exploratory diagnostic of the K = 32 crossover -> results_iter5.json, REPORT_iter5.md.

    python pilot/iter5_run.py                 # checks both freeze records, then runs (resumable)
    python pilot/iter5_run.py --report-only   # re-analyse results_iter5_runs.jsonl and re-render the report

Finished runs are appended to results_iter5_runs.jsonl as they complete, so an interrupted run resumes
where it stopped. The run refuses to start unless config_iter4.yaml and config_iter5.yaml still have
the SHA-256 recorded when each was frozen.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iter4_common import condition_cfg, load_iter4, sha256  # noqa: E402
from run_pilot import _init_worker, decision, describe, paired, simulate_task  # noqa: E402
from self_models import matched_widths  # noqa: E402
from sim_env import window_layout  # noqa: E402
from train_sm import run_job  # noqa: E402

CONF = 8   # primary budget, multiples of n0


def dims_of(cfg):
    E = cfg["environment"]
    return {"rec": E["d_record"] + E["n_kinds"], "z": cfg["measurement"]["d_z"], "s": 2 * cfg["measurement"]["d_z"]}


def load_iter5():
    return yaml.safe_load((HERE / "config_iter5.yaml").read_text(encoding="utf-8"))


def check_freezes(c5):
    f4 = json.loads((HERE / "config_iter4.freeze.json").read_text(encoding="utf-8"))
    assert sha256(HERE / "config_iter4.yaml") == f4["sha256"] == c5["base_config_sha256"], "config_iter4.yaml changed"
    f5 = json.loads((HERE / "config_iter5.freeze.json").read_text(encoding="utf-8"))
    assert sha256(HERE / "config_iter5.yaml") == f5["sha256"], "config_iter5.yaml changed after freezing"
    return f5


def cond_cfg(c4, c5, name):
    cond = c5["conditions"][name]
    cfg = condition_cfg(c4, cond["from"], cond.get("steps"))
    cfg["meta"]["iteration"] = 5
    if cond.get("aux_prebind"):
        cfg["training"]["aux_prebind"] = True
    return cfg


def cond_seeds(c5, name, arch):
    cond = c5["conditions"][name]
    key = cond["archs_seeds"][arch] if "archs_seeds" in cond else cond["seeds"]
    return c5["seeds"][key]


def cond_archs(c5, name):
    cond = c5["conditions"][name]
    return list(cond["archs_seeds"]) if "archs_seeds" in cond else cond["archs"]


def hp_of(c4, c5, name, arch):
    cond = c5["conditions"][name]
    if "hp" in cond:
        return dict(cond["hp"])
    return dict(c4["conditions"][cond["from"]]["hp"]["F" if arch == "Fnoz" else arch])


def read_runs(path):
    runs = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                runs[r["tag"]] = r
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    c4, c5 = load_iter4(), load_iter5()
    freeze = check_freezes(c5)
    runs_path = HERE / c5["meta"]["output_runs"]
    lives_path = HERE / "results_iter5_lives.json"
    if args.report_only:
        finish(c4, c5, freeze, read_runs(runs_path), json.loads(lives_path.read_text(encoding="utf-8")), {})
        return
    nw, nt = c5["meta"]["n_workers"], c5["meta"]["threads_per_worker"]
    t_start = time.time()

    # 1. lives: K = 32 for seeds 11-20, K = 8 for the audit seeds (deterministic in the seed)
    lives = json.loads(lives_path.read_text(encoding="utf-8")) if lives_path.exists() else {}
    need = [("k32", s, cond_cfg(c4, c5, "T32_ext")) for s in c5["seeds"]["main"]]
    need += [("k8", s, cond_cfg(c4, c5, "R_audit")) for s in c5["seeds"]["short"]]
    todo = []
    for k, s, cfg in need:
        cache = HERE / "cache" / f"iter5_{k}"
        cache.mkdir(parents=True, exist_ok=True)
        if f"{k}|{s}" not in lives or not (HERE / lives[f"{k}|{s}"]["path"]).exists():
            todo.append((k, s, cfg, str(cache)))
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(nw, initializer=_init_worker, initargs=(nt,)) as ex:
            futs = {ex.submit(simulate_task, cfg, s, cache, True): (k, s) for k, s, cfg, cache in todo}
            for i, f in enumerate(as_completed(futs), 1):
                k, s = futs[f]
                r = f.result()
                rel = Path(r["path"]).resolve().relative_to(HERE).as_posix()   # relative: no local paths in the archive
                lives[f"{k}|{s}"] = {"seed": s, "path": rel, "m1": r["m1"], "m2": r["m2"], "seconds": r["seconds"]}
                lives_path.write_text(json.dumps(lives), encoding="utf-8")
                print(f"  [lives] {i}/{len(todo)} {k} seed {s} ({time.time() - t0:.0f}s)", flush=True)
    t_lives = time.time() - t0

    # 2. self-model runs, in the priority order of config_iter5.yaml; finished runs are appended
    done = read_runs(runs_path)
    jobs = []
    for name in c5["conditions"]:
        cfg = cond_cfg(c4, c5, name)
        widths = matched_widths(dims_of(cfg), cfg["self_model"], window_layout(cfg))
        k = "k32" if cfg["environment"]["episodes_per_window"] == 32 else "k8"
        for arch in cond_archs(c5, name):
            for s in cond_seeds(c5, name, arch):
                tag = f"{name}|{arch}|{s}"
                if tag in done:
                    continue
                job = {"seed": s, "arch": arch, "lam": 0.0, "budget": CONF, "tag": tag,
                       "hp": hp_of(c4, c5, name, arch), "data_path": str(HERE / lives[f"{k}|{s}"]["path"])}
                jobs.append((job, cfg, widths))
    print(f"{len(jobs)} self-model runs to do ({len(done)} already finished)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(nw, initializer=_init_worker, initargs=(nt,)) as ex:
        futs = [ex.submit(run_job, *j) for j in jobs]
        with runs_path.open("a", encoding="utf-8") as fh:
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                keep = {k: v for k, v in r.items() if k not in ("brier_items", "brier_ref_items")}
                fh.write(json.dumps(keep) + "\n")
                fh.flush()
                done[r["tag"]] = keep
                print(f"  [main] {i}/{len(jobs)} {r['tag']} test BSS {r['test']['bss']:+.3f} "
                      f"({r['seconds']:.0f}s; {time.time() - t0:.0f}s elapsed)", flush=True)
    timing = {"lives_s": t_lives, "main_s": time.time() - t0, "total_s": time.time() - t_start}
    finish(c4, c5, freeze, done, lives, timing)


# ------------------------------------------------------------------------------------ analysis
def onset_step(curve, thr):
    for step, _, v in curve:
        if v >= thr:
            return step
    return None


def finish(c4, c5, freeze, runs, lives, timing):
    A = c5["analysis"]
    base = cond_cfg(c4, c5, "R_audit")
    sesoi = base["meta"]["sesoi_bss"]
    rng = np.random.default_rng(12345)
    R4 = json.loads((HERE / "results_iter4.json").read_text(encoding="utf-8"))["runs"]

    def get(cond, arch, seed):
        """Test BSS and curve of one run: iteration 5, or iteration 4 for T32 / G32 seeds 11-13."""
        r = runs.get(f"{cond}|{arch}|{seed}")
        if r is None and cond in ("T32_ext", "G32_ext"):
            r4 = R4[cond[:3]].get(f"{arch}|{CONF}|0", {}).get(str(seed))
            if r4 is not None:
                return {"bss": r4["test"]["bss"], "curve": r4.get("curve", []), "source": "iteration 4"}
        if r is None:
            return None
        return {"bss": r["test"]["bss"], "curve": r.get("curve", []), "source": "iteration 5"}

    ceil32 = {s: lives[f"k32|{s}"]["m2"]["full"]["bss"] for s in c5["seeds"]["main"] if f"k32|{s}" in lives}

    def block(dcond, fcond, seeds, darch="D", farch="F"):
        rows, D, F, ok = [], [], [], []
        for s in seeds:
            d, f = get(dcond, darch, s), get(fcond, farch, s)
            if d is None or f is None:
                continue
            c = ceil32.get(s)
            rows.append({"seed": s, "D": d["bss"], "F": f["bss"], "ceiling": c,
                         "D_onset": d["bss"] >= A["onset_bss"], "F_onset": f["bss"] >= A["onset_bss"],
                         "D_onset_step": onset_step(d["curve"], A["onset_val_bss"]),
                         "F_onset_step": onset_step(f["curve"], A["onset_val_bss"]),
                         "F_share": f["bss"] / c if c else None, "sources": [d["source"], f["source"]]})
            D.append(d["bss"])
            F.append(f["bss"])
            ok.append(rows[-1]["D_onset"] and rows[-1]["F_onset"])
        out = {"rows": rows, "n": len(rows)}
        if len(rows) >= 2:
            pc = paired(D, F, base, rng)
            out.update({"D": describe(D), "F": describe(F), "D_minus_F": pc, "decision": decision(pc, sesoi)})
        out["P_onset_D"] = float(np.mean([r["D_onset"] for r in rows])) if rows else None
        out["P_onset_F"] = float(np.mean([r["F_onset"] for r in rows])) if rows else None
        Dj = [r["D"] for r, k in zip(rows, ok) if k]
        Fj = [r["F"] for r, k in zip(rows, ok) if k]
        out["joint_onset_n"] = len(Dj)
        if len(Dj) >= 2:
            out["joint_onset_D_minus_F"] = paired(Dj, Fj, base, rng)
        return out

    main, ext = c5["seeds"]["main"], c5["seeds"]["short"]
    res = {
        "T32_10seeds": block("T32_ext", "T32_ext", main),
        "G32_10seeds": block("G32_ext", "G32_ext", main),
        "T32_pre_vs_T32F": block("T32_pre", "T32_ext", main),
        "T32_pre_vs_T32_D": block("T32_pre", "T32_ext", main, "D", "D"),
        "T32_R": block("T32_R", "T32_R", main),
        "T32_R_pre_vs_T32_R_F": block("T32_R_pre", "T32_R", main),
        "N32": {"D": [get("N32", "D", s) and get("N32", "D", s)["bss"] for s in main],
                "F": [get("N32", "F", s) and get("N32", "F", s)["bss"] for s in ext]},
        "T32_long_D_vs_4800": block("T32_long", "T32_ext", ext, "D", "D"),
        "G32_long_F_vs_4800": block("G32_long", "G32_ext", ext, "F", "F"),
    }
    # F_noz floor at K = 32 (iteration 4 seeds 11-13 and T32_ext seeds 14-20)
    res["Fnoz_k32"] = describe([x["bss"] for x in (get("T32_ext", "Fnoz", s) for s in main) if x])
    # reproducibility audit: re-run of R on seeds 11-13 against iteration 4
    audit = []
    for s in ext:
        for arch in ("D", "F"):
            r = runs.get(f"R_audit|{arch}|{s}")
            r4 = R4["R"][f"{arch}|{CONF}|0"].get(str(s))
            if r and r4:
                audit.append({"seed": s, "arch": arch, "rerun": r["test"]["bss"], "recorded": r4["test"]["bss"],
                              "abs_diff": abs(r["test"]["bss"] - r4["test"]["bss"])})
    res["R_audit"] = {"rows": audit, "max_abs_diff": max((a["abs_diff"] for a in audit), default=None),
                      "pass": bool(audit) and all(a["abs_diff"] <= A["audit_tolerance"] for a in audit)}
    # deterministic re-simulation of the lives: ceilings against iteration 4
    chk = json.loads((HERE / "results_iter4.json").read_text(encoding="utf-8"))["checks"]
    life_audit = []
    for k in ("k8", "k32"):
        seeds4, vals4 = chk[k]["seeds"], chk[k]["M2_ceiling"]["values"]
        for s, v in zip(seeds4, vals4):
            if f"{k}|{s}" in lives:
                life_audit.append({"k": k, "seed": s, "recorded": v, "rerun": lives[f"{k}|{s}"]["m2"]["full"]["bss"]})
    res["lives_audit"] = life_audit
    res["ceiling_k32"] = describe(list(ceil32.values())) if ceil32 else None
    R = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "freeze": freeze, "config": c5, "timing_s": timing,
         "environment_info": {"python": platform.python_version(), "torch": __import__("torch").__version__,
                              "numpy": np.__version__},
         "analysis": res, "n_runs": len(runs)}
    (HERE / c5["meta"]["output_results"]).write_text(json.dumps(R, indent=1), encoding="utf-8")
    (HERE / c5["meta"]["output_report"]).write_text(render(R), encoding="utf-8")
    print("analysis written", flush=True)


def _f(x, p=3):
    return "—" if x is None else f"{x:+.{p}f}"


def render(R):
    a, fr = R["analysis"], R["freeze"]
    L = ["# Iteration 5 — exploratory diagnostic of the K = 32 crossover", "",
         f"**Frozen configuration:** `config_iter5.yaml`, SHA-256 `{fr['sha256']}`, frozen {fr['timestamp']}, "
         "before any run of this iteration. Exploratory; see `DESIGN_NOTES_iter5.md` for the a priori readings.", ""]

    def table(title, b, dl="D", fl="F"):
        L.extend([f"## {title}", "", f"| Seed | {dl} | {fl} | ceiling | {dl} onset step | {fl} onset step |",
                  "|---|---|---|---|---|---|"])
        for r in b["rows"]:
            L.append(f"| {r['seed']} | {_f(r['D'])} | {_f(r['F'])} | {_f(r['ceiling'])} | "
                     f"{r['D_onset_step'] or '—'} | {r['F_onset_step'] or '—'} |")
        if "D_minus_F" in b:
            pc = b["D_minus_F"]
            L.append("")
            L.append(f"n = {b['n']}; {dl} {b['D']['mean']:+.3f}, {fl} {b['F']['mean']:+.3f}; {dl} − {fl} {pc['mean']:+.3f} "
                     f"[{pc['ci95'][0]:+.3f}, {pc['ci95'][1]:+.3f}], SD {pc['sd']:.3f}; "
                     f"P(onset) {dl} {b['P_onset_D']:.2f}, {fl} {b['P_onset_F']:.2f}; joint onset n = {b['joint_onset_n']}"
                     + (f", {dl} − {fl} given joint onset {b['joint_onset_D_minus_F']['mean']:+.3f} "
                        f"[{b['joint_onset_D_minus_F']['ci95'][0]:+.3f}, {b['joint_onset_D_minus_F']['ci95'][1]:+.3f}]"
                        if "joint_onset_D_minus_F" in b else ""))
        L.append("")

    table("T32 on ten seeds (seeds 11-13 from iteration 4)", a["T32_10seeds"])
    table("G32 on ten seeds (seeds 11-13 from iteration 4)", a["G32_10seeds"])
    table("T32_pre: D with the bound copy zeroed for the co-timed target, against T32 F", a["T32_pre_vs_T32F"], "D (pre)", "F (T32)")
    table("T32_pre against T32: D only", a["T32_pre_vs_T32_D"], "D (pre)", "D (T32)")
    table("T32_R: co-timed objective with iteration-3 hyperparameters", a["T32_R"])
    table("T32_R_pre: as T32_R with the bound copy zeroed for the target", a["T32_R_pre_vs_T32_R_F"], "D (pre)", "F (T32_R)")
    table("T32 D with 9,600 steps against 4,800 steps", a["T32_long_D_vs_4800"], "D 9600", "D 4800")
    table("G32 F with 9,600 steps against 4,800 steps", a["G32_long_F_vs_4800"], "F 9600", "F 4800")
    L.extend(["## N32: no auxiliary objective at K = 32", "",
              "D (seeds 11-20): " + ", ".join(_f(x) for x in a["N32"]["D"]),
              "", "F (seeds 11-13): " + ", ".join(_f(x) for x in a["N32"]["F"]), ""])
    if a.get("Fnoz_k32"):
        L.extend([f"F_noz at K = 32: {a['Fnoz_k32']['mean']:+.3f} ± {a['Fnoz_k32']['sd']:.3f} (n = {a['Fnoz_k32']['n']})", ""])
    L.extend(["## Reproducibility audit", "", "| Seed | Arch | re-run | recorded | |diff| |", "|---|---|---|---|---|"])
    for r in a["R_audit"]["rows"]:
        L.append(f"| {r['seed']} | {r['arch']} | {r['rerun']:+.4f} | {r['recorded']:+.4f} | {r['abs_diff']:.4f} |")
    L.extend(["", f"Pass (all within tolerance): {a['R_audit']['pass']}; max |diff| {a['R_audit']['max_abs_diff']}", "",
              "| Window | Seed | ceiling re-run | ceiling recorded |", "|---|---|---|---|"])
    for r in a["lives_audit"]:
        L.append(f"| {r['k']} | {r['seed']} | {r['rerun']:+.4f} | {r['recorded']:+.4f} |")
    L.extend(["", f"Timing (s): {R['timing_s']}", ""])
    return "\n".join(L)


if __name__ == "__main__":
    main()
