"""Automated audit of the pilot pipeline: pairing, permutations, alignment, isolation, determinism.

    python pilot/audit_checks.py

Runs on a small life simulated with an unused seed (999), so it takes about a minute and touches no
result file. Each check prints PASS or FAIL; the exit code is the number of failures. These checks
verify the implementation's internal logic; they do not replace an independent human code review.
"""
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iter4_common import condition_cfg, load_iter4  # noqa: E402
from run_pilot import simulate_task  # noqa: E402
from self_models import SelfModel, matched_widths  # noqa: E402
from sim_env import Measurement, simulate_life, window_layout  # noqa: E402
from train_sm import _tensors, apply_misbinding, load_split, misbinding_perms  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def small_cfg():
    cfg = condition_cfg(load_iter4(), "R")
    cfg["data"].update({"n0_windows": 4, "n_val_windows": 4, "n_test_windows": 8, "m2_multiplier": 2})
    cfg["environment"]["burn_in_episodes"] = 50
    cfg["checks"]["decoder_epochs"] = 1
    cfg["checks"]["m1_repeats"] = 2
    return cfg


def main():
    cfg = small_cfg()
    E = cfg["environment"]
    K, T = E["episodes_per_window"], E["steps_per_episode"]
    rng = np.random.default_rng(0)

    # 1. misalignment permutations
    for lam in (0.0, 0.5, 1.0):
        p = misbinding_perms(200, K, lam, rng)
        moved = (p != np.arange(K)).sum(1)
        bij = all(sorted(row) == list(range(K)) for row in p)
        want = int(round(lam * K)) if int(round(lam * K)) >= 2 else 0
        check(f"misbinding_perms lambda={lam}: bijection within window, {want} episodes moved",
              bij and np.all(moved == want), f"moved {sorted(set(moved.tolist()))}")

    # 2. life: determinism, seed isolation, episode alignment of z and s
    a, _ = simulate_life(cfg, 999, with_m1=False)
    b, _ = simulate_life(cfg, 999, with_m1=False)
    c, _ = simulate_life(cfg, 998, with_m1=False)
    check("same seed reproduces the life exactly", all(np.array_equal(a[k], b[k]) for k in a))
    check("different seeds give different lives", not np.array_equal(a["z"], c["z"]))
    meas = Measurement(cfg["measurement"], 2 * E["d_out"] * E["d_in"])
    w, k = 3, 5
    check("s_k equals Phi of the z segment of its own episode",
          np.allclose(meas.Phi(a["z"][w, k * T:(k + 1) * T].astype(np.float64)), a["s"][w, k], atol=1e-4))
    check("query is the record plus noise (same episode), not another episode's record",
          np.corrcoef(a["q"][:, :, 0].ravel(), a["rec"][:, :, 0].ravel())[0, 1] > 0.9
          and abs(np.corrcoef(a["q"][:, 1:, 0].ravel(), a["rec"][:, :-1, 0].ravel())[0, 1]) < 0.2)

    # 3. splits and normalisation
    with tempfile.TemporaryDirectory() as td:
        life = simulate_task(cfg, 999, td, False)
        d = dict(np.load(life["path"]))
        sp = load_split(life["path"], cfg, 999, 1)
        role = d["role"]
        n_tr = cfg["data"]["n0_windows"]
        check("train split has budget x n0 windows, all of training role", len(sp["train"]["y"]) == n_tr)
        check("validation and test splits match their roles",
              len(sp["val"]["y"]) == int((role == 1).sum()) and len(sp["test"]["y"]) == int((role == 2).sum()))
        idx_tr = np.random.default_rng([999, 6]).permutation(np.flatnonzero(role == 0))[:n_tr]
        sets = [set(idx_tr), set(np.flatnonzero(role == 1)), set(np.flatnonzero(role == 2))]
        check("train, validation and test windows are disjoint",
              not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]))
        zt = d["z"][idx_tr].reshape(-1, d["z"].shape[-1])
        check("z is standardised with training statistics only",
              np.allclose(sp["train"]["z"].reshape(-1, zt.shape[-1]).mean(0), 0, atol=1e-4))
        tr = sp["train"]
        n = len(tr["y"])
        ep_s = tr["s"].reshape(n * K, -1)
        ep_z = tr["z"].reshape(n * K, T, -1)
        check("episode view keeps (z segment, s_k, y) of the same episode together",
              np.array_equal(ep_s[2 * K + 3], tr["s"][2, 3]) and np.array_equal(ep_z[2 * K + 3], tr["z"][2, 3 * T:4 * T])
              and tr["y"].reshape(-1)[2 * K + 3] == tr["y"][2, 3])

        # 4. misalignment touches only what it should
        perms = misbinding_perms(n, K, 1.0, np.random.default_rng(1))
        dD = apply_misbinding(tr, "D", perms, T)
        check("D_lambda permutes only the bound copy (z and s unchanged)",
              np.array_equal(dD["z"], tr["z"]) and np.array_equal(dD["s"], tr["s"]) and not np.array_equal(dD["sb"], tr["sb"]))
        dF = apply_misbinding(tr, "F", perms, T)
        k0 = 0
        j = perms[0, k0]
        check("F_lambda moves z segment and s_k together, bound copy untouched",
              np.array_equal(dF["z"][0, k0 * T:(k0 + 1) * T], tr["z"][0, j * T:(j + 1) * T])
              and np.array_equal(dF["s"][0, k0], tr["s"][0, j]) and np.array_equal(dF["sb"], tr["sb"]))
        t = _tensors(tr, True)
        check("F_noz receives no z, no s and no bound s", all(float(t[x].abs().sum()) == 0 for x in ("z", "s", "sb")))

        # 5. architectures: the binding is the only structural difference; parameter parity
        dims = {"rec": tr["rec"].shape[-1], "z": tr["z"].shape[-1], "s": tr["s"].shape[-1]}
        widths = matched_widths(dims, cfg["self_model"], window_layout(cfg))
        mD = SelfModel(dims, cfg["self_model"], window_layout(cfg), True, widths["d_ff_D"])
        mF = SelfModel(dims, cfg["self_model"], window_layout(cfg), False, widths["d_ff_F"])
        extra = {nm.split(".")[0] for nm, _ in mD.named_parameters()} - {nm.split(".")[0] for nm, _ in mF.named_parameters()}
        check("D differs from F only by the binding map s_bind", extra == {"s_bind"}, str(extra))
        check("active parameters matched within 2%", abs(widths["rel_mismatch"]) < 0.02, f"{widths['rel_mismatch']:+.4%}")

        # 6. where the co-timed objective reads from (documents the T32 shortcut and its iteration-5 removal)
        torch.manual_seed(0)
        x = {k: torch.as_tensor(v[:4], dtype=torch.float32) for k, v in tr.items() if k != "y"}
        sb = x["sb"].clone().requires_grad_(True)
        _, aux = mD(x["rec"], x["z"], x["q"], x["s"], sb, return_aux=True)
        aux.sum().backward()
        check("D: co-timed head depends on the bound copy (shortcut present in iterations 3-4)",
              float(sb.grad.abs().sum()) > 0)
        sb2 = x["sb"].clone().requires_grad_(True)
        aux0 = mD.aux_head(mD.encode(x["rec"], x["z"], x["s"], torch.zeros_like(sb2) + 0 * sb2)[:, :K])
        aux0.sum().backward()
        check("D with aux_prebind: co-timed head independent of the bound copy", float(sb2.grad.abs().sum()) == 0)
        sb3 = x["sb"].clone().requires_grad_(True)
        _, aux3 = mF(x["rec"], x["z"], x["q"], x["s"], sb3, return_aux=True)
        aux3.sum().backward()
        check("F: never reads the bound copy", sb3.grad is None or float(sb3.grad.abs().sum()) == 0)

    print(f"\n{len(FAILS)} failure(s)" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
