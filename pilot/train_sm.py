"""Training and evaluation of one self-model condition on one seed."""
import math
import time

import numpy as np
import torch
import torch.nn as nn

from checks import derangement
from scoring import brier_items, score
from self_models import SelfModel
from sim_env import records_with_kind, window_layout


def misbinding_perms(n_windows, K, lam, rng):
    """Per window, a fraction lam of episodes exchange signatures (a derangement among them).

    Signatures stay inside their own window, so the marginal statistics of z are preserved.
    """
    m = int(round(lam * K))
    perms = np.tile(np.arange(K), (n_windows, 1))
    if m >= 2:
        for w in range(n_windows):
            sel = rng.choice(K, m, replace=False)
            perms[w, sel] = sel[derangement(m, rng)]
    return perms


def load_split(path, cfg, seed, budget):
    d = dict(np.load(path))
    E = cfg["environment"]
    nk = E["n_kinds"]
    role = d["role"]
    order = np.random.default_rng([seed, 6]).permutation(np.flatnonzero(role == 0))
    split = {"train": order[: budget * cfg["data"]["n0_windows"]],
             "val": np.flatnonzero(role == 1), "test": np.flatnonzero(role == 2)}
    rec = records_with_kind(d["rec"], d["kind"], nk)
    q = records_with_kind(d["q"], d["kind"], nk)
    tr = split["train"]
    z_mu, z_sd = d["z"][tr].reshape(-1, d["z"].shape[-1]).mean(0), d["z"][tr].reshape(-1, d["z"].shape[-1]).std(0)
    s_mu, s_sd = d["s"][tr].reshape(-1, d["s"].shape[-1]).mean(0), d["s"][tr].reshape(-1, d["s"].shape[-1]).std(0)
    s = (d["s"] - s_mu) / s_sd
    arrays = {"rec": rec, "q": q, "z": (d["z"] - z_mu) / z_sd, "s": s, "sb": s, "y": d["y"].astype(np.int64)}
    return {k: {n: a[idx] for n, a in arrays.items()} for k, idx in split.items()}


def apply_misbinding(part, arch, perms, T):
    part = dict(part)
    if arch == "D":      # D_lambda: wrong s_k bound to the record of e_k; unbound streams intact
        part["sb"] = np.take_along_axis(part["sb"], perms[:, :, None], 1)
    elif arch == "F":    # F_lambda: z segments and s_k of e_k exchanged across episodes; timestamps intact
        n, KT, dz = part["z"].shape
        z = part["z"].reshape(n, KT // T, T, dz)
        part["z"] = np.take_along_axis(z, perms[:, :, None, None], 1).reshape(n, KT, dz)
        part["s"] = np.take_along_axis(part["s"], perms[:, :, None], 1)
    return part


def _tensors(part, noz):
    t = {k: torch.as_tensor(v, dtype=torch.long if k == "y" else torch.float32) for k, v in part.items()}
    if noz:
        for k in ("z", "s", "sb"):
            t[k] = torch.zeros_like(t[k])
    return t


@torch.no_grad()
def predict(model, t):
    model.eval()
    return torch.softmax(model(t["rec"], t["z"], t["q"], t["s"], t["sb"]), -1).reshape(-1, 3).numpy()


def run_job(job, cfg, widths):
    torch.set_num_threads(cfg["meta"].get("threads_per_worker", 1))
    t0 = time.time()
    E, Tc, mcfg = cfg["environment"], cfg["training"], cfg["self_model"]
    K, T = E["episodes_per_window"], E["steps_per_episode"]
    seed, arch, lam, budget, hp = job["seed"], job["arch"], job["lam"], job["budget"], job["hp"]
    base_arch = "F" if arch == "Fnoz" else arch
    sp = load_split(job["data_path"], cfg, seed, budget)
    if lam > 0:
        rng = np.random.default_rng([seed, 8, int(round(lam * 1000))])
        for part in ("train", "val"):   # the intervention covers everything seen during training
            perms = misbinding_perms(len(sp[part]["y"]), K, lam, rng)
            sp[part] = apply_misbinding(sp[part], base_arch, perms, T)
    tt = {k: _tensors(v, arch == "Fnoz") for k, v in sp.items()}
    y_train = sp["train"]["y"]
    base = np.bincount(y_train.ravel(), minlength=3) / y_train.size

    torch.manual_seed(seed * 1000 + budget)
    layout = window_layout(cfg)
    dims = {"rec": tt["train"]["rec"].shape[-1], "z": tt["train"]["z"].shape[-1], "s": tt["train"]["s"].shape[-1]}
    d_ff = widths["d_ff_D"] if base_arch == "D" else widths["d_ff_F"]
    model = SelfModel(dims, mcfg, layout, base_arch == "D", d_ff)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps, warm = job.get("steps", Tc["steps"]), 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    rng_b = np.random.default_rng([seed, 9, budget])   # identical batch order for every condition
    gen = torch.Generator().manual_seed(seed * 1000 + budget + 1)
    n_tr, B = len(y_train), Tc["batch_windows"]
    tr = tt["train"]
    # episode-level view of the training set: (record, z segment, s, y) stay together
    ep = {"rec": tr["rec"].reshape(n_tr * K, -1), "q": tr["q"].reshape(n_tr * K, -1),
          "s": tr["s"].reshape(n_tr * K, -1), "sb": tr["sb"].reshape(n_tr * K, -1), "y": tr["y"].reshape(-1),
          "z": tr["z"].reshape(n_tr * K, T, -1)}
    d_r, sq = E["d_record"], E["query_noise"]
    # iteration-2 options (absent in iteration 1): window-length curriculum, co-timed auxiliary loss
    curriculum = Tc.get("curriculum")          # [[fraction of steps at which the stage ends, K], ...]
    aux_w = hp.get("aux_cotimed_weight", Tc.get("aux_cotimed_weight", 0.0))   # iteration 3: tunable
    # iteration 4: generic, join-agnostic objective (masked-token reconstruction over all streams)
    objective = Tc.get("aux_objective", "cotimed")
    if objective == "masked":
        aux_w = hp.get("aux_weight", Tc.get("aux_weight", 0.0))
    assert not curriculum or Tc["recompose_windows"], "the curriculum assembles windows from episodes"
    best, best_state, curve = -np.inf, None, []
    for step in range(1, steps + 1):
        model.train()
        Kc, off = K, 0
        if curriculum:
            # short windows first: with K = 1 the episode-trace join is trivial; K then grows to the
            # full window. Episodes per step stay constant; slot offset drawn so every timestamp is seen
            Kc = next(k for frac, k in curriculum if step <= frac * steps)
            if Kc < K:
                off = int(rng_b.integers(0, K - Kc + 1))
        Bc = B * K // Kc
        if Tc["recompose_windows"]:
            # each presented window is assembled from K random training episodes, which
            # removes whole-window memorisation; episodes keep their own trace segment
            e = torch.as_tensor(rng_b.integers(0, n_tr * K, (Bc, Kc)))
            rec, q, s, sb, yb = ep["rec"][e], ep["q"][e], ep["s"][e], ep["sb"][e], ep["y"][e]
            z = ep["z"][e].reshape(Bc, Kc * T, -1)
        else:
            idx = torch.as_tensor(rng_b.integers(0, n_tr, B))
            rec, q, s, sb, yb, z = (tr[k][idx] for k in ("rec", "q", "s", "sb", "y", "z"))
        if Tc["redraw_descriptors"]:
            # record descriptors carry no source information; redrawing them at every
            # presentation removes the record -> label memorisation shortcut
            r = torch.randn(rec.shape[:-1] + (d_r,), generator=gen)
            rec = torch.cat([r, rec[..., d_r:]], -1)
            q = torch.cat([r + sq * torch.randn(r.shape, generator=gen), q[..., d_r:]], -1)
        if aux_w and objective == "masked":
            # clean pass for the attribution loss; separate encoder-only pass in which a random 15% of
            # all tokens (records, traces, signatures alike) are masked and their inputs reconstructed
            logits = model(rec, z, q, s, sb, offset=off)
            ce = nn.functional.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1))
            n_tok = Kc + Kc * T + (Kc if mcfg["signature_stream"] else 0)
            m = torch.rand((Bc, n_tok), generator=gen) < Tc["mask_rate"]
            preds = model.reconstruct(rec, z, s, sb, m, offset=off)
            targets = [rec, z] + ([s] if mcfg["signature_stream"] else [])
            err = torch.cat([((p_ - t_) ** 2).mean(-1) for p_, t_ in zip(preds, targets)], 1)
            loss = ce + aux_w * (err * m).sum() / m.sum().clamp(min=1)
        elif aux_w and Tc.get("aux_prebind", False):
            # iteration 5: the co-timed target is predicted from a second encoder pass in which the bound
            # copy is zeroed, so that in D the objective cannot be met by reading s_k back from its own
            # record token; in F (no binding) the second pass equals the first, so F is unchanged
            logits = model(rec, z, q, s, sb, offset=off)
            aux = model.aux_head(model.encode(rec, z, s, torch.zeros_like(sb), off)[:, :Kc])
            ce = nn.functional.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1))
            loss = ce + aux_w * nn.functional.mse_loss(aux, s)
        elif aux_w:
            # self-supervised, label-free: each encoded record token must reconstruct the signature
            # token that carries the same episode timestamp, which rewards attention across co-timed tokens
            logits, aux = model(rec, z, q, s, sb, offset=off, return_aux=True)
            ce = nn.functional.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1))
            loss = ce + aux_w * nn.functional.mse_loss(aux, s)
        else:
            logits = model(rec, z, q, s, sb, offset=off)
            loss = ce = nn.functional.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % Tc["eval_every"] == 0:
            pv = predict(model, tt["val"])
            v = score(pv, sp["val"]["y"].ravel(), base)["bss"]
            curve.append((step, float(ce.item()), v))
            if v > best:
                best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)

    y_test = sp["test"]["y"].ravel()
    p = predict(model, tt["test"])
    out = {**{k: job[k] for k in ("seed", "arch", "lam", "budget", "tag")}, "hp": hp,
           "val_bss": float(best), "test": score(p, y_test, base, cfg["checks"]["ece_bins"]),
           "curve": curve, "brier_items": brier_items(p, y_test).astype(np.float32),
           "brier_ref_items": brier_items(np.broadcast_to(base, p.shape), y_test).astype(np.float32)}
    if base_arch == "D" and lam == 0 and job.get("test_misbinding"):
        rng = np.random.default_rng([seed, 10])
        perms = misbinding_perms(len(sp["test"]["y"]), K, 1.0, rng)
        tmis = _tensors(apply_misbinding(sp["test"], "D", perms, T), False)
        out["test_misbound"] = score(predict(model, tmis), y_test, base, cfg["checks"]["ece_bins"])
    out["seconds"] = time.time() - t0
    return out
