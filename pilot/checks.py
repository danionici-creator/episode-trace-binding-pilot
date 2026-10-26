"""Manipulation check M2 (information ceiling) and the pipeline leakage audit.

An independent decoder, trained with the correct episode-trace pairing on 10 x the
confirmatory training episodes, estimates how much source information each feature set
carries about novel test episodes. Feature sets:
  full        z_t for every step of e_k, s_k, record, timestamp     (ceiling with z)
  s_only      s_k, record, timestamp                                (what Phi retains)
  no_z        record, query, timestamp, window log context          (ceiling without z = leakage audit)
  control     full features, training labels shuffled               (probe control task)
  misaligned  full features, z and s taken from another episode of the same window
"""
import numpy as np
import torch
import torch.nn as nn

from scoring import score
from sim_env import records_with_kind


def _time_feats(t, n_freq, t_max):
    periods = np.logspace(np.log10(3.0), np.log10(2.0 * t_max), n_freq)
    ang = 2 * np.pi * np.asarray(t, float)[..., None] / periods
    return np.concatenate([np.sin(ang), np.cos(ang)], -1)


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


def episode_features(data, idx, cfg, kind_set, rng=None):
    E = cfg["environment"]
    K, T, nk = E["episodes_per_window"], E["steps_per_episode"], E["n_kinds"]
    z = data["z"][idx].reshape(len(idx), K, T, -1)
    s = data["s"][idx]
    if kind_set == "misaligned":
        perm = np.stack([derangement(K, rng) for _ in idx])
        z = np.take_along_axis(z, perm[:, :, None, None], 1)
        s = np.take_along_axis(s, perm[:, :, None], 1)
    rec = records_with_kind(data["rec"][idx], data["kind"][idx], nk)
    pos = np.broadcast_to(_time_feats(np.arange(K) * T, cfg["self_model"]["time_freqs"], K * T),
                          (len(idx), K, 2 * cfg["self_model"]["time_freqs"]))
    if kind_set in ("full", "control", "misaligned"):
        parts = [z.reshape(len(idx), K, -1), s, rec, pos]
    elif kind_set == "s_only":
        parts = [s, rec, pos]
    elif kind_set == "no_z":
        q = records_with_kind(data["q"][idx], data["kind"][idx], nk)
        ctx = np.broadcast_to(data["rec"][idx].mean(1, keepdims=True), data["rec"][idx].shape)
        parts = [rec, q, pos, ctx]
    else:
        raise ValueError(kind_set)
    X = np.concatenate(parts, -1).reshape(len(idx) * K, -1).astype(np.float32)
    return X, data["y"][idx].reshape(-1).astype(np.int64)


def mlp_decoder(Xtr, ytr, Xte, cfg, seed):
    C = cfg["checks"]
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    n_hold = len(Xtr) // 10
    Xh, yh = torch.as_tensor(Xtr[:n_hold]), torch.as_tensor(ytr[:n_hold])
    Xt, yt = torch.as_tensor(Xtr[n_hold:]), torch.as_tensor(ytr[n_hold:])
    H = C["decoder_hidden"]
    net = nn.Sequential(nn.Linear(Xtr.shape[1], H), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.1), nn.Linear(H, 3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state = np.inf, None
    for _ in range(C["decoder_epochs"]):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(512):
            loss = nn.functional.cross_entropy(net(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            hl = nn.functional.cross_entropy(net(Xh), yh).item()
        if hl < best:
            best, best_state = hl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(torch.as_tensor(Xte)), -1).numpy()


def run_m2_and_leakage(data, cfg, seed):
    torch.set_num_threads(cfg["meta"].get("threads_per_worker", 1))
    rng = np.random.default_rng([seed, 7])
    tr_idx = np.flatnonzero(data["role"] == 3)
    te_idx = np.flatnonzero(data["role"] == 2)
    base = np.bincount(data["y"][tr_idx].ravel(), minlength=3) / data["y"][tr_idx].size
    out = {}
    for fs in ("full", "s_only", "no_z", "control", "misaligned"):
        Xtr, ytr = episode_features(data, tr_idx, cfg, fs, rng)
        Xte, yte = episode_features(data, te_idx, cfg, fs, rng)
        if fs == "control":
            ytr = rng.permutation(ytr)
        if fs == "full" and cfg["checks"].get("m2_eval_val"):
            # iteration 3: the ceiling is also read on the validation windows, where candidates are selected
            va_idx = np.flatnonzero(data["role"] == 1)
            Xva, yva = episode_features(data, va_idx, cfg, fs, rng)
            p_all = mlp_decoder(Xtr, ytr, np.concatenate([Xte, Xva]), cfg, seed)
            p = p_all[: len(Xte)]
            out["full_val"] = score(p_all[len(Xte):], yva, base, cfg["checks"]["ece_bins"])
        else:
            p = mlp_decoder(Xtr, ytr, Xte, cfg, seed)
        out[fs] = score(p, yte, base, cfg["checks"]["ece_bins"])
    out["n_train_episodes"] = int(len(tr_idx) * cfg["environment"]["episodes_per_window"])
    out["delta_bss"] = out["full"]["bss"] - out["no_z"]["bss"]
    return out
