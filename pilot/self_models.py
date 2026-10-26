"""Self-models D (explicit episode-trace binding) and F (information control).

Both share one backbone, a memory-augmented transformer: a self-attention encoder over
the window's log tokens (one per episode record, with its timestamp), trace tokens (one
per step: z_t, timestamp, episode-boundary flag) and, if signature_stream is set,
signature tokens (one s_k per episode, with the episode's timestamp), followed by
cross-attention readout blocks in which each query (a retained proposition) attends to
the encoded memory. F receives exactly these streams; the episode-trace pairing is
available to it only through timestamps and boundaries. D receives the same streams plus
the binding B: s_k is written into the log token of episode k, so the association is
given as structure. F is compensated by feed-forward width to match D's active parameters.
"""
import math

import torch
import torch.nn as nn


def time_features(t, n_freq, t_max):
    periods = torch.logspace(math.log10(3.0), math.log10(2.0 * t_max), n_freq)
    ang = 2.0 * math.pi * t[..., None].float() / periods
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class CrossBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff, p):
        super().__init__()
        self.ln_q, self.ln_m, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(p), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, q, mem):
        m = self.ln_m(mem)
        a, _ = self.attn(self.ln_q(q), m, m, need_weights=False)
        q = q + self.drop(a)
        return q + self.drop(self.ff(self.ln_f(q)))


class SelfModel(nn.Module):
    def __init__(self, dims, mcfg, layout, binding, d_ff):
        super().__init__()
        d, h, p = mcfg["d_model"], mcfg["n_heads"], mcfg["dropout"]
        self.binding = binding
        self.signature_stream = mcfg["signature_stream"]
        self.rec_in = nn.Linear(dims["rec"], d)   # shared by records and queries (same content space)
        self.z_in = nn.Linear(dims["z"], d)
        self.time_in = nn.Linear(2 * mcfg["time_freqs"], d)
        self.type_query = nn.Parameter(torch.zeros(d))
        self.type_log = nn.Parameter(torch.zeros(d))
        self.type_trace = nn.Parameter(torch.zeros(d))
        self.bnd = nn.Parameter(torch.zeros(d))
        if self.signature_stream:   # s_k as its own token, stamped with the episode's start time
            self.s_in = nn.Linear(dims["s"], d)
            self.type_sig = nn.Parameter(torch.zeros(d))
        if binding:                 # B: s_k written into the record token of e_k
            self.s_bind = nn.Linear(dims["s"], d)
        layer = nn.TransformerEncoderLayer(d, h, d_ff, p, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, mcfg["enc_layers"], enable_nested_tensor=False)
        self.readout = nn.ModuleList(CrossBlock(d, h, d_ff, p) for _ in range(mcfg["readout_layers"]))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3)
        if mcfg.get("aux_head", False):   # co-timed reconstruction head (training only, same in D and F)
            self.aux_head = nn.Linear(d, dims["s"])
        if mcfg.get("mask_head", False):  # generic masked-token reconstruction (training only, same in D and F)
            self.mask_emb = nn.Parameter(torch.zeros(d))
            self.rec_head = nn.Linear(d, dims["rec"])
            self.z_head = nn.Linear(d, dims["z"])
            if self.signature_stream:
                self.s_head = nn.Linear(d, dims["s"])
        t_max = float(len(layout["t_tr"]))
        self.T = len(layout["t_tr"]) // len(layout["t_log"])
        self.register_buffer("tf_log", time_features(torch.as_tensor(layout["t_log"]), mcfg["time_freqs"], t_max))
        self.register_buffer("tf_tr", time_features(torch.as_tensor(layout["t_tr"]), mcfg["time_freqs"], t_max))
        self.register_buffer("bnd_flag", torch.as_tensor(layout["bnd"])[:, None])

    def encode(self, rec, z, s, s_bound=None, offset=0, mask=None):
        """Encoded memory [record tokens ; trace tokens ; signature tokens].

        mask (B, L) bool, iteration 4: the content of masked tokens (descriptor, trace or signature, and in
        D the bound signature) is replaced by a learned mask vector; timestamps and token types are kept.
        """
        K, T = rec.shape[1], self.T
        tf_log = self.tf_log[offset:offset + K]
        tf_tr = self.tf_tr[offset * T:(offset + K) * T]
        bnd = self.bnd_flag[offset * T:(offset + K) * T]
        if mask is not None:
            c_log = self.rec_in(rec)
            if self.binding:
                c_log = c_log + self.s_bind(s_bound)
            contents = [c_log, self.z_in(z)] + ([self.s_in(s)] if self.signature_stream else [])
            c = torch.where(mask[..., None], self.mask_emb, torch.cat(contents, 1))
            pos = [self.time_in(tf_log) + self.type_log, self.time_in(tf_tr) + self.type_trace + bnd * self.bnd]
            if self.signature_stream:
                pos.append(self.time_in(tf_log) + self.type_sig)
            return self.encoder(c + torch.cat([p.expand(rec.shape[0], -1, -1) for p in pos], 1))
        log = self.rec_in(rec) + self.time_in(tf_log) + self.type_log
        if self.binding:
            log = log + self.s_bind(s_bound)
        tokens = [log, self.z_in(z) + self.time_in(tf_tr) + self.type_trace + bnd * self.bnd]
        if self.signature_stream:
            tokens.append(self.s_in(s) + self.time_in(tf_log) + self.type_sig)
        return self.encoder(torch.cat(tokens, 1))

    def reconstruct(self, rec, z, s, s_bound, mask, offset=0):
        """Masked pass: reconstruct every token's original input vector (records, traces, signatures)."""
        K, KT = rec.shape[1], z.shape[1]
        mem = self.encode(rec, z, s, s_bound, offset, mask)
        out = [self.rec_head(mem[:, :K]), self.z_head(mem[:, K:K + KT])]
        if self.signature_stream:
            out.append(self.s_head(mem[:, K + KT:]))
        return out

    def forward(self, rec, z, q, s, s_bound=None, offset=0, return_aux=False):
        """s: signature stream (unbound); s_bound: signatures written into the records (D only).

        Windows shorter than the full layout (curriculum) occupy episode slots offset .. offset + K - 1.
        """
        K = rec.shape[1]
        mem = self.encode(rec, z, s, s_bound, offset)
        x = self.rec_in(q) + self.type_query
        for blk in self.readout:
            x = blk(x, mem)
        logits = self.head(self.ln_out(x))
        if return_aux:   # prediction of the co-timed signature from each encoded record token
            return logits, self.aux_head(mem[:, :K])
        return logits


def n_active_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_widths(dims, mcfg, layout):
    """d_ff for F chosen to minimise |P_F - P_D|; returns widths, counts and mismatch."""
    d_ff_d = mcfg["d_ff_D"]
    p_d = n_active_params(SelfModel(dims, mcfg, layout, True, d_ff_d))
    best = None
    for d_ff in range(d_ff_d, 4 * d_ff_d):
        p_f = n_active_params(SelfModel(dims, mcfg, layout, False, d_ff))
        if best is None or abs(p_f - p_d) < abs(best[1] - p_d):
            best = (d_ff, p_f)
        if p_f > p_d:
            break
    d_ff_f, p_f = best
    return {"d_ff_D": d_ff_d, "d_ff_F": d_ff_f, "params_D": p_d, "params_F": p_f,
            "rel_mismatch": (p_f - p_d) / p_d}
