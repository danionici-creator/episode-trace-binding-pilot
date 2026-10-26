"""Outcomes (manuscript Sec. 4.6): multiclass Brier skill score, log loss, accuracy, ECE."""
import numpy as np


def brier_items(p, y, n_classes=3):
    """Per-item multiclass Brier score sum_c (p_c - y_c)^2."""
    onehot = np.eye(n_classes)[y]
    return ((p - onehot) ** 2).sum(-1)


def ece(p, y, n_bins=15):
    """Top-label expected calibration error with equal-width bins."""
    conf, pred = p.max(-1), p.argmax(-1)
    correct = (pred == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def score(p, y, base_rates, n_bins=15):
    """All outcomes; the BSS reference predicts the training base rates for every item."""
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(-1, keepdims=True)
    b = brier_items(p, y)
    b_ref = brier_items(np.broadcast_to(base_rates, p.shape), y)
    return {
        "bss": float(1.0 - b.mean() / b_ref.mean()),
        "brier": float(b.mean()),
        "brier_ref": float(b_ref.mean()),
        "log_loss": float(-np.log(p[np.arange(len(y)), y]).mean()),
        "accuracy": float((p.argmax(-1) == y).mean()),
        "ece": ece(p, y, n_bins),
        "n": int(len(y)),
    }
