"""Iteration 4: build the configuration of each condition from config_iter4.yaml.

Every condition starts from the frozen iteration-3 configuration (config_iter4.yaml names it and
records its SHA-256) and applies only the overrides listed for it: K = 32 windows and/or the
generic masked-token objective. R applies none.
"""
import copy
import hashlib
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_iter4():
    return yaml.safe_load((HERE / "config_iter4.yaml").read_text(encoding="utf-8"))


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def condition_cfg(c4, name, steps=None):
    """Full run configuration of one condition (base = iteration 3, plus the condition's overrides)."""
    base_path = HERE / c4["base_config"]
    assert sha256(base_path) == c4["base_config_sha256"], "iteration-3 base configuration changed"
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cond = c4["conditions"][name]
    if cond["k32"]:
        _merge(cfg, c4["k32_overrides"])
    if cond["masked"]:
        _merge(cfg, c4["masked_overrides"])
    cfg["seeds"] = copy.deepcopy(c4["seeds"])
    cfg["training"]["steps"] = steps if steps is not None else cond["steps"]
    cfg["meta"]["iteration"] = 4
    return cfg


def aux_key(c4, name):
    """Hyperparameter key of the auxiliary weight for this condition's objective."""
    return "aux_weight" if c4["conditions"][name]["masked"] else "aux_cotimed_weight"


def hp_for(c4, name, lr, aux):
    return {"lr": lr, "weight_decay": c4["tuning"]["weight_decay"], aux_key(c4, name): aux}
