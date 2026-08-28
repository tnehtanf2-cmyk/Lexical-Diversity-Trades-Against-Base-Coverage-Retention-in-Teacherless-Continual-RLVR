"""State tracker: builds the meta-controller state s_t from cycle metrics.
Regime signals (실험설계 §2): acc, Δacc, pass@k-spread, entropy, repetition,
policy–reference KL, hybrid BWT, domain-verifiability flag, previous budget w_{t-1}.
Hybrid normalization (running EMA z-score + tanh).
"""
from __future__ import annotations
import numpy as np

STATE_KEYS = [
    "acc", "d_acc", "passk_spread", "gen_entropy", "repetition",
    "kl", "bwt", "verifiable_flag",
    "w_explore", "w_stable", "w_prec", "w_diff",
]
STATE_DIM = len(STATE_KEYS)  # 12


class StateTracker:
    def __init__(self, ema=0.1):
        self.ema = ema
        self.prev_acc = 0.0
        self._mean = {k: 0.0 for k in ("passk_spread", "kl")}
        self._var = {k: 1.0 for k in ("passk_spread", "kl")}

    def _z(self, key, x):
        d = x - self._mean[key]
        self._mean[key] = (1 - self.ema) * self._mean[key] + self.ema * x
        self._var[key] = (1 - self.ema) * self._var[key] + self.ema * (d * d)
        return float(np.tanh(d / (np.sqrt(self._var[key]) + 1e-8)))

    def build(self, m: dict, w_prev: dict, verifiable: bool) -> np.ndarray:
        acc = m.get("acc", 0.0)
        s = {
            "acc": acc,
            "d_acc": float(np.tanh((acc - self.prev_acc) * 5.0)),
            "passk_spread": self._z("passk_spread", m.get("passk_spread", 0.0)),
            "gen_entropy": m.get("gen_entropy", 0.0),
            "repetition": m.get("repetition", 0.0),
            "kl": self._z("kl", m.get("kl", 0.0)),
            "bwt": m.get("bwt", 0.0),
            "verifiable_flag": 1.0 if verifiable else 0.0,
            "w_explore": w_prev.get("explore", 0.25),
            "w_stable": w_prev.get("stable", 0.25),
            "w_prec": w_prev.get("prec", 0.25),
            "w_diff": w_prev.get("diff", 0.25),
        }
        self.prev_acc = acc
        return np.array([s[k] for k in STATE_KEYS], dtype=np.float32)
