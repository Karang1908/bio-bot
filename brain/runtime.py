"""Stage 0 brain: the MaleCNS wiring as a rate network, running live on the Mac GPU.

Untrained. Every neuron's activity x in [0, 1] follows its signed, synapse-weighted
inputs through the real connectome:

    input_i = sum_j sign_j * synapses_ji * x_j / total_synapses_into_i
    x_i <- (1 - a) x_i + a * tanh(relu(gain * input_i + u_i - threshold - fatigue_i))
    fatigue_i <- fatigue_i + b * (adapt * x_i - fatigue_i)

Fatigue (spike-frequency adaptation) makes a neuron tire while it stays active.
Without it the connectome's excitatory loops lock into a self-sustaining state that
never fades; with it, activity comes in waves when the senses change and decays
when they stop.

u is external drive: the bodies' senses, fed into the body-sense neurons through a
fixed random mapping (the brain is not told which channel is which). Nothing here
is learned yet, and the brain does not move the bodies; it only listens.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .graph import GROUPS, load

NEURONS_PER_CHANNEL = 24


@dataclass
class BrainParams:
    gain: float = 8.0
    threshold: float = 0.1
    rate: float = 0.5        # a: fraction of the way to the new value per step
    adapt: float = 3.0       # how strongly a neuron tires while active
    recovery: float = 0.08   # b: how fast fatigue follows activity


class Brain:
    def __init__(self, graph: dict | None = None, params: BrainParams | None = None, seed: int = 0):
        g = graph if graph is not None else load()
        self.p = params or BrainParams()
        self.n = int(len(g["body_id"]))
        self.group = g["group"]
        self.visible = g["visible"]
        pre, post, w = g["pre"], g["post"], g["weight"].astype(np.float32)
        insyn = np.bincount(post, weights=w, minlength=self.n).astype(np.float32)
        wn = g["sign"][pre].astype(np.float32) * w / np.maximum(insyn[post], 1.0)
        self._pre = mx.array(pre)
        self._post = mx.array(post)
        self._w = mx.array(wn)
        self.x = mx.zeros((self.n,), dtype=mx.float32)
        self.fatigue = mx.zeros((self.n,), dtype=mx.float32)
        mx.eval(self._pre, self._post, self._w, self.x, self.fatigue)
        self.steps = 0
        # Body-sense neurons (all senses except photoreceptors) receive the bodies' channels.
        self.sense_pool = np.flatnonzero((g["group"] == 0) & ~g["eye"])
        self.rng = np.random.default_rng(seed)
        self._channel_map: dict[str, np.ndarray] = {}
        self._group_index = [np.flatnonzero(self.group == i) for i in range(len(GROUPS))]
        self.n_edges = int(len(pre))
        self.n_synapses = int(w.sum())

    # -- sensory wiring -----------------------------------------------------------------
    def channels_for(self, key: str, count: int) -> np.ndarray:
        """Fixed random neurons for `count` channels of body `key` (stable for the session)."""
        m = self._channel_map.get(key)
        if m is None or m.shape[0] != count:
            m = self.rng.choice(self.sense_pool, size=(count, NEURONS_PER_CHANNEL))
            self._channel_map[key] = m
        return m

    # -- dynamics -------------------------------------------------------------------------
    def step(self, drive: np.ndarray | None = None) -> None:
        p = self.p
        msg = self._w * self.x[self._pre]
        inp = mx.zeros((self.n,), dtype=mx.float32).at[self._post].add(msg)
        h = p.gain * inp - p.threshold - self.fatigue
        if drive is not None:
            h = h + mx.array(drive.astype(np.float32))
        self.x = (1 - p.rate) * self.x + p.rate * mx.tanh(mx.maximum(h, 0.0))
        self.fatigue = self.fatigue + p.recovery * (p.adapt * self.x - self.fatigue)
        mx.eval(self.x, self.fatigue)
        self.steps += 1

    def reset(self) -> None:
        self.x = mx.zeros((self.n,), dtype=mx.float32)
        self.fatigue = mx.zeros((self.n,), dtype=mx.float32)

    # -- read -----------------------------------------------------------------------------
    def activity(self) -> np.ndarray:
        return np.asarray(self.x)

    def group_summary(self, act: np.ndarray | None = None) -> list[dict]:
        a = self.activity() if act is None else act
        out = []
        for (key, name, desc), idx in zip(GROUPS, self._group_index):
            ga = a[idx]
            out.append({"key": key, "name": name, "description": desc, "neurons": int(len(idx)),
                        "mean": float(ga.mean()) if len(ga) else 0.0,
                        "active": float((ga > 0.1).mean()) if len(ga) else 0.0})
        return out
