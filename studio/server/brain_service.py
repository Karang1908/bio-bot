"""Runs the Stage 0 brain beside the world: the bodies' senses in, live activity out."""
from __future__ import annotations

import threading
import time

import numpy as np

from brain.graph import CACHE, GROUPS
from brain.runtime import Brain

from .sim import Simulation

BRAIN_HZ = 20
DRIVE_GAIN = 0.4       # sense-neuron drive per unit of normalised joint speed
DRIVE_FLOOR = 0.3      # rad/s (or m/s): movement smaller than this barely registers


class BrainService:
    """Owns the brain thread. `available` is False until graph_v1.npz exists."""

    def __init__(self, sim: Simulation):
        self.sim = sim
        self.available = CACHE.exists()
        self.brain: Brain | None = None
        self.listening = False
        self.lock = threading.Lock()
        self._act = np.zeros(0, np.uint8)
        self._summary: list[dict] = []
        self._rms: dict[str, np.ndarray] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reset_requested = False
        self.step_ms = 0.0

    # -- lifecycle ------------------------------------------------------------------------
    def start(self) -> None:
        if not self.available:
            return
        from brain.graph import load
        g = load()
        self._visible = np.flatnonzero(g["visible"])
        pos = g["pos"][self._visible]
        pos = (pos - pos.mean(0)) / 1000.0  # micrometres -> millimetres, centred
        header = np.array([len(self._visible)], np.uint32).tobytes()
        self._points = header + pos.astype(np.float32).tobytes() + g["group"][self._visible].astype(np.uint8).tobytes()
        self._act = np.zeros(len(self._visible), np.uint8)
        self.listening = True
        # MLX GPU streams are per thread: the brain is built and stepped on its own thread only.
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(g, ready), name="brain", daemon=True)
        self._thread.start()
        ready.wait(timeout=60)

    def attach(self, sim: Simulation) -> None:
        """Listen to a different world (the studio rebuilt it with a new map or lineup)."""
        self._rms = {}
        self.sim = sim

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def set_listening(self, on: bool) -> None:
        self.listening = on
        if not on:
            self._reset_requested = True  # applied on the brain thread

    # -- loop -------------------------------------------------------------------------------
    def _drive(self) -> np.ndarray:
        """Joint speeds of every body -> drive on that body's randomly assigned sense neurons."""
        b = self.brain
        drive = np.zeros(b.n, np.float32)
        sim = self.sim
        with sim.lock:
            speeds = {k: sim.d.qvel[h.dof_adr].copy() for k, h in sim.world.bodies.items()}
        for key, v in speeds.items():
            rms = self._rms.get(key)
            if rms is None or rms.shape != v.shape:
                rms = np.full(v.shape, DRIVE_FLOOR, np.float64)
            rms = np.sqrt(0.98 * rms**2 + 0.02 * v**2)  # each channel judged against its own history
            self._rms[key] = rms
            level = np.clip(np.abs(v) / (rms + DRIVE_FLOOR), 0, 3) * DRIVE_GAIN
            neurons = b.channels_for(key, len(v))
            np.add.at(drive, neurons, np.repeat(level[:, None], neurons.shape[1], axis=1))
        return drive

    def _run(self, graph: dict, ready: threading.Event) -> None:
        self.brain = Brain(graph)
        self._summary = self.brain.group_summary()
        ready.set()
        period = 1.0 / BRAIN_HZ
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self._reset_requested:
                self._reset_requested = False
                with self.lock:
                    self.brain.reset()
                self._act[:] = 0
                self._summary = self.brain.group_summary(np.zeros(self.brain.n, np.float32))
            if self.listening and not self.sim.paused:
                drive = self._drive()
                with self.lock:
                    self.brain.step(drive)
                    act = self.brain.activity()
                self.step_ms = (time.perf_counter() - t0) * 1000
                self._act = (np.clip(act[self._visible], 0, 1) * 255).astype(np.uint8)
                self._summary = self.brain.group_summary(act)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    # -- read -------------------------------------------------------------------------------
    def frame(self) -> bytes:
        return self._act.tobytes()

    def points(self) -> bytes:
        """Count, then float32 xyz (centred, in mm) and uint8 group per drawable neuron."""
        return self._points

    def info(self) -> dict:
        b = self.brain
        base = {
            "available": self.available,
            "listening": self.listening,
            "groups": [{"key": k, "name": n, "description": d} for k, n, d in GROUPS],
        }
        if b is None:
            base["message"] = ("Brain not built: download the MaleCNS tables and run "
                               "`.venv/bin/python -m brain.graph` (see Requirements.md).")
            return base
        active = float(np.mean(self._act > 25)) if len(self._act) else 0.0
        return {**base, "neurons": b.n, "connections": b.n_edges, "synapses": b.n_synapses,
                "drawn": int(len(self._visible)), "steps": b.steps, "stepMs": round(self.step_ms, 1),
                "activeFraction": active, "summary": self._summary}
