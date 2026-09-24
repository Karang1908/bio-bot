"""The running world: physics thread, body controllers, water, and studio commands."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from world.controllers import MODES, BodyController, actuator_kind
from world.free_world import World
from world.water import WaterForces

CONTROL_EVERY = 2          # physics steps per control update (1 ms steps -> 500 Hz control)
MANUAL_DRIVE_TIMEOUT = 0.6  # s: the car stops if the studio stops sending throttle


@dataclass
class Goal:
    text: str
    verb: str | None
    target: str | None
    bodies: list[str]
    status: str
    executed_by: str | None = None
    point: list[float] | None = None


def _movable_bodies(m: mujoco.MjModel) -> np.ndarray:
    """Every body that belongs to a tree with degrees of freedom (bodies and props)."""
    roots = m.body_rootid
    return np.array([b for b in range(1, m.nbody) if m.body_dofnum[roots[b]] > 0 or m.body_dofnum[b] > 0])


class Simulation:
    def __init__(self, world: World):
        self.world = world
        self.m = world.model
        self.d = mujoco.MjData(self.m)
        self.lock = threading.RLock()
        self.controllers = {k: BodyController(world, k, seed=i) for i, k in enumerate(world.bodies)}
        self.water = WaterForces(self.m, world.map.water if world.map else [], _movable_bodies(self.m))
        self.speed = 1.0
        self.paused = False
        self.rtf = 0.0
        self.goal: Goal | None = None
        self.log: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._drive_at: dict[str, float] = {}
        self.apple_spots = world.map.apple_spots if world.map else {}
        self.apple_spot = next(iter(self.apple_spots), "")
        self.reset()

    # -- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="physics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        dt = self.m.opt.timestep
        wall0, sim0 = time.perf_counter(), self.d.time
        rtf_wall, rtf_sim = wall0, self.d.time
        while not self._stop.is_set():
            if self.paused:
                self.rtf = 0.0  # report "not running", not the last speed before the pause
                time.sleep(0.02)
                wall0, sim0 = time.perf_counter(), self.d.time
                rtf_wall, rtf_sim = wall0, sim0
                continue
            target = sim0 + (time.perf_counter() - wall0) * self.speed
            budget_end = time.perf_counter() + 0.02
            with self.lock:
                while self.d.time < target and time.perf_counter() < budget_end:
                    self._step(dt)
            if self.d.time < target - 0.25:            # can't keep up: drop the backlog
                wall0, sim0 = time.perf_counter(), self.d.time
            now = time.perf_counter()
            if now - rtf_wall > 0.5:
                self.rtf = (self.d.time - rtf_sim) / (now - rtf_wall)
                rtf_wall, rtf_sim = now, self.d.time
            time.sleep(0.001)

    def _step(self, dt: float) -> None:
        if self._nstep % CONTROL_EVERY == 0:
            now = time.perf_counter()
            for k, c in self.controllers.items():
                if c.drive is not None and c.mode == "manual" and now - self._drive_at.get(k, 0) > MANUAL_DRIVE_TIMEOUT:
                    c.drive.throttle = c.drive.steer = 0.0
                self.d.ctrl[c.h.actuators] = c.compute(self.d, dt * CONTROL_EVERY)
            self.water.apply(self.d)
        mujoco.mj_step(self.m, self.d)
        self._nstep += 1
        if not np.isfinite(self.d.qpos).all():
            self._event("warn", "Physics became unstable; world reset.")
            self.reset(keep_modes=True)

    # -- commands ---------------------------------------------------------------------
    def reset(self, keep_modes: bool = False) -> None:
        with self.lock:
            modes = {k: c.mode for k, c in self.controllers.items()}
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[:] = self.world.home_qpos
            self._nstep = 0
            for k, c in self.controllers.items():
                c.set_mode(modes[k] if keep_modes else "hold")
                c.strength[:] = 1.0
                if c.hover is not None:
                    c.hover.setpoint = np.array([c.h.home_qpos[0], c.h.home_qpos[1], 1.2])
                if c.drive is not None:
                    c.drive.target = None
            if self.apple_spot:
                self._place_apple(self.apple_spot)
            mujoco.mj_forward(self.m, self.d)
            self.goal = None

    def _apple_dofs(self) -> tuple[int, int]:
        jid = self.m.body_jntadr[self.world.objects["apple"]]
        return int(self.m.jnt_qposadr[jid]), int(self.m.jnt_dofadr[jid])

    def _place_apple(self, spot: str) -> None:
        q, v = self._apple_dofs()
        self.d.qpos[q:q + 3] = self.apple_spots[spot]
        self.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        self.d.qvel[v:v + 6] = 0

    def place_apple(self, spot: str) -> None:
        if spot not in self.apple_spots:
            raise ValueError(f"unknown spot {spot!r}; choose from {list(self.apple_spots)}")
        with self.lock:
            self.apple_spot = spot
            self._place_apple(spot)
        self._event("world", f"Apple placed {spot}.")

    def _ctrl(self, key: str) -> BodyController:
        if key not in self.controllers:
            raise KeyError(key)
        return self.controllers[key]

    def set_mode(self, key: str, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self.lock:
            self._ctrl(key).set_mode(mode)
        self._event("body", f"{self.world.bodies[key].spec.label}: {mode}.")

    def set_strength(self, key: str, value: float) -> None:
        with self.lock:
            self._ctrl(key).strength[:] = float(np.clip(value, 0.0, 1.0))
        self._event("damage", f"{self.world.bodies[key].spec.label}: actuator strength {value:.0%}.")

    def set_manual(self, key: str, index: int, value: float) -> None:
        c = self._ctrl(key)
        if not 0 <= index < len(c.manual):
            raise ValueError(f"actuator index out of range 0..{len(c.manual) - 1}")
        with self.lock:
            if c.mode != "manual":
                c.set_mode("manual")
            c.set_manual(index, value)

    def set_drive(self, key: str, throttle: float, steer: float) -> None:
        c = self._ctrl(key)
        if c.drive is None:
            raise ValueError(f"{key} isn't a vehicle")
        with self.lock:
            if c.mode != "manual":
                c.set_mode("manual")
            c.drive.throttle = float(np.clip(throttle, -1, 1))
            c.drive.steer = float(np.clip(steer, -1, 1))
            self._drive_at[key] = time.perf_counter()

    def nudge(self, key: str, dx: float, dy: float, dz: float) -> None:
        """Move an aerial body's hover target (manual flying)."""
        c = self._ctrl(key)
        if c.hover is None:
            raise ValueError(f"{key} can't fly")
        with self.lock:
            sp = c.hover.setpoint + np.array([dx, dy, dz])
            sp[2] = float(np.clip(sp[2], 0.3, 12.0))
            c.hover.setpoint = sp

    def reset_body(self, key: str) -> None:
        h = self.world.bodies[key]
        with self.lock:
            self.d.qpos[h.qpos_adr] = h.home_qpos
            self.d.qvel[h.dof_adr] = 0
            c = self._ctrl(key)
            c.noise[:] = 0
            if c.hover is not None:
                c.hover.setpoint = np.array([h.home_qpos[0], h.home_qpos[1], 1.2])
            if c.drive is not None:
                c.drive.target = None
            mujoco.mj_forward(self.m, self.d)
        self._event("body", f"{h.spec.label}: reset to spawn.")

    def go_to(self, key: str, point) -> None:
        """Built-in teachers only: the drone flies above a point, the car drives to it."""
        c = self._ctrl(key)
        with self.lock:
            c.set_mode("hold")
            if c.hover is not None:
                c.hover.setpoint = np.array([point[0], point[1], max(point[2] + 0.8, 0.5)])
            elif c.drive is not None:
                c.drive.target = np.array(point[:2], float)
            else:
                raise ValueError(f"{key} has no built-in way to travel yet")

    def object_position(self, name: str) -> np.ndarray:
        with self.lock:
            return self.d.xpos[self.world.objects[name]].copy()

    def _event(self, kind: str, text: str) -> None:
        self.log.append({"t": round(float(self.d.time), 2), "kind": kind, "text": text})
        del self.log[:-200]

    # -- read -------------------------------------------------------------------------
    def poses(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        with self.lock:
            return float(self.d.time), float(self.rtf), self.d.xpos.copy(), self.d.xquat.copy()

    def status(self) -> dict:
        with self.lock:
            wet = set(self.water.bodies[self.water.wet].tolist())
            bodies = {}
            for k, c in self.controllers.items():
                h = c.h
                R = self.d.xmat[h.root_body].reshape(3, 3)
                v = self.d.cvel[h.root_body][3:]
                bodies[k] = {
                    "mode": c.mode,
                    "connected": c.mode != "off",
                    "strength": float(c.strength.mean()),
                    "position": [round(float(x), 3) for x in self.d.xpos[h.root_body]],
                    "upright": round(float(R[2, 2]), 3),
                    "speed": round(float(np.linalg.norm(v)), 3),
                    "wet": bool(wet & set(h.body_ids.tolist())),
                    "setpoint": [round(float(x), 2) for x in c.hover.setpoint] if c.hover is not None else None,
                    "target": [round(float(x), 2) for x in c.drive.target] if c.drive is not None and c.drive.target is not None else None,
                }
            return {
                "time": round(float(self.d.time), 3),
                "rtf": round(float(self.rtf), 2),
                "speed": self.speed,
                "paused": self.paused,
                "bodies": bodies,
                "apple": {"spot": self.apple_spot,
                          "position": [round(float(x), 3) for x in self.d.xpos[self.world.objects["apple"]]]},
                "goal": self.goal.__dict__ if self.goal else None,
                "log": self.log[-40:],
            }

    # -- inspector --------------------------------------------------------------------
    def inspect(self, key: str) -> dict:
        """Static description of one body: its parts, joints, actuators and sensors."""
        c = self._ctrl(key)
        m, h = self.m, c.h
        pre = h.spec.prefix

        def nm(obj, i):
            return (mujoco.mj_id2name(m, obj, i) or "").removeprefix(pre)

        joints = []
        for j in h.joints:
            t = int(m.jnt_type[j])
            if t == mujoco.mjtJoint.mjJNT_FREE:
                continue
            joints.append({"name": nm(mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint {j}",
                           "type": {2: "slide", 3: "hinge", 1: "ball"}.get(t, "?"),
                           "range": [round(float(x), 3) for x in m.jnt_range[j]] if m.jnt_limited[j] else None,
                           "part": int(m.jnt_bodyid[j]), "qadr": int(m.jnt_qposadr[j])})
        actuators = []
        for i, a in enumerate(h.actuators):
            lo, hi = c.manual_range(i)
            kind = actuator_kind(m, a)
            j = int(c.jids[i])
            part = int(m.jnt_bodyid[j]) if j >= 0 else int(h.root_body)
            actuators.append({"index": i, "name": nm(mujoco.mjtObj.mjOBJ_ACTUATOR, a), "kind": kind,
                              "joint": nm(mujoco.mjtObj.mjOBJ_JOINT, j) if j >= 0 else None,
                              "part": part, "range": [round(lo, 3), round(hi, 3)],
                              "unit": "rad" if (c.position[i] or c.motor_joint[i]) and j >= 0 else
                                      {"thrust": "N", "grip": "0–1", "torque": "N·m"}.get(kind, "")})
        sensors = [{"name": nm(mujoco.mjtObj.mjOBJ_SENSOR, s),
                    "type": mujoco.mjtSensor(int(m.sensor_type[s])).name.removeprefix("mjSENS_").lower(),
                    "dim": int(m.sensor_dim[s]), "adr": int(m.sensor_adr[s])} for s in h.sensors]
        return {
            "key": key, "label": h.spec.label, "kind": h.spec.kind,
            "mass": round(float(m.body_subtreemass[h.root_body]), 3),
            "parts": [{"id": int(b), "name": nm(mujoco.mjtObj.mjOBJ_BODY, b)} for b in h.body_ids],
            "joints": joints, "actuators": actuators, "sensors": sensors,
            "dof": int(len(h.dof_adr)),
        }

    def live(self, key: str) -> dict:
        c = self._ctrl(key)
        h = c.h
        with self.lock:
            joints = [float(self.d.qpos[self.m.jnt_qposadr[j]]) for j in h.joints
                      if self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
            sens = [float(x) for s in h.sensors
                    for x in self.d.sensordata[self.m.sensor_adr[s]:self.m.sensor_adr[s] + self.m.sensor_dim[s]]]
            out = {
                "mode": c.mode,
                "joints": [round(x, 4) for x in joints],
                "ctrl": [round(float(x), 4) for x in self.d.ctrl[h.actuators]],
                "force": [round(float(x), 3) for x in self.d.actuator_force[h.actuators]],
                "manual": [round(float(x), 4) for x in c.manual],
                "sensors": [round(x, 4) for x in sens],
            }
            if c.drive is not None:
                R = self.d.xmat[h.root_body].reshape(3, 3)
                out["drive"] = {"throttle": c.drive.throttle, "steer": c.drive.steer,
                                "speed": round(float(R[:, 0] @ self.d.qvel[h.dof_adr[:3]]), 2)}
            if c.hover is not None:
                out["setpoint"] = [round(float(x), 2) for x in c.hover.setpoint]
            return out
