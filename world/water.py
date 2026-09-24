"""Simplified water: buoyancy and drag on every body part below the surface.

MuJoCo has no water, so each control step we apply an external force to every
part inside a lake's footprint and under its surface:
  lift  = submerged * mass * g * BUOYANCY      (slightly > 1: bodies float)
  drag  = -submerged * mass * LIN_DRAG * v     (linear)
        - submerged * inertia * ANG_DRAG * w  (spin)
where `submerged` ramps from 0 to 1 over the first DEPTH_RAMP metres below the
surface. This makes bodies float, slow down and struggle in water; paddling
limbs push against the drag. It is not fluid dynamics.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .maps import Water

BUOYANCY = 1.15
LIN_DRAG = 3.0     # 1/s
ANG_DRAG = 1.0     # 1/s
DEPTH_RAMP = 0.12  # m


class WaterForces:
    def __init__(self, model: mujoco.MjModel, lakes: list[Water], bodies: np.ndarray):
        self.m = model
        self.lakes = lakes
        self.bodies = bodies                          # body ids that can get wet
        self.mass = model.body_mass[bodies]
        self.inertia = model.body_inertia[bodies].mean(axis=1)   # kg*m^2, principal average
        self.g = -model.opt.gravity[2]
        self.root = model.body_rootid[bodies]
        self.wet = np.zeros(len(bodies), bool)

    def apply(self, d: mujoco.MjData) -> None:
        b = self.bodies
        d.xfrc_applied[b] = 0.0
        if not self.lakes or len(b) == 0:
            return
        p = d.xipos[b]                                 # centres of mass
        sub = np.zeros(len(b))
        for lake in self.lakes:
            inside = lake.contains(p[:, 0], p[:, 1])
            depth = lake.level - p[:, 2]
            sub = np.maximum(sub, np.where(inside, np.clip(depth / DEPTH_RAMP, 0.0, 1.0), 0.0))
        self.wet = sub > 0
        if not self.wet.any():
            return
        # Velocity of each part's centre of mass from the subtree-com spatial velocity.
        w = d.cvel[b, :3]
        v = d.cvel[b, 3:] + np.cross(w, p - d.subtree_com[self.root])
        m = (self.mass * sub)[:, None]
        force = -m * LIN_DRAG * v
        force[:, 2] += (self.mass * sub) * self.g * BUOYANCY
        d.xfrc_applied[b, :3] = force
        d.xfrc_applied[b, 3:] = -(self.inertia * sub)[:, None] * ANG_DRAG * w
