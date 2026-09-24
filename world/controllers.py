"""Built-in, non-learned controllers for each body.

These are the "free teachers" and placeholders the studio uses until a trained
brain is connected:
  off    - body disconnected from any brain: actuators produce no force (limp)
  hold   - stand in the home pose; the drone hovers; the car brakes (or drives to a target)
  babble - motor babbling: smooth random commands, the first step of self-discovery
  manual - a person drives it from the studio: per-actuator targets, the drone's
           hover target, or the car's throttle and steering
"""
from __future__ import annotations

import numpy as np
import mujoco

from .free_world import BodyHandle, World
from .vehicles import MAX_STEER, MAX_TORQUE

MODES = ("off", "hold", "babble", "manual")

# PD gains for torque-motor joints (Go2), in N*m/rad and N*m*s/rad.
KP_MOTOR = 40.0
KD_MOTOR = 1.2


def _is_position(model, a):
    return model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and model.actuator_biasprm[a, 1] != 0


def actuator_kind(model, a) -> str:
    """Plain description of what one actuator does (for the inspector)."""
    trn = model.actuator_trntype[a]
    if trn == mujoco.mjtTrn.mjTRN_BODY:
        return "grip"
    if trn == mujoco.mjtTrn.mjTRN_SITE:
        return "thrust"
    if _is_position(model, a):
        return "position"
    return "torque"


class BodyController:
    def __init__(self, world: World, key: str, seed: int = 0):
        m = world.model
        self.m = m
        self.h: BodyHandle = world.bodies[key]
        self.key = key
        self.kind = self.h.spec.kind
        self.mode = "hold"
        self.rng = np.random.default_rng(seed)
        acts = self.h.actuators
        self.lo = m.actuator_ctrlrange[acts, 0].copy()
        self.hi = m.actuator_ctrlrange[acts, 1].copy()
        self.limited = m.actuator_ctrllimited[acts].astype(bool)
        self.position = np.array([_is_position(m, a) for a in acts], dtype=bool)
        joint_trn = m.actuator_trntype[acts] == mujoco.mjtTrn.mjTRN_JOINT
        self.motor_joint = joint_trn & ~self.position
        # trnid is a joint id only for joint transmissions (sites/bodies/tendons otherwise),
        # so mask before indexing the joint tables.
        safe = np.where(joint_trn, m.actuator_trnid[acts, 0], 0)
        self.jids = np.where(joint_trn, safe, -1)
        self.qadr = np.where(joint_trn, m.jnt_qposadr[safe], 0)
        self.dadr = np.where(joint_trn, m.jnt_dofadr[safe], 0)
        home = world.home_qpos
        # Position targets: keyframe ctrl if the model ships one, else the home joint angle.
        self.hold_target = np.where(self.position, self.h.home_ctrl, 0.0)
        if not np.any(self.h.home_ctrl):
            self.hold_target = np.where(self.position & joint_trn, home[self.qadr], 0.0)
        self.joint_target = home[self.qadr].copy()     # for PD on torque motors
        # Manual targets live in each actuator's own terms: position servos and PD joints
        # take an angle, everything else takes its raw control value.
        self.manual = np.where(self.motor_joint, self.joint_target, np.where(self.position, self.hold_target, 0.0))
        self.saved_gain = m.actuator_gainprm[acts].copy()
        self.saved_bias = m.actuator_biasprm[acts].copy()
        self.enabled = True
        self.strength = np.ones(len(acts))                # damage: per-actuator multiplier
        self.noise = np.zeros(len(acts))
        self.babble_amp = 0.35
        self.hover = Hover(world, self.h) if self.kind == "aerial" else None
        self.drive = Drive(world, self.h) if self.kind == "vehicle" else None

    # -- connection -------------------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        """Disconnect = actuators produce no force at all (gains and biases zeroed)."""
        acts = self.h.actuators
        if enabled:
            self.m.actuator_gainprm[acts] = self.saved_gain
            self.m.actuator_biasprm[acts] = self.saved_bias
        else:
            self.m.actuator_gainprm[acts] = 0
            self.m.actuator_biasprm[acts] = 0
        self.enabled = enabled

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.set_enabled(mode != "off")
        self.noise[:] = 0
        if self.drive is not None and mode != "hold":
            self.drive.target = None

    def manual_range(self, i: int) -> tuple[float, float]:
        """Limits for manual target i (joint range for PD joints, control range otherwise)."""
        if self.motor_joint[i]:
            j = self.jids[i]
            r = self.m.jnt_range[j]
            return (float(r[0]), float(r[1])) if self.m.jnt_limited[j] else (-3.14, 3.14)
        return float(self.lo[i]), float(self.hi[i])

    def set_manual(self, i: int, value: float) -> None:
        lo, hi = self.manual_range(i)
        self.manual[i] = float(np.clip(value, lo, hi))

    # -- control ----------------------------------------------------------------------
    def _babble(self, dt: float) -> np.ndarray:
        # Ornstein-Uhlenbeck noise: smooth random motion with ~0.5 s correlation time.
        theta, sigma = 2.0, 1.0
        self.noise += -theta * self.noise * dt + sigma * np.sqrt(dt) * self.rng.standard_normal(len(self.noise))
        return self.noise

    def compute(self, d: mujoco.MjData, dt: float) -> np.ndarray:
        n = len(self.h.actuators)
        if self.mode == "off":
            return np.zeros(n)
        wiggle = self._babble(dt) * self.babble_amp if self.mode == "babble" else np.zeros(n)
        if self.drive is not None:
            u = self.drive.compute(d, self.mode, self._babble(dt) if self.mode == "babble" else None)
        elif self.hover is not None:
            u = self.hover.compute(d, wiggle[:4] * 2.0 if self.mode == "babble" else None)
        elif self.mode == "manual":
            u = self.manual.copy()
            mj = self.motor_joint
            if mj.any():
                u[mj] = KP_MOTOR * (self.manual[mj] - d.qpos[self.qadr[mj]]) - KD_MOTOR * d.qvel[self.dadr[mj]]
        else:
            half = np.where(self.limited, (self.hi - self.lo) / 2, 1.0)
            u = self.hold_target + wiggle * half
            if self.motor_joint.any():
                mj = self.motor_joint
                target = self.joint_target[mj] + wiggle[mj] * 0.6
                q = d.qpos[self.qadr[mj]]
                qd = d.qvel[self.dadr[mj]]
                u[mj] = KP_MOTOR * (target - q) - KD_MOTOR * qd
        u = u * self.strength
        return np.where(self.limited, np.clip(u, self.lo, self.hi), u)


class Hover:
    """Geometric position + attitude controller for a quadrotor (hold = hover)."""

    def __init__(self, world: World, h: BodyHandle):
        m = world.model
        self.m, self.h = m, h
        self.body = h.root_body
        self.mass = m.body_subtreemass[self.body]
        self.J = m.body_inertia[self.body].copy()
        self.lo = m.actuator_ctrlrange[h.actuators, 0]
        self.hi = m.actuator_ctrlrange[h.actuators, 1]
        self.dof = h.dof_adr[:6]
        self.setpoint = np.array([h.home_qpos[0], h.home_qpos[1], 1.2])
        qw, qx, qy, qz = h.home_qpos[3:7]
        self.yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))  # hold spawn heading
        self.mixer_inv = np.linalg.pinv(self._mixer(world))

    def _mixer(self, world: World) -> np.ndarray:
        """Map actuator commands -> [thrust along body z, torque x, y, z] (body frame)."""
        m = self.m
        d = mujoco.MjData(m)
        d.qpos[:] = world.home_qpos
        cols = []
        for a in self.h.actuators:
            d.ctrl[:] = 0
            d.ctrl[a] = 1.0
            mujoco.mj_forward(m, d)
            R = d.xmat[self.body].reshape(3, 3)
            f_world = d.qfrc_actuator[self.dof[:3]]
            tau_body = d.qfrc_actuator[self.dof[3:]]
            cols.append([(R.T @ f_world)[2], *tau_body])
        return np.array(cols).T

    def compute(self, d: mujoco.MjData, torque_noise=None) -> np.ndarray:
        p = d.xpos[self.body]
        R = d.xmat[self.body].reshape(3, 3)
        v = d.qvel[self.dof[:3]]
        w = d.qvel[self.dof[3:]]
        g = -self.m.opt.gravity[2]
        a_des = 3.0 * (self.setpoint - p) - 2.8 * v + np.array([0, 0, g])
        a_des[:2] = np.clip(a_des[:2], -4, 4)
        thrust = self.mass * a_des @ R[:, 2]
        zd = a_des / np.linalg.norm(a_des)
        xc = np.array([np.cos(self.yaw), np.sin(self.yaw), 0])
        yd = np.cross(zd, xc); yd /= np.linalg.norm(yd)
        Rd = np.column_stack([np.cross(yd, zd), yd, zd])
        e_mat = Rd.T @ R - R.T @ Rd
        e_r = 0.5 * np.array([e_mat[2, 1], e_mat[0, 2], e_mat[1, 0]])
        tau = self.J * (-60.0 * e_r - 12.0 * w)
        if torque_noise is not None:
            tau = tau + self.J * torque_noise[:3] * 20.0
        u = self.mixer_inv @ np.array([thrust, *tau])
        return np.clip(u, self.lo, self.hi)


class Drive:
    """Go-kart: brake when holding, pure-pursuit to a target, or a person's throttle and steering."""

    def __init__(self, world: World, h: BodyHandle):
        m = world.model
        self.m, self.h = m, h
        self.body = h.root_body
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in h.actuators]
        self.i_drive = [i for i, n in enumerate(names) if "drive_" in n]
        self.i_steer = [i for i, n in enumerate(names) if "steer_" in n]
        wheel_joints = m.actuator_trnid[h.actuators[self.i_drive], 0]
        self.wheel_dofs = m.jnt_dofadr[wheel_joints]
        self.lin = h.dof_adr[:3]
        self.target: np.ndarray | None = None
        self.throttle = 0.0     # manual, -1..1
        self.steer = 0.0        # manual, -1..1

    def compute(self, d: mujoco.MjData, mode: str, noise=None) -> np.ndarray:
        u = np.zeros(len(self.h.actuators))
        R = d.xmat[self.body].reshape(3, 3)
        speed = float(R[:, 0] @ d.qvel[self.lin])
        spin = d.qvel[self.wheel_dofs]
        if mode == "babble" and noise is not None:
            torque, steer = 0.6 * noise[0] * MAX_TORQUE, 0.9 * noise[1] * MAX_STEER
        elif mode == "manual":
            torque = self.throttle * MAX_TORQUE if abs(self.throttle) > 0.02 else None
            steer = self.steer * MAX_STEER
        elif self.target is not None:
            to = self.target[:2] - d.xpos[self.body][:2]
            dist = float(np.linalg.norm(to))
            if dist < 1.5:
                self.target = None
                torque, steer = None, 0.0
            else:
                heading = np.arctan2(R[1, 0], R[0, 0])
                err = (np.arctan2(to[1], to[0]) - heading + np.pi) % (2 * np.pi) - np.pi
                steer = float(np.clip(1.6 * err, -MAX_STEER, MAX_STEER))
                v_des = min(5.0, 0.8 * dist) * max(0.25, np.cos(err))
                torque = float(np.clip(40.0 * (v_des - speed), -MAX_TORQUE, MAX_TORQUE))
        else:
            torque, steer = None, 0.0
        u[self.i_drive] = -15.0 * spin if torque is None else torque   # None = brake
        u[self.i_steer] = steer
        return u
