"""A go-kart: a fifth, non-animal body the brain can be plugged into.

x is forward. Rear-wheel drive (two torque motors), front-wheel steering (two
position servos). About 150 kg, 1.9 m long.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

WHEEL_R = 0.22
WHEEL_HALF_W = 0.09
MAX_TORQUE = 70.0     # N*m per rear wheel
MAX_STEER = 0.55      # rad

RED = (0.78, 0.12, 0.10, 1)
BLACK = (0.08, 0.08, 0.09, 1)
DARK = (0.18, 0.18, 0.2, 1)
CHROME = (0.75, 0.76, 0.78, 1)
GLASS = (0.55, 0.7, 0.8, 0.35)


def quat_axis(axis: str, deg: float) -> list[float]:
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return [math.cos(h), *(math.sin(h) * c for c in v)]


def _box(body, name, pos, half, rgba, **kw):
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half, rgba=rgba, **kw)


def _wheel(parent, name: str, pos) -> mujoco.MjsBody:
    wheel = parent.add_body(name=name, pos=pos)
    wheel.add_joint(name=f"{name}_spin", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.05)
    wheel.add_geom(name=f"{name}_tyre", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[WHEEL_R, WHEEL_HALF_W, 0],
                   quat=quat_axis("x", 90), rgba=BLACK, mass=8.0, friction=[1.4, 0.02, 0.001], condim=6)
    wheel.add_geom(name=f"{name}_hub", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.11, WHEEL_HALF_W + 0.004, 0],
                   quat=quat_axis("x", 90), rgba=CHROME, contype=0, conaffinity=0, mass=0.0)
    return wheel


def car_spec() -> mujoco.MjSpec:
    s = mujoco.MjSpec()
    s.modelname = "kart"
    s.compiler.degree = False
    chassis = s.worldbody.add_body(name="chassis", pos=[0, 0, WHEEL_R + 0.06])
    chassis.add_freejoint(name="root")
    # Frame and body shell (the collision shape is the floor pan plus the shell).
    _box(chassis, "pan", [0, 0, 0.0], [0.95, 0.38, 0.04], DARK, mass=70)
    _box(chassis, "shell", [-0.05, 0, 0.13], [0.78, 0.40, 0.09], RED, mass=40)
    _box(chassis, "nose", [0.86, 0, 0.08], [0.16, 0.30, 0.06], RED, mass=6)
    _box(chassis, "bumper_front", [1.0, 0, 0.03], [0.04, 0.44, 0.05], BLACK, mass=3)
    _box(chassis, "bumper_rear", [-0.98, 0, 0.05], [0.04, 0.44, 0.06], BLACK, mass=3)
    _box(chassis, "seat", [-0.25, 0, 0.30], [0.22, 0.24, 0.08], BLACK, mass=5)
    _box(chassis, "seat_back", [-0.45, 0, 0.48], [0.05, 0.24, 0.20], BLACK, mass=3)
    _box(chassis, "windscreen", [0.38, 0, 0.36], [0.02, 0.34, 0.14], GLASS, mass=1,
         contype=0, conaffinity=0)
    for side, y in (("l", 0.36), ("r", -0.36)):
        chassis.add_geom(name=f"rollbar_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                         fromto=[-0.55, y, 0.2, -0.55, y, 0.75], size=[0.025, 0, 0], rgba=CHROME, mass=1.5)
        chassis.add_geom(name=f"headlight_{side}", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[1.02, y * 0.8, 0.12],
                         size=[0.045, 0, 0], rgba=(1.0, 0.95, 0.8, 1), contype=0, conaffinity=0, mass=0.0)
        chassis.add_geom(name=f"taillight_{side}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-1.02, y * 0.8, 0.13],
                         size=[0.01, 0.05, 0.025], rgba=(0.9, 0.05, 0.05, 1), contype=0, conaffinity=0, mass=0.0)
    chassis.add_geom(name="rollbar_top", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                     fromto=[-0.55, 0.36, 0.75, -0.55, -0.36, 0.75], size=[0.025, 0, 0], rgba=CHROME, mass=1.5)
    chassis.add_geom(name="steering_wheel", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0.12, 0, 0.44],
                     quat=quat_axis("y", -60), size=[0.13, 0.015, 0], rgba=BLACK, contype=0, conaffinity=0, mass=0.0)
    chassis.add_site(name="imu", pos=[0, 0, 0.1])

    # Front wheels steer (hinge about z), then spin; rear wheels only spin.
    for side, y in (("l", 0.52), ("r", -0.52)):
        knuckle = chassis.add_body(name=f"front_{side}", pos=[0.64, y, -0.02])
        knuckle.add_joint(name=f"steer_{side}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                          range=[-MAX_STEER, MAX_STEER], limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=2.0)
        knuckle.add_geom(name=f"knuckle_{side}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0],
                         rgba=CHROME, contype=0, conaffinity=0, mass=1.0)
        _wheel(knuckle, f"wheel_f{side}", [0, 0, 0])
    for side, y in (("l", 0.52), ("r", -0.52)):
        _wheel(chassis, f"wheel_r{side}", [-0.62, y, -0.02])

    # Front wheels hang off the steering knuckle (a grandchild of the chassis), so the
    # automatic parent-child exclusion does not cover them: without this they jam on the pan.
    for side in ("l", "r"):
        s.add_exclude(bodyname1="chassis", bodyname2=f"wheel_f{side}")
        s.add_exclude(bodyname1="chassis", bodyname2=f"front_{side}")

    for side in ("l", "r"):
        a = s.add_actuator(name=f"drive_{side}", target=f"wheel_r{side}_spin", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_motor()
        a.ctrlrange = [-MAX_TORQUE, MAX_TORQUE]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    for side in ("l", "r"):
        a = s.add_actuator(name=f"steer_{side}", target=f"steer_{side}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=600, kv=30)
        a.ctrlrange = [-MAX_STEER, MAX_STEER]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    s.add_sensor(name="accelerometer", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                 objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    s.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    return s


def forward_speed(xmat: np.ndarray, lin_vel: np.ndarray) -> float:
    return float(xmat.reshape(3, 3)[:, 0] @ lin_vel)
