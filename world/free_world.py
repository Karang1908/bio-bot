"""Build a world: one map (sandbox stage or open world) plus any lineup of bodies.

Everything is a single MuJoCo model, so the bodies share one physics simulation.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .bodies import BODIES, BY_KEY, BodySpec
from .maps import MAPS, MapInfo
from .scale import cgs_to_si_factors, scale_spec
from .vehicles import car_spec

TIMESTEP = 0.001  # the 100x fly needs ~1 ms (its 0.1 ms step scaled by sqrt(100))
SANDBOX_SPACING = 3.0


@dataclass
class BodyHandle:
    """Where one body lives inside the compiled world model."""
    spec: BodySpec
    root_body: int
    qpos_adr: np.ndarray       # all qpos indices owned by this body (root first)
    dof_adr: np.ndarray        # all dof indices owned by this body
    actuators: np.ndarray      # actuator ids
    body_ids: np.ndarray       # all body ids of this body
    home_qpos: np.ndarray      # qpos values at the qpos_adr indices
    home_ctrl: np.ndarray      # ctrl values for `actuators`
    sensors: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    joints: np.ndarray = field(default_factory=lambda: np.zeros(0, int))


@dataclass
class World:
    model: mujoco.MjModel
    bodies: dict[str, BodyHandle]
    objects: dict[str, int]    # named props (e.g. "apple") -> body id
    home_qpos: np.ndarray
    map: MapInfo | None = None
    lineup: list[str] = field(default_factory=list)


def _quat_yaw(deg: float) -> np.ndarray:
    h = math.radians(deg) / 2
    return np.array([math.cos(h), 0, 0, math.sin(h)])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _load_body_spec(b: BodySpec) -> mujoco.MjSpec:
    if b.builder == "car":
        return car_spec()
    child = mujoco.MjSpec.from_file(str(b.path))
    # Absolute mesh/texture paths so assets resolve after attaching.
    meshdir = b.path.parent / (child.meshdir or "")
    texdir = b.path.parent / (child.texturedir or child.meshdir or "")
    for m in child.meshes:
        if m.file:
            m.file = str(meshdir / m.file)
    for tx in child.textures:
        if tx.file:
            tx.file = str(texdir / tx.file)
    if b.magnification:
        scale_spec(child, *cgs_to_si_factors(b.magnification))
    return child


def _spawns(info: MapInfo, lineup: list[str]) -> dict[str, tuple[float, float, float]]:
    if info.spawns:
        return {k: info.spawns[k] for k in lineup}
    # Sandbox: a row facing the camera (bodies look toward -y).
    n = len(lineup)
    return {k: ((i - (n - 1) / 2) * SANDBOX_SPACING, 0.0, -90.0) for i, k in enumerate(lineup)}


def build(map_key: str = "open", bodies: list[str] | None = None, timestep: float = TIMESTEP) -> World:
    lineup = [b.key for b in BODIES if bodies is None or b.key in bodies]
    unknown = set(bodies or []) - set(BY_KEY)
    if unknown:
        raise ValueError(f"unknown bodies: {sorted(unknown)}")
    if map_key not in MAPS:
        raise ValueError(f"unknown map {map_key!r}; choose from {list(MAPS)}")

    spec = mujoco.MjSpec()
    spec.modelname = f"{map_key}_world"
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    # No global air model: MuJoCo's fluid forces cost ~1 ms/step on the fly's 67 parts
    # (8x the rest of its physics) and air drag is negligible for every body here.
    # Water forces are applied separately (world/water.py).
    spec.option.density = 0.0
    spec.option.viscosity = 0.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    info = MAPS[map_key](spec)
    spawns = _spawns(info, lineup)

    child_keys: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for key in lineup:
        b = BY_KEY[key]
        child = _load_body_spec(b)
        # Take the standing keyframe out of the child; it is re-applied per body below.
        k0 = child.keys[0] if len(child.keys) else None
        child_keys[key] = (np.array(k0.qpos), np.array(k0.ctrl)) if k0 is not None else None
        for k in list(child.keys):
            child.delete(k)
        x, y, yaw = spawns[key]
        frame = spec.worldbody.add_frame(pos=[x, y, 0.0], quat=_quat_yaw(yaw))
        spec.attach(child, prefix=b.prefix, frame=frame)

    # Children's <option> values (timestep, cgs gravity, ...) are overridden by the
    # world's on purpose; MuJoCo warns about each one.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict")
        model = spec.compile()
    home = model.qpos0.copy()

    def name(obj, i):
        return mujoco.mj_id2name(model, obj, i) or ""

    handles: dict[str, BodyHandle] = {}
    for key in lineup:
        b = BY_KEY[key]
        bids = np.array([i for i in range(model.nbody)
                         if name(mujoco.mjtObj.mjOBJ_BODY, i).startswith(b.prefix)])
        # Joints may be unnamed (e.g. free roots), so assign them by owning body.
        body_set = set(bids.tolist())
        jids = [j for j in range(model.njnt) if model.jnt_bodyid[j] in body_set]
        qadr, dadr = [], []
        for j in jids:
            nq = {0: 7, 1: 4, 2: 1, 3: 1}[int(model.jnt_type[j])]
            nv = {0: 6, 1: 3, 2: 1, 3: 1}[int(model.jnt_type[j])]
            qadr += range(model.jnt_qposadr[j], model.jnt_qposadr[j] + nq)
            dadr += range(model.jnt_dofadr[j], model.jnt_dofadr[j] + nv)
        acts = np.array([a for a in range(model.nu)
                         if name(mujoco.mjtObj.mjOBJ_ACTUATOR, a).startswith(b.prefix)], dtype=int)
        sids = np.array([s for s in range(model.nsensor)
                         if name(mujoco.mjtObj.mjOBJ_SENSOR, s).startswith(b.prefix)], dtype=int)
        qadr = np.array(qadr)
        home_q = home[qadr].copy()
        home_ctrl = np.zeros(len(acts))
        x, y, yaw = spawns[key]
        k0 = child_keys[key]
        if k0 is not None:
            kq, kc = k0
            if len(kq) == len(qadr):
                home_q = kq.copy()
                home_q[:2] = [x, y]                   # keyframe root is at the origin
                home_q[3:7] = _quat_mul(_quat_yaw(yaw), kq[3:7])
            if len(kc) == len(acts):
                home_ctrl = kc.copy()
        home[qadr] = home_q
        handles[key] = BodyHandle(spec=b, root_body=int(bids[0]), qpos_adr=qadr,
                                  dof_adr=np.array(dadr), actuators=acts, body_ids=bids,
                                  home_qpos=home_q, home_ctrl=home_ctrl, sensors=sids,
                                  joints=np.array(jids, dtype=int))

    _rest_on_ground(model, home, handles)
    objects = {"apple": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple")}
    return World(model=model, bodies=handles, objects=objects, home_qpos=home, map=info, lineup=lineup)


def _rest_on_ground(model: mujoco.MjModel, home: np.ndarray, handles: dict[str, BodyHandle],
                    clearance: float = 0.003) -> None:
    """Shift each body vertically so its lowest collision geom sits just above z=0."""
    d = mujoco.MjData(model)
    d.qpos[:] = home
    mujoco.mj_kinematics(model, d)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for h in handles.values():
        body_set = set(h.body_ids.tolist())
        low = np.inf
        for g in range(model.ngeom):
            if model.geom_bodyid[g] not in body_set:
                continue
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            corners = center + signs * half
            world_z = (d.geom_xmat[g].reshape(3, 3) @ corners.T)[2] + d.geom_xpos[g][2]
            low = min(low, world_z.min())
        if np.isfinite(low):
            h.home_qpos[2] += clearance - low
            home[h.qpos_adr[2]] = h.home_qpos[2]
