"""The dog's world and every sense of the real Go2, for training thousands of dogs at once in MJX (DogMind.md §3).

The world is a fenced park, like the park in the studio's open world (world/maps.py) but small enough for MJX:
grass with patches, a fence, three trees, the owner, a ball, a charging pad (where the dog "eats"), and an object
that is sometimes thrown at the dog.

What the dog senses (the brain is never told what any of it is):
  eye      the Go2's front camera: 120 x 90 degrees, sampled at a fly's acuity (one point every 5 degrees),
           24 x 18 points, retinotopic (neighbours stay neighbours, as in the fly's eye), two colour channels
           (a dog sees blue and yellow; a fly's photoreceptors split the spectrum in a similar way)
  lidar    the L1 LiDAR on the chin: 360 x 90 degrees, one ray every 10 degrees, 36 x 9 distances
  ears     two ears, three pitch bands (low: something flying at it; mid: the owner's voice; high: the ball)
  nose     two nostrils, two smells (the ball, the charging pad). The real Go2 has no nose; a dog's world is
           mostly smell, so the dog gets one.
  paws     the force on each paw (the Go2 EDU's foot sensors)
  skin     the force on the trunk and head (being hit, bumping into things)
  battery  charge left; it drains with effort and fills on the charging pad
  heat     the temperature of each of the 12 motors; a hot motor gets weaker, as on the real robot

Time is compressed so that a battery runs flat in minutes rather than hours, and a motor overheats after about
a minute of hard running.

  python train/dog_world.py      # builds the world and saves pictures of what the dog senses (results/dog_world/)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FEET = ["FL", "FR", "RL", "RR"]
ARENA = 6.0                     # the fence: a 12 x 12 m square
EYE_POS = (0.34, 0.0, 0.02)     # the front camera, on the face (base frame: x forward, y left, z up)
EYE_FOV = (120.0, 90.0)         # degrees, horizontal x vertical
EYE_STEP = 5.0                  # degrees between eye points: a fly's interommatidial angle
LIDAR_POS = (0.293, 0.0, -0.06)  # the L1 dome under the chin
LIDAR_STEP = 10.0
LIDAR_RANGE = 30.0              # m (no return beyond)
HAZE = 15.0                     # m: distant things fade toward the sky colour
BALL_R, THROWN_R = 0.11, 0.08   # a football; a fist-sized object
THROW_EVERY = 10.0              # s: one object thrown at the dog every ~10 s (when throwing is on)
THROW_FLIGHT = (0.5, 0.9)       # s: from release to the dog (about 4-9 m/s, a hard throw of a soft toy)
THROW_FROM = 1.7                # m: released at a hand's height, above the trees
BATTERY_J = 14400.0             # J: ~3 minutes at 80 W (compressed time)
IDLE_W = 30.0                   # W drawn just being awake
CHARGE_PER_S = 1.0 / 60.0       # a full charge in a minute on the pad
HEAT_K = 0.0064                 # degC per s per (N m)^2: 12 N m held steadily settles ~55 degC above ambient
COOL_S = 60.0                   # s: motor cooling time constant
AMBIENT, HOT, MELT = 25.0, 65.0, 90.0   # degC: weaker from HOT, at 40% strength by MELT
# how each thing looks to a dichromat (blue, yellow)
LOOK = {"grass": (0.12, 0.42), "fence": (0.55, 0.60), "tree": (0.10, 0.20), "owner": (0.45, 0.35),
        "ball": (0.05, 0.95), "thrown": (0.10, 0.30), "pad": (0.85, 0.20), "dog": (0.35, 0.40),
        "sky": (0.90, 0.60)}
TREES = 3


def add_world(spec) -> None:
    """Put the park and the robot's extra senses into a Go2 spec (called by brain_dog.build_model(world=True), which
    has already given the trunk and head their collision shapes). Collision bits: 1 = the ground (floor, paws,
    loose things); 2 = the trunk and head against the ground and the world's things."""
    import mujoco
    T = mujoco.mjtGeom
    base = spec.body("base")
    for f in FEET:                                         # paw force: a touch sensor around each paw
        spec.site(f"{f}_foot").size = [0.03, 0.03, 0.03]
        spec.add_sensor(name=f"{f}_touch", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                        objname=f"{f}_foot")
    base.add_site(name="skin", type=T.mjGEOM_BOX, size=[0.26, 0.07, 0.09], pos=[0.06, 0, 0])
    spec.add_sensor(name="skin_touch", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="skin")

    world = spec.worldbody
    for i, (n, at) in enumerate([((-1, 0, 0), (ARENA, 0)), ((1, 0, 0), (-ARENA, 0)),
                                 ((0, -1, 0), (0, ARENA)), ((0, 1, 0), (0, -ARENA))]):
        # the fence: an invisible rail (a capsule lying behind the board) stops things, the board is what the eye
        # and the LiDAR see. Not a plane: a plane is an endless half-space, and anything parked outside the park
        # (the thrown thing between throws) would sit 44 m deep inside it and blow up the physics.
        along_x = n[0] == 0
        c = ARENA + 0.37                                   # the rail's inner side lines up with the board's face
        ends = ([-ARENA - 0.4, -c * n[1], 0.4, ARENA + 0.4, -c * n[1], 0.4] if along_x else
                [-c * n[0], -ARENA - 0.4, 0.4, -c * n[0], ARENA + 0.4, 0.4])
        world.add_geom(name=f"fence_rail{i}", type=T.mjGEOM_CAPSULE, size=[0.4, 0, 0], fromto=ends,
                       rgba=[0, 0, 0, 0], contype=0, conaffinity=2)
        world.add_geom(name=f"fence{i}", type=T.mjGEOM_BOX, pos=[at[0], at[1], 0.4],
                       size=[ARENA, 0.03, 0.4] if along_x else [0.03, ARENA, 0.4], contype=0, conaffinity=0,
                       rgba=[0.8, 0.8, 0.7, 1])
    for i in range(TREES):
        b = world.add_body(name=f"tree{i}", mocap=True, pos=[3.0 + i, 3.0, 0.0])
        b.add_geom(name=f"tree{i}", type=T.mjGEOM_CAPSULE, size=[0.12, 0.7, 0], pos=[0, 0, 0.82], contype=0,
                   conaffinity=2, rgba=[0.3, 0.2, 0.1, 1])
    b = world.add_body(name="owner", mocap=True, pos=[-3.0, 0.0, 0.0])
    b.add_geom(name="owner", type=T.mjGEOM_CAPSULE, size=[0.2, 0.65, 0], pos=[0, 0, 0.85], contype=0, conaffinity=2,
               rgba=[0.6, 0.4, 0.3, 1])
    b = world.add_body(name="pad", mocap=True, pos=[2.0, -2.0, 0.0])
    b.add_geom(name="pad", type=T.mjGEOM_BOX, size=[0.35, 0.35, 0.004], pos=[0, 0, 0.004], contype=0, conaffinity=0,
               rgba=[0.2, 0.3, 0.9, 1])
    # the thrown thing is a soft toy: light, with a soft contact (a stiff 0.3 kg ball hit the dog with up to 22 kN
    # and blew up the GPU physics, which solves contacts in a single iteration for speed)
    for name, r, m, soft, at in (("ball", BALL_R, 0.43, 0.02, [1.5, 0.0, BALL_R]),
                                 ("thrown", THROWN_R, 0.2, 0.04, [50.0, 50.0, THROWN_R])):
        b = world.add_body(name=name, pos=at)
        b.add_freejoint(name=name)
        b.add_geom(name=name, type=T.mjGEOM_SPHERE, size=[r, 0, 0], mass=m, contype=3, conaffinity=3, condim=3,
                   friction=[0.8, 0.01, 0.01], solref=[soft, 1.0],
                   rgba=[1.0, 0.8, 0.1, 1] if name == "ball" else [0.7, 0.2, 0.2, 1])
    key = spec.keys[0]                                     # home: the dog as before, the ball out front, the thrown thing parked
    key.qpos = list(key.qpos) + [1.5, 0.0, BALL_R, 1, 0, 0, 0] + [50.0, 50.0, THROWN_R, 1, 0, 0, 0]


def eye_dirs():
    """Unit directions of the eye's points in the base frame, [H, W, 3]; row 0 looks up, column 0 looks left."""
    w, h = int(EYE_FOV[0] / EYE_STEP), int(EYE_FOV[1] / EYE_STEP)
    az = np.radians(EYE_FOV[0] / 2 - EYE_STEP * (np.arange(w) + 0.5))
    el = np.radians(EYE_FOV[1] / 2 - EYE_STEP * (np.arange(h) + 0.5))
    E, A = np.meshgrid(el, az, indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)


def lidar_dirs():
    """[9, 36, 3]: elevation -40..+40 degrees, all the way round."""
    az = np.radians(np.arange(0.0, 360.0, LIDAR_STEP))
    el = np.radians(np.arange(40.0, -41.0, -LIDAR_STEP))
    E, A = np.meshgrid(el, az, indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)


def geom_looks(model) -> np.ndarray:
    """[ngeom, 2]: how each geom looks (blue, yellow)."""
    out = np.tile(np.array(LOOK["dog"]), (model.ngeom, 1))
    for g in range(model.ngeom):
        name = model.geom(g).name
        kind = ("grass" if name == "floor" else "fence" if name.startswith("fence") else
                "tree" if name.startswith("tree") else name if name in LOOK else None)
        if kind:
            out[g] = LOOK[kind]
    return out


def shade(xp, dist, gid, origin, dirs, looks, geom_xpos):
    """What each ray sees, [..., 2], from where it hit (shared by the MJX eye and the CPU check). Grass has
    patches (so the ground moves past the eye as the dog walks), round things are lit from above, and distant
    things fade toward the sky."""
    hit = gid >= 0
    g = xp.where(hit, gid, 0)
    d = xp.where(hit, dist, 1e3)
    p = origin + dirs * d[..., None]
    base = looks[g]
    patches = 0.75 + 0.25 * xp.sin(3.1 * p[..., 0]) * xp.sin(2.7 * p[..., 1]) + 0.1 * xp.sin(11.0 * p[..., 0] + 7.0 * p[..., 1])
    normal = p - geom_xpos[g]
    normal = normal / (xp.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)
    lit = 0.55 + 0.45 * xp.clip(normal[..., 2], 0.0, 1.0)
    floor = (g == 0)[..., None]
    col = base * xp.where(floor, patches[..., None], lit[..., None])
    sky = xp.asarray(LOOK["sky"]) * (0.7 + 0.3 * xp.clip(dirs[..., 2:3], 0.0, 1.0))
    fade = xp.exp(-d / HAZE)[..., None]
    return xp.where(hit[..., None], col * fade + sky * (1.0 - fade), sky)


def make_world(model, throws: bool = False):
    """JAX functions for the world, per dog (vmap them): place (a new life), tick (each control step: throws,
    battery, motor heat), sense (everything the dog senses now), strength (how much each motor can still do)."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from mujoco.mjx._src import ray as mjx_ray

    mx = mjx.put_model(model, impl="jax")
    base = model.body("base").id
    group = [1, 1, 0, 1, 0, 0]                    # rays see the world (0, 1) and the dog's collision shapes (3), not meshes (2)
    eye = jp.asarray(eye_dirs())
    lidar = jp.asarray(lidar_dirs())
    looks = jp.asarray(geom_looks(model))
    ball_q = int(model.jnt_qposadr[model.joint("ball").id])
    ball_v = int(model.jnt_dofadr[model.joint("ball").id])
    thr_q = int(model.jnt_qposadr[model.joint("thrown").id])
    thr_v = int(model.jnt_dofadr[model.joint("thrown").id])
    mocap = {n: int(model.body_mocapid[model.body(n).id]) for n in [*[f"tree{i}" for i in range(TREES)], "owner", "pad"]}
    touch = [model.sensor_adr[model.sensor(f"{f}_touch").id] for f in FEET]
    skin = int(model.sensor_adr[model.sensor("skin_touch").id])
    parked = jp.array([50.0, 50.0, THROWN_R])

    def place(key, d):
        """A new life: trees, owner, pad and ball somewhere in the park, each in its own sixth of the circle around
        the dog (so nothing starts inside anything else), nothing in the air."""
        ks = jax.random.split(key, 9)
        turn = jax.random.uniform(ks[8], (), minval=0.0, maxval=2 * np.pi)
        def spot(k, sector, rmin, rmax):
            k1, k2 = jax.random.split(k)
            r = jax.random.uniform(k1, (), minval=rmin, maxval=rmax)
            a = turn + (sector + jax.random.uniform(k2, (), minval=0.15, maxval=0.85)) * np.pi / 3
            return jp.clip(jp.array([r * jp.cos(a), r * jp.sin(a)]), -ARENA + 0.5, ARENA - 0.5)
        mp = d.mocap_pos
        for i in range(TREES):
            mp = mp.at[mocap[f"tree{i}"], :2].set(spot(ks[i], 2 * i, 1.5, 5.0))
        mp = mp.at[mocap["owner"], :2].set(spot(ks[3], 5, 2.0, 5.0)).at[mocap["pad"], :2].set(spot(ks[4], 3, 1.5, 4.0))
        ball = jp.concatenate([spot(ks[5], 1, 1.0, 4.0), jp.array([BALL_R, 1.0, 0.0, 0.0, 0.0])])
        qpos = d.qpos.at[ball_q:ball_q + 7].set(ball).at[thr_q:thr_q + 7].set(jp.concatenate([parked, jp.array([1.0, 0, 0, 0])]))
        qvel = d.qvel.at[ball_v:ball_v + 6].set(0.0).at[thr_v:thr_v + 6].set(0.0)
        d = mjx.forward(mx, d.replace(mocap_pos=mp, qpos=qpos, qvel=qvel))
        state = {"battery": jax.random.uniform(ks[6], (), minval=0.5, maxval=1.0), "heat": jp.full(model.nu, AMBIENT),
                 "voice": jp.zeros(()), "flying": jp.zeros(()), "throw_now": jp.zeros((), bool), "key": ks[7]}
        return d, state

    def tick(d, state, dt, spoke=False):
        """Each control step: an object may be thrown, the battery drains or charges, the motors heat or cool."""
        k_throw, k_from, k_far, k_time, k_next = jax.random.split(state["key"], 5)
        torque = d.actuator_force
        power = IDLE_W + jp.sum(jp.abs(torque * d.qvel[6:18])) + 0.5 * jp.sum(torque ** 2) * 0.05
        on_pad = jp.all(jp.abs(d.qpos[:2] - d.mocap_pos[mocap["pad"], :2]) < 0.35)
        battery = jp.clip(state["battery"] - power * dt / BATTERY_J + on_pad * CHARGE_PER_S * dt, 0.0, 1.0)
        heat = state["heat"] + (HEAT_K * torque ** 2 - (state["heat"] - AMBIENT) / COOL_S) * dt
        # throwing: from 3-5 m away, aimed at where the dog's trunk will be, arriving in 0.4-0.8 s
        thr = d.qpos[thr_q:thr_q + 3]
        idle = jp.linalg.norm(thr[:2] - parked[:2]) < 1.0
        go = idle & ((throws & (jax.random.uniform(k_throw) < dt / THROW_EVERY)) | state["throw_now"])   # tests throw on cue
        a = jax.random.uniform(k_from, (), minval=-np.pi, maxval=np.pi)
        r = jax.random.uniform(k_far, (), minval=3.0, maxval=5.0)
        t_f = jax.random.uniform(k_time, (), minval=THROW_FLIGHT[0], maxval=THROW_FLIGHT[1])
        target = d.subtree_com[base] + d.qvel[0:3] * t_f
        start = jp.array([d.qpos[0] + r * jp.cos(a), d.qpos[1] + r * jp.sin(a), THROW_FROM])
        start = start.at[:2].set(jp.clip(start[:2], -ARENA + 0.3, ARENA - 0.3))       # thrown from inside the fence
        v0 = (target - start) / t_f - 0.5 * jp.array([0.0, 0.0, -9.81]) * t_f
        flying = jp.where(go, 0.0, state["flying"] + dt)
        back = ~idle & (flying > 3.0)                                     # done: put it away
        qpos = d.qpos.at[thr_q:thr_q + 3].set(jp.where(go, start, jp.where(back, parked, thr)))
        qvel = d.qvel.at[thr_v:thr_v + 3].set(jp.where(go, v0, jp.where(back, 0.0, d.qvel[thr_v:thr_v + 3])))
        voice = jp.where(spoke, 1.0, state["voice"] * jp.exp(-dt / 0.5))
        d = d.replace(qpos=qpos, qvel=qvel)
        return d, {"battery": battery, "heat": heat, "voice": voice, "flying": flying, "throw_now": jp.zeros((), bool),
                   "key": k_next}

    def strength(state):
        """How much of its commanded move each motor still makes: hot motors and a flat battery are weak."""
        hot = 1.0 - 0.6 * jp.clip((state["heat"] - HOT) / (MELT - HOT), 0.0, 1.0)
        return hot * jp.clip(state["battery"] / 0.1, 0.3, 1.0)

    def cast(d, origin, dirs):
        flat = dirs.reshape(-1, 3)
        dist, gid = jax.vmap(lambda v: mjx_ray.ray(mx, d, origin, v, geomgroup=group, bodyexclude=base))(flat)
        return dist.reshape(dirs.shape[:-1]), gid.reshape(dirs.shape[:-1])

    def sense(d, state, key, sight=True):
        """sight=False: everything but the eye and the LiDAR (the cheap lines; no rays cast)."""
        R, pos = d.xmat[base], d.xpos[base]
        out = {}
        if sight:
            o_eye = pos + R @ jp.array(EYE_POS)
            e_dirs = eye @ R.T
            dist, gid = cast(d, o_eye, e_dirs)
            o_lid = pos + R @ jp.array(LIDAR_POS)
            l_dist, l_gid = cast(d, o_lid, lidar @ R.T)
            out = {"eye": shade(jp, dist, gid, o_eye, e_dirs, looks, d.geom_xpos),
                   "lidar": jp.where(l_gid >= 0, jp.minimum(l_dist, LIDAR_RANGE), LIDAR_RANGE)}
        # ears: two ears on the sides of the head; three bands (low: a flying thing, mid: the owner, high: the ball)
        ball, thr = d.qpos[ball_q:ball_q + 3], d.qpos[thr_q:thr_q + 3]
        owner = d.mocap_pos[mocap["owner"]] + jp.array([0.0, 0.0, 1.5])
        sources = jp.stack([thr, owner, ball])
        loud = jp.stack([jp.sum(d.qvel[thr_v:thr_v + 3] ** 2) / 20.0, state["voice"],
                         jp.linalg.norm(d.qvel[ball_v:ball_v + 3]) / 2.0])
        ears = []
        for side in (1.0, -1.0):
            ear = pos + R @ jp.array([0.25, 0.06 * side, 0.05])
            axis = R @ jp.array([0.0, side, 0.0])
            to = sources - ear
            dist_s = jp.linalg.norm(to, axis=-1)
            facing = 0.6 + 0.4 * jp.sum(to / dist_s[:, None] * axis, -1)
            ears.append(loud * facing / (1.0 + dist_s ** 2))
        # nose: two nostrils, the ball's smell and the pad's, patchy like a real plume
        k1, k2 = jax.random.split(key)
        smells = jp.stack([ball, d.mocap_pos[mocap["pad"]]])
        nose = []
        for i, side in enumerate((1.0, -1.0)):
            nostril = pos + R @ jp.array([0.36, 0.015 * side, -0.02])
            c = jp.exp(-jp.linalg.norm(smells - nostril, axis=-1) / 1.5)
            nose.append(c * jp.exp(0.3 * jax.random.normal(jax.random.fold_in(k1, i), (2,))))
        return {**out, "ears": jp.concatenate(ears), "nose": jp.concatenate(nose),
                "paws": d.sensordata[jp.array(touch)], "skin": d.sensordata[skin:skin + 1],
                "battery": state["battery"][None], "heat": state["heat"]}

    def where(d):
        """For the critic and for rewards: the true positions of the ball, the thrown object, the owner and the pad."""
        return {"ball": d.qpos[ball_q:ball_q + 3], "ball_vel": d.qvel[ball_v:ball_v + 3],
                "thrown": d.qpos[thr_q:thr_q + 3], "thrown_vel": d.qvel[thr_v:thr_v + 3],
                "owner": d.mocap_pos[mocap["owner"]], "pad": d.mocap_pos[mocap["pad"]]}

    return {"place": place, "tick": tick, "sense": sense, "strength": strength, "where": where,
            "eye_shape": tuple(eye.shape[:2]), "lidar_shape": tuple(lidar.shape[:2])}


# ============================================================================ the CPU check: what the dog sees
def see_cpu(model, data):
    """The eye and the LiDAR computed with MuJoCo's own C ray caster (the MJX version must agree with this)."""
    import mujoco
    base = model.body("base").id
    group = np.array([1, 1, 0, 1, 0, 0], np.uint8)
    R, pos = data.xmat[base].reshape(3, 3), data.xpos[base]
    looks = geom_looks(model)

    def cast(origin, dirs):
        flat = dirs.reshape(-1, 3)
        dist, gid = np.zeros(len(flat)), np.zeros(len(flat), int)
        out = np.zeros(1, np.int32)
        for i, v in enumerate(flat):
            dist[i] = mujoco.mj_ray(model, data, origin, v, group, 1, base, out)
            gid[i] = out[0]
        return dist.reshape(dirs.shape[:-1]), gid.reshape(dirs.shape[:-1])
    o_eye = pos + R @ np.array(EYE_POS)
    e_dirs = eye_dirs() @ R.T
    dist, gid = cast(o_eye, e_dirs)
    eye = shade(np, dist, gid, o_eye, e_dirs, looks, data.geom_xpos)
    o_lid = pos + R @ np.array(LIDAR_POS)
    l_dist, l_gid = cast(o_lid, lidar_dirs() @ R.T)
    return {"eye": eye, "eye_gid": gid, "lidar": np.where(l_gid >= 0, np.minimum(l_dist, LIDAR_RANGE), LIDAR_RANGE),
            "lidar_gid": l_gid}


def _png(path: Path, rgb: np.ndarray) -> None:
    """Write an [H, W, 3] uint8 image as PNG (no imaging library needed)."""
    import struct
    import zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _grow(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize."""
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[ys][:, xs]


def _picture(model, data, out: Path) -> None:
    """Top: the scene from behind the dog, and from its head. Bottom: what its eye gets (blue and yellow shown as
    colour), and its LiDAR (near = bright; all the way round, straight ahead in the middle)."""
    import mujoco
    seen = see_cpu(model, data)
    r = mujoco.Renderer(model, 360, 480)
    base = model.body("base").id
    R, pos = data.xmat[base].reshape(3, 3), data.xpos[base]
    heading = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    cam = mujoco.MjvCamera()
    cam.type, cam.lookat, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, pos, 2.6, heading, -20
    r.update_scene(data, cam)
    behind = r.render()
    head = mujoco.MjvCamera()
    head.type, head.lookat = mujoco.mjtCamera.mjCAMERA_FREE, pos + R @ np.array([1.34, 0.0, 0.02])
    head.distance, head.azimuth, head.elevation = 1.0, heading, 0.0
    r.update_scene(data, head)
    camera = r.render()
    to_rgb = lambda e: np.stack([e[..., 1], e[..., 1], e[..., 0]], -1)   # yellow -> red+green, blue -> blue
    eye = _grow((np.clip(to_rgb(seen["eye"]), 0, 1) * 255).astype(np.uint8), 360, 480)
    lid = 1.0 - np.clip(np.log1p(seen["lidar"]) / np.log1p(LIDAR_RANGE), 0, 1)
    lid = np.roll(lid, lid.shape[1] // 2, axis=1)[:, ::-1]                # ahead in the middle, left on the left
    lid = np.stack([_grow((lid * 255).astype(np.uint8), 120, 480)] * 3, -1)
    bottom = np.concatenate([eye, np.concatenate([lid, np.full((240, 480, 3), 255, np.uint8)], 0)], 1)
    _png(out, np.concatenate([np.concatenate([behind, camera], 1), bottom], 0))


if __name__ == "__main__":
    import sys
    import mujoco
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import brain_dog
    model = brain_dog.build_model(repo / "third_party" / "mujoco_menagerie" / "unitree_go2", world=True)
    data = mujoco.MjData(model)
    data.qpos[:] = model.keyframe("home").qpos
    print(f"world: {model.ngeom} geoms, {model.nbody} bodies, {model.nmocap} mocap, nq {model.nq}, dog mass "
          f"{model.body_subtreemass[model.body('base').id]:.3f} kg, sensors {model.nsensor}")
    rng = np.random.default_rng(1)
    out = repo / "results" / "dog_world"
    out.mkdir(parents=True, exist_ok=True)
    for shot in range(3):
        for i in range(TREES):
            a, r_ = rng.uniform(-np.pi, np.pi), rng.uniform(1.5, 5.0)
            data.mocap_pos[model.body_mocapid[model.body(f"tree{i}").id], :2] = [r_ * np.cos(a), r_ * np.sin(a)]
        data.mocap_pos[model.body_mocapid[model.body("owner").id], :2] = [3.0, rng.uniform(-1.5, 1.5)]
        data.mocap_pos[model.body_mocapid[model.body("pad").id], :2] = [1.5, rng.uniform(-1.0, 1.0)]
        ball = model.jnt_qposadr[model.joint("ball").id]
        data.qpos[ball:ball + 2] = [1.2, rng.uniform(-0.6, 0.6)]
        mujoco.mj_forward(model, data)
        seen = see_cpu(model, data)
        names = lambda g: sorted({model.geom(int(x)).name or f"geom{int(x)}" for x in np.unique(g) if x >= 0})
        print(f"shot {shot}: the eye sees {names(seen['eye_gid'])}; the LiDAR {names(seen['lidar_gid'])}; "
              f"nearest LiDAR return {seen['lidar'].min():.2f} m")
        _picture(model, data, out / f"senses_{shot}.png")
    # physics: the motors hold the standing pose for 2 s; does the world stay sane?
    data.ctrl[:] = model.keyframe("home").ctrl
    for _ in range(500):
        mujoco.mj_step(model, data)
    print(f"after 2 s of physics: trunk at {data.xpos[1][2]:.3f} m, ball at {data.qpos[ball:ball + 3].round(3)}, "
          f"paw forces {[round(float(data.sensordata[model.sensor_adr[model.sensor(f + '_touch').id]]), 1) for f in FEET]} N")
