"""The two places bodies can live: a plain sandbox stage and a small open world.

Open world (80 m x 80 m, origin at the house):
  home     furnished room at the centre (table, sofa, shelves, low cabinet)
  road     a 5 m wide loop around the house, with the go-kart parked on it
  lake     to the east, sloping beaches, ~1.3 m deep, simplified water physics
  hills    to the north-west, up to ~4.5 m
  stairs   steps and a ramp up to a 1.2 m platform, inside the road loop
  park     to the south: balls to push, cones, trees
  city     downtown to the north: six buildings (7-30 m) behind the sidewalk
Sidewalks line both sides of the road and street lamps stand along its inner edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .vehicles import quat_axis

WOOD = (0.55, 0.38, 0.24, 1)
DARK_WOOD = (0.32, 0.22, 0.15, 1)
FABRIC = (0.29, 0.36, 0.44, 1)
WALL = (0.86, 0.84, 0.80, 1)
METAL = (0.55, 0.57, 0.6, 1)
CRATE = (0.62, 0.5, 0.33, 1)
STONE = (0.62, 0.61, 0.58, 1)
ASPHALT = (0.2, 0.2, 0.22, 1)
BARK = (0.33, 0.24, 0.16, 1)
LEAVES = (0.23, 0.42, 0.2, 1)
CONE = (0.95, 0.42, 0.08, 1)
PAVING = (0.72, 0.71, 0.68, 1)
LAMP = (0.22, 0.23, 0.25, 1)


@dataclass(frozen=True)
class Place:
    key: str
    label: str
    x: float
    y: float
    view: float  # camera distance that frames it


@dataclass(frozen=True)
class Water:
    """An elliptical body of water: horizontal extent and surface height."""
    cx: float
    cy: float
    rx: float
    ry: float
    level: float

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((x - self.cx) / self.rx) ** 2 + ((y - self.cy) / self.ry) ** 2 <= 1.0


@dataclass
class MapInfo:
    key: str
    label: str
    places: list[Place]
    apple_spots: dict[str, tuple[float, float, float]]
    water: list[Water] = field(default_factory=list)
    spawns: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # x, y, yaw deg
    size: float = 20.0  # half extent of the playable area, metres


def _box(parent, name, pos, half, rgba, quat=None, **kw):
    args = dict(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half, rgba=rgba, **kw)
    if quat is not None:
        args["quat"] = quat
    return parent.add_geom(**args)


def _visual(**kw):
    return dict(contype=0, conaffinity=0, **kw)


def _light(spec):
    spec.worldbody.add_light(name="sun", pos=[4, -6, 20], dir=[-0.3, 0.45, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True,
                             diffuse=[0.8, 0.8, 0.75])


def _room(wb) -> None:
    room = wb.add_body(name="room")
    t = 0.06
    _box(room, "room_floor", [0, 0, 0.0005], [4, 3, 0.0005], (0.62, 0.48, 0.34, 1), **_visual())
    _box(room, "floor_rug", [0.6, 0, 0.004], [2.2, 1.5, 0.004], (0.52, 0.30, 0.26, 1))
    _box(room, "wall_n", [0, 3, 0.5], [4, t, 0.5], WALL)
    _box(room, "wall_s", [0, -3, 0.5], [4, t, 0.5], WALL)
    _box(room, "wall_e", [4, 0, 0.5], [t, 3, 0.5], WALL)
    _box(room, "wall_w1", [-4, 1.9, 0.5], [t, 1.1, 0.5], WALL)   # doorway between y=-0.8..0.8
    _box(room, "wall_w2", [-4, -1.9, 0.5], [t, 1.1, 0.5], WALL)
    table = wb.add_body(name="table", pos=[1.6, 1.2, 0])
    _box(table, "table_top", [0, 0, 0.74], [0.6, 0.38, 0.02], WOOD)
    for i, (x, y) in enumerate([(0.55, 0.33), (-0.55, 0.33), (0.55, -0.33), (-0.55, -0.33)]):
        _box(table, f"table_leg{i}", [x, y, 0.36], [0.025, 0.025, 0.36], DARK_WOOD)
    sofa = wb.add_body(name="sofa", pos=[0.8, -2.45, 0])
    _box(sofa, "sofa_seat", [0, 0, 0.22], [1.0, 0.4, 0.2], FABRIC)
    _box(sofa, "sofa_back", [0, -0.32, 0.55], [1.0, 0.1, 0.33], FABRIC)
    _box(sofa, "sofa_arm_l", [-1.0, 0, 0.35], [0.1, 0.4, 0.33], FABRIC)
    _box(sofa, "sofa_arm_r", [1.0, 0, 0.35], [0.1, 0.4, 0.33], FABRIC)
    cab = wb.add_body(name="cabinet", pos=[3.3, -0.2, 0])   # 0.14 m gap underneath: only the fly fits
    _box(cab, "cabinet_box", [0, 0, 0.44], [0.3, 0.9, 0.3], DARK_WOOD)
    for i, (x, y) in enumerate([(0.27, 0.87), (-0.27, 0.87), (0.27, -0.87), (-0.27, -0.87)]):
        _box(cab, f"cabinet_leg{i}", [x, y, 0.07], [0.02, 0.02, 0.07], METAL)
    shelf = wb.add_body(name="shelf", pos=[-1.5, 2.75, 0])
    for i, z in enumerate([0.02, 0.45, 0.9]):
        _box(shelf, f"shelf_board{i}", [0, 0, z], [0.7, 0.18, 0.015], WOOD)
    for i, x in enumerate([-0.7, 0.7]):
        _box(shelf, f"shelf_side{i}", [x, 0, 0.46], [0.015, 0.18, 0.46], WOOD)


def _apple(wb, pos) -> None:
    apple = wb.add_body(name="apple", pos=list(pos))
    apple.add_freejoint()
    apple.add_geom(name="apple_fruit", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                   size=[0.042, 0.042, 0.038], rgba=(0.78, 0.07, 0.08, 1), mass=0.15)
    apple.add_geom(name="apple_stem", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   fromto=[0, 0, 0.034, 0.004, 0, 0.052], size=[0.003, 0, 0],
                   rgba=(0.35, 0.22, 0.1, 1), mass=0.002)


# ---------------------------------------------------------------------------- sandbox
def build_sandbox(spec: mujoco.MjSpec) -> MapInfo:
    wb = spec.worldbody
    wb.add_geom(name="stage_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1],
                friction=[1.0, 0.005, 0.0001])
    _light(spec)
    _apple(wb, (2.5, 2.5, 0.05))
    return MapInfo(
        key="sandbox", label="Sandbox",
        places=[Place("stage", "Stage", 0, 0, 7)],
        apple_spots={"on the floor": (2.5, 2.5, 0.05), "in front of the bodies": (0, -2.0, 0.05)},
        size=15.0,
    )


# ---------------------------------------------------------------------------- open world
HALF = 40.0          # terrain half extent, metres
N = 161              # terrain samples per side (0.5 m spacing)
LAKE = Water(cx=31.0, cy=6.0, rx=8.5, ry=13.0, level=-0.35)
CITY = (-10.0, 23.0, 22.0, 38.0)   # x0, y0, x1, y1 of the flat downtown block
# Downtown buildings: name (facade style), centre x, y, half width x, half depth y, height.
BUILDINGS = [("bldg_brick_0", -5.0, 28.5, 3.5, 4.0, 10.0), ("bldg_glass_1", 2.5, 29.5, 3.0, 5.0, 24.0),
             ("bldg_concrete_2", 9.5, 27.0, 3.0, 3.5, 14.0), ("bldg_glass_3", 17.5, 28.0, 3.5, 4.5, 18.0),
             ("bldg_glass_4", 9.5, 34.5, 3.0, 2.5, 30.0), ("bldg_concrete_5", -5.0, 35.5, 3.5, 2.2, 7.0)]


def terrain_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ground height (m). Flat at 0 across the house, road and park."""
    h = np.zeros(np.broadcast(x, y).shape)
    # hills, north-west
    for hx, hy, amp, r in ((-29, 27, 4.5, 7.5), (-20, 33, 2.6, 5.5), (-33, 12, 1.8, 5.0), (-12, 36, 1.2, 4.0)):
        h += amp * np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * r * r))
    # gentle rolls in the south-east
    h += 0.35 * np.exp(-((x - 20) ** 2 + (y + 28) ** 2) / 60) + 0.25 * np.exp(-((x - 30) ** 2 + (y + 20) ** 2) / 40)
    # lake basin, east: smooth bowl with beaches
    d = np.sqrt(((x - LAKE.cx) / (LAKE.rx + 3)) ** 2 + ((y - LAKE.cy) / (LAKE.ry + 3)) ** 2)
    h -= 1.7 * np.clip(1.0 - d, 0, 1) ** 0.8
    # keep the built-up centre and downtown perfectly flat
    centre = np.clip((np.maximum(np.abs(x), np.abs(y)) - 23.0) / 3.0, 0, 1)
    x0, y0, x1, y1 = CITY
    outside = np.maximum(np.maximum(x0 - x, x - x1), np.maximum(y0 - y, y - y1))   # > 0 outside the block
    city = np.clip(outside / 3.0, 0, 1)
    # fade to 0 at the map edge so the terrain meets the plain around it without a cliff
    edge = np.clip((HALF - np.maximum(np.abs(x), np.abs(y))) / 6.0, 0, 1)
    edge = edge * edge * (3 - 2 * edge)
    return h * np.minimum(centre, city) * edge


def build_open(spec: mujoco.MjSpec) -> MapInfo:
    wb = spec.worldbody
    # Terrain heightfield. MuJoCo rescales hfield data to [0, 1] (min -> 0, max -> 1),
    # so the geom sits at the lowest point and the elevation size is the full range:
    # surface = lo + data * (hi - lo) reproduces terrain_height exactly (flat ground at 0).
    xs = np.linspace(-HALF, HALF, N)
    X, Y = np.meshgrid(xs, xs)                    # row index follows y, column follows x
    H = terrain_height(X, Y)
    lo, hi = float(H.min()), float(H.max())
    spec.add_hfield(name="terrain", size=[HALF, HALF, hi - lo, 1.0], nrow=N, ncol=N,
                    userdata=((H - lo) / (hi - lo)).ravel().tolist())
    wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain", pos=[0, 0, lo],
                friction=[1.0, 0.005, 0.0001])
    _light(spec)
    _room(wb)

    # Road loop around the house, with sidewalks either side (visual surfaces on flat ground;
    # a few centimetres thick so they read as layers, but bodies stand on the terrain).
    for name, pos, half in (("road_w", [-18.5, 0, 0.006], [2.5, 21, 0.006]),
                            ("road_e", [18.5, 0, 0.006], [2.5, 21, 0.006]),
                            ("road_n", [0, 18.5, 0.006], [16, 2.5, 0.006]),
                            ("road_s", [0, -18.5, 0.006], [16, 2.5, 0.006])):
        _box(wb, name, pos, half, ASPHALT, **_visual())
    walks = [("w_out", [-22, 0], [1, 23]), ("e_out", [22, 0], [1, 23]), ("n_out", [0, 22], [21, 1]),
             ("s_out", [0, -22], [21, 1]), ("w_in", [-15, 0], [1, 16]), ("e_in", [15, 0], [1, 16]),
             ("n_in", [0, 15], [14, 1]), ("s_in", [0, -15], [14, 1])]
    for name, (x, y), (hx, hy) in walks:
        _box(wb, f"sidewalk_{name}", [x, y, 0.015], [hx, hy, 0.015], PAVING, **_visual())

    # Street lamps on the inner sidewalk, heads over the road. The poles are solid. They are geoms on
    # the world body, not bodies of their own: 16 extra static bodies cost ~20% physics speed.
    for i, (x, y, yaw) in enumerate([(s * 15.0, t, 0 if s > 0 else 180) for s in (-1, 1) for t in (-11, -4, 4, 11)]
                                    + [(t, s * 15.0, 90 if s > 0 else -90) for s in (-1, 1) for t in (-11, -4, 4, 11)]):
        c, n = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        q = quat_axis("z", yaw)
        wb.add_geom(name=f"lamp{i}_pole", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, 2.1],
                    size=[0.07, 2.1, 0], rgba=LAMP)
        _box(wb, f"lamp{i}_arm", [x + 0.65 * c, y + 0.65 * n, 4.15], [0.65, 0.04, 0.04], LAMP, quat=q, **_visual())
        _box(wb, f"lamp{i}_head", [x + 1.25 * c, y + 1.25 * n, 4.08], [0.24, 0.11, 0.05], (1.0, 0.93, 0.78, 1),
             quat=q, **_visual())

    # Downtown, north of the road: solid buildings with a few rooftop units.
    for name, x, y, hx, hy, h in BUILDINGS:
        _box(wb, name, [x, y, h / 2], [hx, hy, h / 2], (1, 1, 1, 1))
        if h > 12:
            _box(wb, f"{name}_roofbox", [x - hx * 0.35, y + hy * 0.3, h + 0.6], [hx * 0.3, hy * 0.25, 0.6], STONE)

    # Stairs and a ramp up to a platform, inside the loop (north-east of the house).
    stairs = wb.add_body(name="stairs", pos=[9.0, 10.0, 0])
    for i in range(6):
        h = 0.2 * (i + 1)
        _box(stairs, f"step{i}", [-2.5 + 0.35 * i, 0, h / 2], [0.175, 1.0, h / 2], STONE)
    _box(stairs, "platform", [0.425, 0, 0.6], [1.0, 1.0, 0.6], STONE)   # flush with the top step
    tilt = math.degrees(math.atan2(1.2, 3.0))
    _box(stairs, "ramp", [2.925, 0, 0.6], [1.62, 1.0, 0.04], METAL, quat=quat_axis("y", tilt))  # 1.2 m over 3 m

    # Park, south: balls to push, cones, benches.
    for i, (x, y) in enumerate([(-3, -11), (-1, -13), (2, -11.5)]):
        ball = wb.add_body(name=f"ball{i}", pos=[x, y, 0.12])
        ball.add_freejoint()
        ball.add_geom(name=f"ball{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.11, 0, 0],
                      rgba=[(0.9, 0.9, 0.92, 1), (0.95, 0.75, 0.1, 1), (0.2, 0.45, 0.85, 1)][i], mass=0.43)
    for i, (x, y) in enumerate([(5, -9), (6, -10.5), (7, -12), (8, -13.5)]):
        wb.add_geom(name=f"cone{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, 0.2],
                    size=[0.12, 0.2, 0], rgba=CONE)
    for i, (x, y) in enumerate([(-7, -9), (-7, -14)]):
        bench = wb.add_body(name=f"bench{i}", pos=[x, y, 0])
        _box(bench, f"bench{i}_seat", [0, 0, 0.45], [0.8, 0.22, 0.03], WOOD)
        for j, dx in enumerate((-0.7, 0.7)):
            _box(bench, f"bench{i}_leg{j}", [dx, 0, 0.22], [0.04, 0.2, 0.22], METAL)

    # Crates near the road.
    for i, (x, y, s) in enumerate([(-12, 12, 0.3), (-12.5, 12.8, 0.22), (12, -12, 0.35)]):
        crate = wb.add_body(name=f"crate{i}", pos=[x, y, s])
        crate.add_freejoint()
        _box(crate, f"crate{i}_geom", [0, 0, 0], [s, s, s], CRATE, density=200)

    # Trees: solid trunks, leafy crowns you can walk under.
    rng = np.random.default_rng(3)
    spots = [(-26, -14), (-30, -6), (-22, -26), (-9, -24), (8, -26), (-34, -30), (-14, -33), (26, -33),
             (-24, 18), (-35, 34), (-16, 28), (28, -14), (26, 30), (35, 26), (-36, -18), (22, -24)]
    for i, (x, y) in enumerate(spots):
        base = float(terrain_height(np.array(x, float), np.array(y, float)))
        tall = 2.2 + rng.uniform(0, 1.6)
        wb.add_geom(name=f"tree{i}_trunk", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[x, y, base + tall / 2], size=[0.16, tall / 2, 0], rgba=BARK)
        r = 1.0 + rng.uniform(0, 0.6)
        wb.add_geom(name=f"tree{i}_leaves", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                    pos=[x, y, base + tall + r * 0.6], size=[r, r, r * 1.2], rgba=LEAVES, **_visual())

    # Water surface (visual; buoyancy and drag are applied by the simulation).
    wb.add_geom(name="water_lake", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=[LAKE.cx, LAKE.cy, LAKE.level],
                size=[LAKE.rx + 1.2, LAKE.ry + 1.8, 0.004], rgba=(0.16, 0.42, 0.55, 0.72), **_visual())

    hill = float(terrain_height(np.array(-29.0), np.array(27.0)))
    _apple(wb, (2.2, -1.7, 0.05))
    return MapInfo(
        key="open", label="Open world",
        places=[
            Place("home", "Home", 0, 0, 12),
            Place("road", "Road", -18.5, 0, 16),
            Place("stairs", "Stairs", 9, 10, 10),
            Place("park", "Park", 0, -12, 12),
            Place("lake", "Lake", LAKE.cx, LAKE.cy, 24),
            Place("hills", "Hills", -27, 26, 22),
            Place("city", "City", 6, 22, 30),
        ],
        apple_spots={
            "behind the sofa": (2.2, -1.7, 0.05),
            "on the table": (1.75, 1.3, 0.80),
            "under the cabinet": (3.3, -0.2, 0.05),
            "on the shelf": (-1.3, 2.75, 0.50),
            "on the platform": (9.4, 10.0, 1.27),
            "in the park": (0.5, -12.0, 0.05),
            "by the lake": (LAKE.cx - LAKE.rx - 1.5, LAKE.cy, 0.1),
            "on the hilltop": (-29.0, 27.0, hill + 0.1),
            "downtown": (4.0, 22.0, 0.08),
        },
        water=[LAKE],
        spawns={"humanoid": (-2.4, 0.9, 0), "dog": (-2.4, -0.5, 0), "drone": (-2.4, -1.8, 0),
                "fly": (-2.4, 2.0, 0), "car": (-18.5, -6.0, 90)},
        size=HALF,
    )


MAPS = {"sandbox": build_sandbox, "open": build_open}
