"""Worlds build, are physically sane, and every controller mode is stable."""
import warnings

import mujoco
import numpy as np
import pytest

from world.controllers import BodyController
from world.free_world import build
from world.maps import BUILDINGS, CITY, HALF, LAKE, terrain_height
from world.scale import cgs_to_si_factors
from world.water import WaterForces


def _build(*a, **k):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict")
        return build(*a, **k)


@pytest.fixture(scope="module")
def world():
    return _build("open")


def _run(world, mode, seconds, water=False, d=None):
    m = world.model
    if d is None:
        d = mujoco.MjData(m)
        d.qpos[:] = world.home_qpos
    ctrls = {k: BodyController(world, k, seed=1) for k in world.bodies}
    for c in ctrls.values():
        c.set_mode(mode)
    wf = WaterForces(m, world.map.water, np.arange(1, m.nbody)) if water else None
    try:
        for i in range(int(seconds / m.opt.timestep)):
            if i % 2 == 0:
                for c in ctrls.values():
                    d.ctrl[c.h.actuators] = c.compute(d, 2 * m.opt.timestep)
                if wf:
                    wf.apply(d)
            mujoco.mj_step(m, d)
            assert np.isfinite(d.qpos).all(), f"{mode}: NaN at step {i}"
    finally:
        for c in ctrls.values():
            c.set_enabled(True)  # the model is shared across tests
    return d, ctrls


def test_open_world_has_every_body(world):
    assert set(world.bodies) == {"humanoid", "dog", "drone", "fly", "car"}
    assert {p.key for p in world.map.places} >= {"home", "road", "lake", "hills", "stairs", "park"}


def test_sandbox_lineups():
    for lineup in (["humanoid"], ["humanoid", "dog"], ["humanoid", "dog", "drone"], None):
        w = _build("sandbox", lineup)
        assert w.lineup == (lineup or ["humanoid", "dog", "drone", "fly", "car"])
    with pytest.raises(ValueError):
        _build("sandbox", ["cat"])


def test_terrain_matches_the_heightfield(world):
    m = world.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for x, y in ((0.0, 0.0), (-29.0, 27.0), (20.0, -28.0)):
        dist = mujoco.mj_ray(m, d, np.array([x, y, 20.0]), np.array([0, 0, -1.0]), None, 1, -1, np.zeros(1, np.int32))
        expected = float(terrain_height(np.array(x), np.array(y)))
        assert 20 - dist == pytest.approx(expected, abs=0.02), (x, y)


def test_terrain_meets_the_plain_and_downtown_is_flat_and_solid(world):
    edge = np.linspace(-HALF, HALF, 81)
    for xs, ys in ((edge, np.full_like(edge, HALF)), (np.full_like(edge, -HALF), edge)):
        assert np.abs(terrain_height(xs, ys)).max() < 1e-9           # no cliff where the map ends
    x0, y0, x1, y1 = CITY
    X, Y = np.meshgrid(np.linspace(x0, x1, 30), np.linspace(y0, y1, 30))
    assert np.abs(terrain_height(X, Y)).max() < 1e-9                 # buildings stand on flat ground
    m = world.model
    for name, *_ in BUILDINGS:                                        # buildings collide (not just visual)
        g = m.geom(name)
        assert g.contype[0] and g.conaffinity[0], name


def test_fly_scaling_keeps_physics_consistent(world):
    length, mass, time = cgs_to_si_factors(100)
    assert (length, time) == pytest.approx((1.0, 10.0))
    assert world.model.body_subtreemass[world.bodies["fly"].root_body] == pytest.approx(1.0, rel=0.05)


def test_hold_keeps_everyone_up(world):
    d, ctrls = _run(world, "hold", 3.0)
    for key, c in ctrls.items():
        up = d.xmat[c.h.root_body].reshape(3, 3)[2, 2]
        assert up > 0.95, f"{key} tipped over under hold (upright={up:.2f})"
    assert d.xpos[ctrls["drone"].h.root_body][2] == pytest.approx(1.2, abs=0.1)


def test_off_and_babble_are_stable(world):
    d, ctrls = _run(world, "off", 1.5)
    assert np.abs(d.actuator_force[ctrls["dog"].h.actuators]).max() == pytest.approx(0.0)
    _run(world, "babble", 1.5)


def test_car_drives_itself_to_a_target():
    w = _build("open", ["car"])
    m = w.model
    d = mujoco.MjData(m)
    d.qpos[:] = w.home_qpos
    c = BodyController(w, "car")
    c.drive.target = np.array([-18.5, 12.0])
    for i in range(10000):
        if i % 2 == 0:
            d.ctrl[c.h.actuators] = c.compute(d, 0.002)
        mujoco.mj_step(m, d)
    assert c.drive.target is None                                          # reached (within 1.5 m)
    assert np.linalg.norm(d.xpos[c.h.root_body][:2] - [-18.5, 12.0]) < 2.0


def test_bodies_float_in_the_lake():
    w = _build("open", ["dog"])
    h = w.bodies["dog"]
    d = mujoco.MjData(w.model)
    d.qpos[:] = w.home_qpos
    d.qpos[h.qpos_adr[:3]] = [LAKE.cx, LAKE.cy, LAKE.level + 0.3]
    d, _ = _run(w, "hold", 5.0, water=True, d=d)
    z = d.xpos[h.root_body][2]
    assert LAKE.level - 0.4 < z < LAKE.level + 0.3          # floating near the surface, not on the bottom
