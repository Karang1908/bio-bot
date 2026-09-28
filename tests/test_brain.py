"""Stage 0 brain: the cached MaleCNS graph and the rate-network runtime."""
import numpy as np
import pytest

from brain.graph import CACHE, GROUPS

pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="MaleCNS graph not built (scripts/fetch_assets.py --connectome)")


@pytest.fixture(scope="module")
def graph():
    from brain.graph import load
    return load()


def test_graph_matches_malecns(graph):
    assert len(graph["body_id"]) == 166_700              # annotated neurons in MaleCNS v1.0
    assert len(graph["pre"]) == 25_582_938               # neuron-to-neuron connections among them
    assert set(np.unique(graph["sign"])) <= {-1, 0, 1}
    assert graph["group"].max() == len(GROUPS) - 1
    assert graph["visible"].mean() > 0.99                 # almost every neuron can be drawn
    assert np.isfinite(graph["pos"][graph["visible"]]).all()


def test_brain_is_silent_without_input_and_bounded(graph):
    from brain.runtime import Brain
    b = Brain(graph)
    for _ in range(10):
        b.step()
    assert b.activity().max() == 0.0                      # no input, no activity
    drive = np.zeros(b.n, np.float32)
    drive[np.random.default_rng(1).choice(b.sense_pool, 4000, replace=False)] = 1.0  # like real channels
    peak: dict[str, float] = {}
    for _ in range(30):
        b.step(drive)
        act = b.activity()
        assert 0.0 <= act.min() and act.max() <= 1.0
        for g in b.group_summary(act):
            peak[g["key"]] = max(peak.get(g["key"], 0.0), g["active"])
    for key in ("central", "descending", "ascending", "cord"):  # the signal spreads past the senses
        assert peak[key] > 0.01, (key, peak[key])


def test_activity_fades_after_input_stops(graph):
    from brain.runtime import Brain
    b = Brain(graph)
    drive = np.zeros(b.n, np.float32)
    drive[np.random.default_rng(1).choice(b.sense_pool, 4000, replace=False)] = 1.0
    for _ in range(40):
        b.step(drive)
    for _ in range(120):
        b.step()
    assert (b.activity() > 0.1).mean() < 0.01              # fatigue stops it locking on


def test_trainable_core_keeps_the_sensorimotor_loop():
    """Stage 1 core (brain/core.py): body senses in, motor neurons out, real wiring between."""
    from brain.core import ROLES, load_core
    c = load_core()
    role, sign, pre, post = c["role"], c["sign"], c["pre"], c["post"]
    count = {name: int((role == i).sum()) for i, name in enumerate(ROLES)}
    assert len(role) == 2500 and count["sense"] > 150 and count["motor"] > 150
    assert (sign[role == 0] > 0).mean() > 0.9                     # mechanosensory neurons are excitatory
    assert (c["synapses"] >= 10).all() and (pre != post).all()
    live = sign[pre] != 0                                         # a signless unit's outputs carry nothing
    reach = role == 0
    for _ in range(2):
        nxt = reach.copy()
        nxt[post[live & reach[pre]]] = True
        reach = nxt
    assert reach[role == 1].mean() > 0.95                         # senses reach motor neurons within 2 steps
