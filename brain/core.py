"""Stage 1 brain: the trainable sensorimotor core of the MaleCNS connectome.

The full graph (brain/graph.py, 166,700 neurons) is too big to train with reinforcement
learning on free GPUs. This keeps the same wiring at cell-type level (TechnicalDesign 3.4,
"coarse" variant): one unit per cell type, the synapse counts and neurotransmitter signs
between types preserved, the optic lobes left out (no body has a camera yet).

Kept units:
  body senses  every mechanosensory / proprioceptive / tactile sensory type
  motor        every motor-neuron type
  neck         every descending (brain -> body) and ascending (body -> brain) type
  the rest     the nerve-cord and central-brain types that exchange the most synapses with
               those, up to TARGET_UNITS in total
Edges: type-to-type connections with at least MIN_SYNAPSES synapses among kept units.

Output (data/malecns/core_v1.npz): name, role, sign, size per unit; pre, post, synapses per edge.
"""
from __future__ import annotations

import time

import numpy as np
import pyarrow.feather as pf

from .graph import ANNOTATIONS, DATA, load

CACHE = DATA / "core_v1.npz"
TARGET_UNITS = 2500
MIN_SYNAPSES = 10
ROLES = ["sense", "motor", "descending", "ascending", "cord", "central"]
BODY_SENSE = ("mechanosensory", "mechanosensory_proprioceptive", "mechanosensory_tactile", "mechanosensory_tbc")
OPTIC = ("ol_intrinsic", "visual_projection", "ol_sensory", "visual_centrifugal", "ol_other")


def _role(superclass: str, cls: str) -> int:
    if "sensory" in superclass:
        return 0 if cls in BODY_SENSE else -1          # smell, taste, heat...: not wired to a body yet
    if "motor" in superclass:
        return 1
    if superclass.startswith("descending"):
        return 2
    if superclass.startswith("ascending"):
        return 3
    if superclass.startswith("vnc"):
        return 4
    return 5


def build(verbose: bool = True) -> dict:
    t0 = time.time()
    g = load()
    ann = pf.read_table(ANNOTATIONS, columns=["bodyId", "type", "superclass", "class"]).to_pandas()
    ann = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(g["body_id"])
    sc = ann["superclass"].fillna("unknown").to_numpy()
    cls = ann["class"].fillna("").to_numpy()
    typ = np.where(ann["type"].isna(), "untyped:" + sc, ann["type"].fillna("").to_numpy())
    keep = ~np.isin(sc, OPTIC) & (g["group"] != 1)

    names, unit_of = np.unique(typ[keep], return_inverse=True)
    n_all = len(names)
    unit = np.full(len(typ), -1)
    unit[np.where(keep)[0]] = unit_of
    # role and sign of a type: the majority over its neurons
    roles = np.full(n_all, 5)
    role_n = np.array([_role(s, c) for s, c in zip(sc[keep], cls[keep])])
    order = np.argsort(unit_of, kind="stable")
    bounds = np.searchsorted(unit_of[order], np.arange(n_all + 1))
    signs_n = g["sign"][keep]
    sign = np.zeros(n_all, np.int8)
    size = np.diff(bounds)
    for u in range(n_all):
        members = order[bounds[u]:bounds[u + 1]]
        vals, counts = np.unique(role_n[members], return_counts=True)
        roles[u] = vals[np.argmax(counts)]
        s = signs_n[members]
        sign[u] = int(np.sign(s.sum())) if s.any() else 0

    # type-level synapse counts
    m = (unit[g["pre"]] >= 0) & (unit[g["post"]] >= 0)
    key = unit[g["pre"][m]].astype(np.int64) * n_all + unit[g["post"][m]]
    uniq, inv = np.unique(key, return_inverse=True)
    syn = np.bincount(inv, weights=g["weight"][m]).astype(np.int64)
    pre_t, post_t = uniq // n_all, uniq % n_all

    must = np.isin(roles, [0, 1, 2, 3])
    # everyone else ranked by synapses exchanged with the sensorimotor loop
    touch = np.zeros(n_all)
    np.add.at(touch, pre_t, np.where(must[post_t], syn, 0))
    np.add.at(touch, post_t, np.where(must[pre_t], syn, 0))
    touch[must | (roles < 0)] = -1
    extra = np.argsort(touch)[::-1][: max(TARGET_UNITS - int(must.sum()), 0)]
    chosen = np.zeros(n_all, bool)
    chosen[must] = True
    chosen[extra[touch[extra] > 0]] = True

    new = np.full(n_all, -1)
    new[chosen] = np.arange(chosen.sum())
    e = chosen[pre_t] & chosen[post_t] & (syn >= MIN_SYNAPSES) & (pre_t != post_t)
    out = dict(name=names[chosen].astype(str), role=roles[chosen].astype(np.int8), sign=sign[chosen], size=size[chosen].astype(np.int32),
               pre=new[pre_t[e]].astype(np.int32), post=new[post_t[e]].astype(np.int32), synapses=syn[e].astype(np.int32))
    np.savez_compressed(CACHE, **out)
    if verbose:
        r = out["role"]
        print(f"core: {len(out['name']):,} units ({', '.join(f'{ROLES[i]} {int((r == i).sum())}' for i in range(6))}), "
              f"{len(out['pre']):,} edges, {int(out['synapses'].sum()):,} synapses; "
              f"signs +{int((out['sign'] > 0).sum())} -{int((out['sign'] < 0).sum())} 0:{int((out['sign'] == 0).sum())}; "
              f"{time.time() - t0:.0f} s -> {CACHE.name} ({CACHE.stat().st_size / 1e6:.1f} MB)")
    return out


def load_core() -> dict:
    if not CACHE.exists():
        return build()
    with np.load(CACHE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ============================================================================ v2: two sides, and every sense
CACHE_V2 = DATA / "core_v2.npz"
TARGET_UNITS_V2 = 7000
MIN_SYNAPSES_V2 = 5             # half of v1's: splitting a type by side splits its synapses
MODALITIES = ["body", "vision", "hearing", "smell", "internal"]
HEARING = ("auditory", "wind_gravity")          # Johnston's organ, the fly's ear
CIRCUITS = ("CX", "Kenyon_Cell", "MBON", "DAN", "ALPN", "ALLN", "ALIN", "ALON")   # navigation, learning, smell


def _role_v2(superclass: str, cls: str, subclass: str) -> tuple[int, int]:
    """(role, modality): modality is where a sense neuron's nerve enters, -1 for every other neuron. Visual
    projection neurons are the eye's way in (the columns before them are the eye module, not units); endocrine
    cells sense the body's inner state. Role -1: left out (photoreceptors and optic-lobe columns, taste)."""
    if superclass == "visual_projection":
        return 0, 1
    if superclass.endswith("endocrine"):
        return 0, 4
    if superclass in OPTIC:
        return -1, -1
    if "sensory" in superclass:
        if cls in BODY_SENSE:
            return 0, 2 if subclass in HEARING else 0
        if cls == "olfactory":
            return 0, 3
        return -1, -1
    return _role(superclass, cls), -1


def build_v2(verbose: bool = True) -> dict:
    """The core with both sides of the fly kept apart (one unit per cell type per side), so the brain can tell
    left from right, plus the senses the full robot needs (vision through the visual projection neurons, hearing,
    smell, inner state) and the fly's navigation (central complex), learning (mushroom body) and smell (antennal
    lobe) circuits. Every v1 type is kept on both sides; v1_unit / v1_edge say where each unit and connection
    came from (-1: new), so a brain trained on v1 carries over."""
    t0 = time.time()
    g = load()
    v1 = load_core()
    ann = pf.read_table(ANNOTATIONS, columns=["bodyId", "type", "superclass", "class", "subclass", "somaSide",
                                              "rootSide"]).to_pandas()
    ann = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(g["body_id"])
    sc = ann["superclass"].fillna("unknown").to_numpy()
    cls = ann["class"].fillna("").to_numpy()
    sub = ann["subclass"].fillna("").to_numpy()
    typ = np.where(ann["type"].isna(), "untyped:" + sc, ann["type"].fillna("").to_numpy()).astype(str)
    soma, root = ann["somaSide"].to_numpy(), ann["rootSide"].to_numpy()
    side = np.where(np.isin(soma, ["L", "R"]), soma, np.where(np.isin(root, ["L", "R"]), root, "M")).astype(str)
    rm = np.array([_role_v2(s, c, b) for s, c, b in zip(sc, cls, sub)])
    keep = (rm[:, 0] >= 0) & (g["group"] != 1) | (sc == "visual_projection")

    names, unit_of = np.unique(np.char.add(np.char.add(typ[keep], "|"), side[keep]), return_inverse=True)
    n_all = len(names)
    unit = np.full(len(typ), -1)
    unit[np.where(keep)[0]] = unit_of
    order = np.argsort(unit_of, kind="stable")
    bounds = np.searchsorted(unit_of[order], np.arange(n_all + 1))
    roles, mods = np.full(n_all, 5), np.full(n_all, -1)
    sign, size = np.zeros(n_all, np.int8), np.diff(bounds)
    circuit = np.zeros(n_all, bool)
    kept = np.where(keep)[0]
    for u in range(n_all):
        members = kept[order[bounds[u]:bounds[u + 1]]]
        vals, counts = np.unique(rm[members, 0], return_counts=True)
        roles[u] = vals[np.argmax(counts)]
        mods[u] = np.bincount(rm[members, 1] + 1).argmax() - 1 if roles[u] == 0 else -1
        s = g["sign"][members]
        sign[u] = int(np.sign(s.sum())) if s.any() else 0
        circuit[u] = np.isin(cls[members], CIRCUITS).mean() > 0.5
    base_type = np.array([n.rsplit("|", 1)[0] for n in names])

    m = (unit[g["pre"]] >= 0) & (unit[g["post"]] >= 0)
    key = unit[g["pre"][m]].astype(np.int64) * n_all + unit[g["post"][m]]
    uniq, inv = np.unique(key, return_inverse=True)
    syn = np.bincount(inv, weights=g["weight"][m]).astype(np.int64)
    pre_t, post_t = uniq // n_all, uniq % n_all

    must = np.isin(roles, [0, 1, 2, 3]) | circuit | np.isin(base_type, v1["name"])
    touch = np.zeros(n_all)
    np.add.at(touch, pre_t, np.where(must[post_t], syn, 0))
    np.add.at(touch, post_t, np.where(must[pre_t], syn, 0))
    touch[must] = -1
    extra = np.argsort(touch)[::-1][: max(TARGET_UNITS_V2 - int(must.sum()), 0)]
    chosen = must.copy()
    chosen[extra[touch[extra] > 0]] = True

    new = np.full(n_all, -1)
    new[chosen] = np.arange(chosen.sum())
    e = chosen[pre_t] & chosen[post_t] & (syn >= MIN_SYNAPSES_V2) & (pre_t != post_t)
    pre2, post2 = new[pre_t[e]].astype(np.int32), new[post_t[e]].astype(np.int32)
    v1_of = {n: i for i, n in enumerate(v1["name"])}
    v1_unit = np.array([v1_of.get(t, -1) for t in base_type[chosen]], np.int32)
    v1_key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(v1["pre"], v1["post"]))}
    v1_edge = np.array([v1_key.get((int(v1_unit[a]), int(v1_unit[b])), -1) for a, b in zip(pre2, post2)], np.int32)
    out = dict(name=names[chosen].astype(str), role=roles[chosen].astype(np.int8), modality=mods[chosen].astype(np.int8),
               sign=sign[chosen], size=size[chosen].astype(np.int32), pre=pre2, post=post2,
               synapses=syn[e].astype(np.int32), v1_unit=v1_unit, v1_edge=v1_edge)
    np.savez_compressed(CACHE_V2, **out)
    if verbose:
        r, md = out["role"], out["modality"]
        print(f"core v2: {len(out['name']):,} units ({', '.join(f'{ROLES[i]} {int((r == i).sum())}' for i in range(6))}); "
              f"senses by nerve: {', '.join(f'{MODALITIES[i]} {int((md == i).sum())}' for i in range(len(MODALITIES)))}; "
              f"{len(pre2):,} edges, {int(out['synapses'].sum()):,} synapses; from v1: "
              f"{int((v1_unit >= 0).sum()):,} units, {int((v1_edge >= 0).sum()):,} edges "
              f"({len(np.unique(v1_edge[v1_edge >= 0])):,} of v1's {len(v1['pre']):,}); {time.time() - t0:.0f} s -> "
              f"{CACHE_V2.name} ({CACHE_V2.stat().st_size / 1e6:.1f} MB)")
    return out


def load_core_v2() -> dict:
    if not CACHE_V2.exists():
        return build_v2()
    with np.load(CACHE_V2, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    build()
