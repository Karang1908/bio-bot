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


if __name__ == "__main__":
    build()
