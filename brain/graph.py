"""Stage 0: turn the MaleCNS connectome tables into one cached, signed graph.

Output (data/malecns/graph_v1.npz):
  body_id  int64[N]   MaleCNS body ids of the annotated neurons
  group    uint8[N]   index into GROUPS (plain-language neuron groups)
  sign     int8[N]    +1 excitatory, -1 inhibitory, 0 unknown / neuromodulatory
  pos      float32[N,3] soma position in micrometres (sensory neurons: see below)
  visible  bool[N]    has a position to draw
  eye      bool[N]    photoreceptors (visual input neurons; no camera feeds them yet)
  pre, post int32[E]  edge endpoints (indices into the arrays above)
  weight   int32[E]   synapse count

Sensory neurons have their cell bodies outside the CNS, so the tables give them no
soma position; they are drawn at the synapse-weighted mean position of the neurons
they feed (falling back to the neurons feeding them).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as pf

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "malecns"
ANNOTATIONS = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = DATA / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
CACHE = DATA / "graph_v1.npz"
VOXEL_UM = 0.008  # MaleCNS voxels are 8 nm

# Plain-language groups, in the order signals flow.
GROUPS = [
    ("senses", "Senses", "Touch, stretch, taste and smell neurons: where the bodies' signals come in"),
    ("vision", "Vision", "The optic lobes: the fly's visual system (no camera is connected yet)"),
    ("central", "Central brain", "Where signals meet: learning, navigation, decisions"),
    ("descending", "Brain → body", "Descending neurons carrying commands down the neck"),
    ("ascending", "Body → brain", "Ascending neurons carrying body state up the neck"),
    ("cord", "Nerve cord", "The fly's spinal cord: local reflexes and rhythms"),
    ("motor", "Muscles", "Motor and output neurons: where commands leave the nervous system"),
]

SIGN = {"acetylcholine": 1, "glutamate": -1, "gaba": -1, "histamine": -1}


def group_of(superclass: str) -> int:
    s = superclass
    if "sensory" in s:
        return 0
    if "motor" in s or "efferent" in s:
        return 6
    if s.startswith("descending"):
        return 3
    if s.startswith("ascending"):
        return 4
    if s.startswith("vnc"):
        return 5
    if s.startswith("ol_") or s.startswith("visual"):
        return 1
    return 2


def build(verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)

    ann = pd.read_feather(ANNOTATIONS, columns=["bodyId", "superclass", "somaLocation"])
    ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
    body_id = ann.bodyId.to_numpy(np.int64)
    n = len(body_id)
    group = np.array([group_of(s) for s in ann.superclass], dtype=np.uint8)

    nt = pd.read_feather(NEUROTRANSMITTERS, columns=["body", "consensus_nt"]).set_index("body").consensus_nt
    sign = np.array([SIGN.get(nt.get(b), 0) for b in body_id], dtype=np.int8)

    pos = np.full((n, 3), np.nan, dtype=np.float32)
    has = ann.somaLocation.notna().to_numpy()
    pos[has] = np.stack(ann.somaLocation[has].to_numpy()).astype(np.float32) * VOXEL_UM
    log(f"{n:,} neurons; {int(has.sum()):,} with soma positions ({time.perf_counter()-t0:.1f}s)")

    # Stream the 150M-row weights table; keep edges between annotated neurons only.
    table = pf.read_table(WEIGHTS, memory_map=True)
    ids = __import__("pyarrow").array(body_id)
    pres, posts, ws = [], [], []
    for batch in table.to_batches(max_chunksize=8_000_000):
        pre = batch.column("body_pre")
        post = batch.column("body_post")
        keep = pc.and_(pc.is_in(pre, value_set=ids), pc.is_in(post, value_set=ids))
        pre_np = pc.filter(pre, keep).to_numpy()
        post_np = pc.filter(post, keep).to_numpy()
        pres.append(np.searchsorted(body_id, pre_np).astype(np.int32))
        posts.append(np.searchsorted(body_id, post_np).astype(np.int32))
        ws.append(pc.filter(batch.column("weight"), keep).to_numpy().astype(np.int32))
    pre = np.concatenate(pres)
    post = np.concatenate(posts)
    weight = np.concatenate(ws)
    log(f"{len(pre):,} edges between annotated neurons, {int(weight.sum()):,} synapses "
        f"({time.perf_counter()-t0:.1f}s)")

    # Place neurons without a soma at the weighted mean of their partners' positions.
    for src, dst in ((pre, post), (post, pre)):  # first: neurons they feed; then: feeding them
        missing = np.isnan(pos[:, 0])
        edge = missing[src] & ~np.isnan(pos[dst, 0])
        if not edge.any():
            break
        acc = np.zeros((n, 3))
        wsum = np.zeros(n)
        np.add.at(acc, src[edge], pos[dst[edge]] * weight[edge, None])
        np.add.at(wsum, src[edge], weight[edge])
        fill = missing & (wsum > 0)
        pos[fill] = (acc[fill] / wsum[fill, None]).astype(np.float32)
    visible = ~np.isnan(pos[:, 0])
    rng = np.random.default_rng(0)
    placed = visible & ~has
    pos[placed] += rng.normal(0, 4.0, size=(int(placed.sum()), 3)).astype(np.float32)  # spread stacked points
    log(f"{int(visible.sum()):,} neurons drawable ({int(placed.sum()):,} placed from partners)")

    eye = (ann.superclass == "ol_sensory").to_numpy()
    out = dict(body_id=body_id, group=group, sign=sign, pos=np.nan_to_num(pos), visible=visible, eye=eye,
               pre=pre, post=post, weight=weight)
    np.savez(CACHE, **out)
    log(f"saved {CACHE.relative_to(REPO)} ({CACHE.stat().st_size/1e6:.0f} MB, {time.perf_counter()-t0:.1f}s)")
    return out


def load() -> dict:
    if not CACHE.exists():
        return build()
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    build()
