"""Export the compiled world as a static scene description + one binary mesh blob.

The browser builds its 3D scene once from this, then only receives body poses.
"""
from __future__ import annotations

import gzip
import struct

import mujoco
import numpy as np

from world.free_world import World

VISUAL_GROUPS = (0, 1, 2)
DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)


def _name(m, obj, i):
    return mujoco.mj_id2name(m, obj, i) or ""


def build_scene(world: World) -> tuple[dict, bytes]:
    m = world.model
    mesh_ids = sorted({int(m.geom_dataid[g]) for g in range(m.ngeom)
                       if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH})
    owner = {}
    for key, h in world.bodies.items():
        for b in h.body_ids:
            owner[int(b)] = key

    # --- meshes: [float32 verts][uint32 faces] per mesh, concatenated ---
    chunks, meshes, offset = [], [], 0
    for i in mesh_ids:
        va, vn = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
        fa, fn = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
        verts = np.ascontiguousarray(m.mesh_vert[va:va + vn], dtype=np.float32).tobytes()
        faces = np.ascontiguousarray(m.mesh_face[fa:fa + fn], dtype=np.uint32).tobytes()
        meshes.append({"id": i, "name": _name(m, mujoco.mjtObj.mjOBJ_MESH, i),
                       "vertOffset": offset, "vertCount": vn,
                       "faceOffset": offset + len(verts), "faceCount": fn})
        chunks += [verts, faces]
        offset += len(verts) + len(faces)
    # --- heightfields (terrain): float32 heights, row-major (row follows y), in metres ---
    hfields = []
    for i in range(m.nhfield):
        nr, nc = int(m.hfield_nrow[i]), int(m.hfield_ncol[i])
        adr = int(m.hfield_adr[i])
        raw = np.asarray(m.hfield_data[adr:adr + nr * nc], dtype=np.float32)
        geom = next((g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD
                     and m.geom_dataid[g] == i), None)
        base = float(m.geom_pos[geom][2]) if geom is not None else 0.0
        heights = (base + raw * float(m.hfield_size[i][2])).astype(np.float32).tobytes()
        hfields.append({"id": i, "nrow": nr, "ncol": nc, "halfX": float(m.hfield_size[i][0]),
                        "halfY": float(m.hfield_size[i][1]), "offset": offset, "geom": geom})
        chunks.append(heights)
        offset += len(heights)
    blob = b"".join(chunks)

    # --- geoms ---
    geoms = []
    for g in range(m.ngeom):
        group = int(m.geom_group[g])
        mat = int(m.geom_matid[g])
        rgba = m.geom_rgba[g].astype(np.float32)
        metallic, roughness, emission = 0.0, 0.6, 0.0
        if mat >= 0:
            if np.allclose(rgba, DEFAULT_RGBA):
                rgba = m.mat_rgba[mat]
            metallic = float(max(m.mat_metallic[mat], 0.0))
            roughness = float(m.mat_roughness[mat]) if m.mat_roughness[mat] >= 0 else \
                float(np.clip(1.0 - m.mat_shininess[mat], 0.15, 0.95))
            emission = float(m.mat_emission[mat])
        body = int(m.geom_bodyid[g])
        geoms.append({
            "id": g,
            "name": _name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
            "body": body,
            "agent": owner.get(body),
            "type": mujoco.mjtGeom(int(m.geom_type[g])).name.removeprefix("mjGEOM_").lower(),
            "size": [float(x) for x in m.geom_size[g]],
            "pos": [float(x) for x in m.geom_pos[g]],
            "quat": [float(x) for x in m.geom_quat[g]],
            "rgba": [float(x) for x in rgba],
            "metallic": metallic,
            "roughness": roughness,
            "emission": emission,
            "mesh": int(m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None,
            "hfield": int(m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD else None,
            "visual": group in VISUAL_GROUPS,
        })

    agents = []
    for key, h in world.bodies.items():
        s = h.spec
        agents.append({
            "key": key, "label": s.label, "credit": s.credit, "notes": s.notes,
            "kind": s.kind,
            "root": h.root_body, "actuators": int(len(h.actuators)),
            "sensors": int(len(h.sensors)), "joints": int(len(h.dof_adr)),
            "mass": float(m.body_subtreemass[h.root_body]),
            "magnification": s.magnification,
        })

    info = world.map
    scene = {
        "bodies": [{"id": b, "name": _name(m, mujoco.mjtObj.mjOBJ_BODY, b), "agent": owner.get(b)}
                   for b in range(m.nbody)],
        "geoms": geoms,
        "meshes": meshes,
        "hfields": hfields,
        "meshBytes": len(blob),
        "agents": agents,
        "objects": world.objects,
        "timestep": float(m.opt.timestep),
        "map": {
            "key": info.key, "label": info.label, "size": info.size,
            "places": [p.__dict__ for p in info.places],
            "water": [w.__dict__ for w in info.water],
            "appleSpots": list(info.apple_spots),
        } if info else None,
        "lineup": world.lineup,
    }
    return scene, gzip.compress(blob, compresslevel=5)


def pack_poses(sim_time: float, rtf: float, xpos: np.ndarray, xquat: np.ndarray) -> bytes:
    """Binary frame: float32 [time, rtf, nbody, then (x,y,z,qw,qx,qy,qz) per body]."""
    n = xpos.shape[0]
    body = np.concatenate([xpos, xquat], axis=1).astype(np.float32).ravel()
    return struct.pack("<3f", sim_time, rtf, n) + body.tobytes()
