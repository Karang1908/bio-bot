"""Rescale a MuJoCo model spec by physical dimension.

Every quantity is multiplied by length^a * mass^b * time^c according to its
units, so a model authored in one unit system and size (e.g. flybody: a 2.5 mm
fly in cm-g-s) can live in a metre-kilogram-second world at a different size.

Choosing time = sqrt(length_ratio * g_old / g_new) keeps gravity consistent.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

HINGE_LIKE = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_BALL)


def _mul(vec, factor):
    return np.asarray(vec, dtype=float) * factor


def _scale_solref(solref, t):
    # solref[0] > 0 is a time constant; <= 0 means direct stiffness, left alone.
    s = np.asarray(solref, dtype=float).copy()
    if s[0] > 0:
        s[0] *= t
    return s


def _scale_solimp(solimp, length):
    s = np.asarray(solimp, dtype=float).copy()
    s[2] *= length  # width is a distance
    return s


def scale_spec(spec: mujoco.MjSpec, length: float, mass: float, time: float) -> None:
    """Scale `spec` in place. Factors convert old units to new units."""
    force = mass * length / time**2
    torque = force * length

    for b in spec.bodies:
        b.pos = _mul(b.pos, length)
        b.ipos = _mul(b.ipos, length)
        b.mass *= mass
        b.inertia = _mul(b.inertia, mass * length**2)
        b.fullinertia = _mul(b.fullinertia, mass * length**2)

    for g in spec.geoms:
        g.size = _mul(g.size, length)
        g.pos = _mul(g.pos, length)
        g.fromto = _mul(g.fromto, length)
        g.margin *= length
        g.gap *= length
        g.density *= mass / length**3
        if g.mass > 0:
            g.mass *= mass
        fr = np.asarray(g.friction, dtype=float).copy()
        fr[1:] *= length  # torsional and rolling friction have length units
        g.friction = fr
        g.solref = _scale_solref(g.solref, time)
        g.solimp = _scale_solimp(g.solimp, length)

    for s in spec.sites:
        s.pos = _mul(s.pos, length)
        s.size = _mul(s.size, length)
        s.fromto = _mul(s.fromto, length)

    for m in spec.meshes:
        m.scale = _mul(m.scale, length)
        m.refpos = _mul(m.refpos, length)

    joint_type = {}
    for j in spec.joints:
        joint_type[j.name] = j.type
        j.pos = _mul(j.pos, length)
        j.margin *= length
        if j.type in HINGE_LIKE:
            unit = torque
            j.armature *= mass * length**2
        else:  # slide / free: translational units
            unit = force / length
            j.armature *= mass
            if j.type == mujoco.mjtJoint.mjJNT_SLIDE:
                j.range = _mul(j.range, length)
                j.ref *= length
                j.springref *= length
        j.stiffness *= unit
        j.damping *= (torque if j.type in HINGE_LIKE else force / length) * time
        j.frictionloss *= torque if j.type in HINGE_LIKE else force
        j.solref_limit = _scale_solref(j.solref_limit, time)
        j.solref_friction = _scale_solref(j.solref_friction, time)

    for t in spec.tendons:
        t.range = _mul(t.range, length)
        t.springlength = _mul(t.springlength, length)
        t.stiffness *= force / length
        t.damping *= force / length * time
        t.frictionloss *= force
        t.armature *= mass
        t.width *= length
        t.margin *= length

    for a in spec.actuators:
        # Units of actuator force and of actuator length depend on the transmission.
        if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and joint_type.get(a.target) in HINGE_LIKE:
            f_unit, len_unit = torque, 1.0          # torque; length in radians
        elif a.trntype == mujoco.mjtTrn.mjTRN_BODY:
            f_unit, len_unit = force, 1.0           # adhesion: force per unit ctrl
        else:
            f_unit, len_unit = force, length        # slide joint, tendon, site
        gain = np.asarray(a.gainprm, dtype=float).copy()
        bias = np.asarray(a.biasprm, dtype=float).copy()
        # affine gain/bias: force = g0*ctrl + b0 + b1*length + b2*velocity
        gain[0] *= f_unit
        bias[0] *= f_unit
        bias[1] *= f_unit / len_unit
        bias[2] *= f_unit / len_unit * time
        a.gainprm, a.biasprm = gain, bias
        a.forcerange = _mul(a.forcerange, f_unit)
        a.lengthrange = _mul(a.lengthrange, len_unit)
        if len_unit != 1.0 and a.biastype == mujoco.mjtBias.mjBIAS_AFFINE:
            a.ctrlrange = _mul(a.ctrlrange, len_unit)  # position targets in length units
        dyn = np.asarray(a.dynprm, dtype=float).copy()
        if a.dyntype in (mujoco.mjtDyn.mjDYN_FILTER, mujoco.mjtDyn.mjDYN_FILTEREXACT):
            dyn[0] *= time
        a.dynprm = dyn

    for c in spec.cameras:
        c.pos = _mul(c.pos, length)
    for li in spec.lights:
        li.pos = _mul(li.pos, length)
    for k in spec.keys:
        if len(k.qpos) >= 3:
            q = np.asarray(k.qpos, dtype=float).copy()
            q[:3] *= length  # assumes a free root joint, true for all bodies used here
            k.qpos = q


def cgs_to_si_factors(magnification: float, g_old: float = 981.0, g_new: float = 9.81):
    """Factors for a cm-g-s model enlarged `magnification` times into metres."""
    length = 0.01 * magnification
    mass = 1e-3 * magnification**3
    time = math.sqrt(length * g_old / g_new)
    return length, mass, time
