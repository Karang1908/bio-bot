"""A real dog's movements, retargeted onto the Unitree Go2: what the teacher shows.

Source: the dog motion capture of Zhang, Starke, Komura and Saito, "Mode-Adaptive Neural Networks for Quadruped Motion
Control" (SIGGRAPH 2018). 51 recordings of one dog, 147,541 frames at 60 Hz (about 41 minutes), provided by Bandai Namco
Studios under CC BY-NC 4.0 (non-commercial use with attribution). Raw BVH files:
https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip -> data/dog_mocap/

For every frame:
  1. forward kinematics of the dog skeleton gives the trunk (hips -> chest, left <- right) and the four feet;
  2. each foot is expressed relative to its leg root in the trunk's frame and scaled from the dog's leg length to the Go2's;
  3. each Go2 leg is solved in closed form (abduction about x, then a two-link thigh and calf about y), within joint limits;
  4. the second is labelled from the motion itself: stand, walk, trot, pace, canter, sit, lie, jump, turn-left,
     turn-right (labels are measured, not taken from file names).
Output: data/dog_mocap/go2_dog_motions.npz at 50 Hz (the brain's control rate).

  python train/dog_mocap.py              # build the file and print what the dog does, second by second
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "dog_mocap"
OUT = SRC / "go2_dog_motions.npz"

# Go2 legs (go2_mjx.xml): hip joint offset from the trunk, the thigh's sideways offset, thigh and calf lengths
GO2_HIP = {"FL": (0.1934, 0.0465), "FR": (0.1934, -0.0465), "RL": (-0.1934, 0.0465), "RR": (-0.1934, -0.0465)}
GO2_SIDE = {"FL": 0.0955, "FR": -0.0955, "RL": 0.0955, "RR": -0.0955}
L_THIGH = L_CALF = 0.213
LIMITS = {"hip": (-0.9472, 0.9472), "thigh": (-1.4, 2.5), "calf": (-2.6227, -0.84776)}
GO2_STAND_DROP = 0.265                       # foot below the hip joint when standing at the home pose
LEGS = ["FL", "FR", "RL", "RR"]              # Menagerie actuator order: FL hip/thigh/calf, FR, RL, RR
# the dog's skeleton: leg roots and feet (BVH joint names; the feet are the end sites below these joints)
DOG_ROOT = {"FL": "LeftArm", "FR": "RightArm", "RL": "LeftUpLeg", "RR": "RightUpLeg"}
DOG_FOOT = {"FL": "LeftHand", "FR": "RightHand", "RL": "LeftFoot", "RR": "RightFoot"}
LABELS = ["stand", "walk", "trot", "pace", "canter", "run", "sit", "lie", "jump", "turn-left", "turn-right", "other"]


# ============================================================================ BVH
def read_bvh(path: Path):
    """Joints (name, parent, offset, channels), frame time, and the motion as [frames, channels]."""
    lines = path.read_text().split("\n")
    joints, stack, i = [], [], 0
    while not lines[i].strip().startswith("MOTION"):
        t = lines[i].split()
        if t and t[0] in ("ROOT", "JOINT"):
            joints.append({"name": t[1], "parent": stack[-1] if stack else -1, "offset": None, "channels": []})
            current = len(joints) - 1
        elif t and t[0] == "End":
            joints.append({"name": joints[stack[-1]]["name"] + "_end", "parent": stack[-1], "offset": None, "channels": []})
            current = len(joints) - 1
        elif t and t[0] == "{":
            stack.append(current)
        elif t and t[0] == "}":
            stack.pop()
        elif t and t[0] == "OFFSET":
            joints[current]["offset"] = np.array([float(v) for v in t[1:4]])
        elif t and t[0] == "CHANNELS":
            joints[current]["channels"] = t[2:]
        i += 1
    n_frames = int(lines[i + 1].split()[1])
    frame_time = float(lines[i + 2].split()[2])
    data = np.array([[float(v) for v in l.split()] for l in lines[i + 3:i + 3 + n_frames] if l.strip()])
    return joints, frame_time, data


def euler_matrix(angles_deg: np.ndarray, order: str) -> np.ndarray:
    """[F,3] angles in the channel order (e.g. 'ZXY') -> [F,3,3] rotation, applied in that order (BVH convention)."""
    F = angles_deg.shape[0]
    R = np.broadcast_to(np.eye(3), (F, 3, 3)).copy()
    for k, axis in enumerate(order):
        a = np.radians(angles_deg[:, k])
        c, s = np.cos(a), np.sin(a)
        M = np.zeros((F, 3, 3))
        if axis == "X":
            M[:, 0, 0] = 1; M[:, 1, 1] = c; M[:, 1, 2] = -s; M[:, 2, 1] = s; M[:, 2, 2] = c
        elif axis == "Y":
            M[:, 1, 1] = 1; M[:, 0, 0] = c; M[:, 0, 2] = s; M[:, 2, 0] = -s; M[:, 2, 2] = c
        else:
            M[:, 2, 2] = 1; M[:, 0, 0] = c; M[:, 0, 1] = -s; M[:, 1, 0] = s; M[:, 1, 1] = c
        R = R @ M
    return R


def forward_kinematics(joints, data) -> dict:
    """World positions [F,3] of every joint (and end site), in the file's units and axes."""
    F = data.shape[0]
    rot, pos, col = [None] * len(joints), [None] * len(joints), 0
    for j, jt in enumerate(joints):
        ch = jt["channels"]
        local_t = np.broadcast_to(jt["offset"], (F, 3)).copy()
        rot_ch = [c for c in ch if c.endswith("rotation")]
        vals = data[:, col:col + len(ch)] if ch else np.zeros((F, 0))
        for k, c in enumerate(ch):
            if c.endswith("position"):
                local_t[:, "XYZ".index(c[0])] = vals[:, k]
        local_r = euler_matrix(np.stack([vals[:, ch.index(c)] for c in rot_ch], -1), "".join(c[0] for c in rot_ch)) \
            if rot_ch else np.broadcast_to(np.eye(3), (F, 3, 3))
        col += len(ch)
        p = jt["parent"]
        if p < 0:
            rot[j], pos[j] = local_r, local_t
        else:
            rot[j] = rot[p] @ local_r
            pos[j] = pos[p] + np.einsum("fij,fj->fi", rot[p], local_t)
    return {jt["name"]: pos[j] for j, jt in enumerate(joints)}


# ============================================================================ Go2 leg in closed form
def go2_leg_ik(p: np.ndarray, side: float) -> np.ndarray:
    """Foot position p [F,3] relative to the leg's abduction joint, in the trunk frame -> [F,3] (hip, thigh, calf).
    The foot is at R_x(hip) @ (x', side, z'), with (x', z') = -(l2 sin t + l3 sin(t+c), l2 cos t + l3 cos(t+c))."""
    py, pz = p[:, 1], p[:, 2]
    L = np.sqrt(np.maximum(py ** 2 + pz ** 2 - side ** 2, 1e-6))
    hip = np.arctan2(py * L + pz * side, py * side - pz * L)
    a, b = -p[:, 0], L                                               # leg plane: forward (negated) and down
    d2 = np.clip(a ** 2 + b ** 2, (L_THIGH - L_CALF) ** 2 + 1e-6, (L_THIGH + L_CALF - 1e-3) ** 2)
    calf = -np.arccos(np.clip((d2 - L_THIGH ** 2 - L_CALF ** 2) / (2 * L_THIGH * L_CALF), -1.0, 1.0))
    thigh = np.arctan2(a, b) - np.arctan2(L_CALF * np.sin(calf), L_THIGH + L_CALF * np.cos(calf))
    out = np.stack([hip, thigh, calf], -1)
    for k, part in enumerate(("hip", "thigh", "calf")):
        out[:, k] = np.clip(out[:, k], *LIMITS[part])
    return out


def go2_leg_fk(q: np.ndarray, side: float) -> np.ndarray:
    """Inverse of go2_leg_ik (for the round-trip check)."""
    hip, thigh, calf = q[:, 0], q[:, 1], q[:, 2]
    x = -(L_THIGH * np.sin(thigh) + L_CALF * np.sin(thigh + calf))
    z = -(L_THIGH * np.cos(thigh) + L_CALF * np.cos(thigh + calf))
    c, s = np.cos(hip), np.sin(hip)
    return np.stack([x, c * side - s * z, s * side + c * z], -1)


# ============================================================================ one recording
def retarget(path: Path) -> dict:
    joints, dt, data = read_bvh(path)
    P = forward_kinematics(joints, data)
    # which file axis is "up": the one along which the hips sit highest above the feet
    lift = [(P["Hips"][:, a] - P[DOG_FOOT["RL"] + "_end"][:, a]).mean() for a in range(3)]
    up = int(np.argmax(lift))
    # to a right-handed z-up frame (BVH here is y-up)
    to_zup = (lambda v: np.stack([v[:, 2], v[:, 0], v[:, 1]], -1)) if up == 1 else (lambda v: v)
    P = {k: to_zup(v) for k, v in P.items()}
    root = {l: P[DOG_ROOT[l]] for l in LEGS}
    foot = {l: P[DOG_FOOT[l] + "_end"] if DOG_FOOT[l] + "_end" in P else P[DOG_FOOT[l]] for l in LEGS}
    front, rear = (root["FL"] + root["FR"]) / 2, (root["RL"] + root["RR"]) / 2
    left, right = (root["FL"] + root["RL"]) / 2, (root["FR"] + root["RR"]) / 2
    x = front - rear
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = left - right
    y -= (y * x).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    R = np.stack([x, y, z], -1)                                      # trunk frame -> world (columns)
    centre = (front + rear) / 2
    body_len = np.median(np.linalg.norm(front - rear, axis=-1))
    rel = {l: np.einsum("fji,fj->fi", R, foot[l] - root[l]) for l in LEGS}   # foot from its root, trunk frame
    drop = {l: np.median(-rel[l][:, 2]) for l in LEGS}                       # the dog's typical leg drop
    q = []
    for l in LEGS:
        s = GO2_STAND_DROP / max(drop[l], 1e-6)
        p = rel[l] * s
        p[:, 1] = p[:, 1] + GO2_SIDE[l]                               # the Go2 hip's sideways offset
        q.append(go2_leg_ik(p, GO2_SIDE[l]))
    q = np.concatenate(q, -1)                                          # [F,12] FL, FR, RL, RR (hip, thigh, calf)
    scale = GO2_STAND_DROP / np.median(list(drop.values()))
    # trunk motion: height, pitch (nose up > 0), roll, heading; velocities in the heading frame, scaled to Go2 size
    pitch = np.arcsin(np.clip(x[:, 2], -1, 1))
    roll = np.arcsin(np.clip(-y[:, 2], -1, 1))
    heading = np.unwrap(np.arctan2(x[:, 1], x[:, 0]))
    height = (centre[:, 2] - np.minimum.reduce([foot[l][:, 2] for l in LEGS])) * scale
    vel = np.gradient(centre[:, :2], dt, axis=0) * scale
    fwd = np.cos(heading) * vel[:, 0] + np.sin(heading) * vel[:, 1]
    side = -np.sin(heading) * vel[:, 0] + np.cos(heading) * vel[:, 1]
    yaw_rate = np.gradient(heading, dt)
    # a foot is down when it is near its own lowest level (each marker sits at its own height above the ground)
    # and barely moving horizontally
    contact = []
    for l in LEGS:
        z = foot[l][:, 2]
        speed = np.linalg.norm(np.gradient(foot[l][:, :2], dt, axis=0), axis=-1)
        contact.append((z - np.percentile(z, 5) < 0.06 * body_len) & (speed < 0.5 * body_len))
    contact = np.stack(contact, -1)
    return {"q": q, "height": height, "pitch": pitch, "roll": roll, "fwd": fwd, "side": side, "yaw_rate": yaw_rate,
            "contact": contact, "dt": dt, "scale": scale, "body_len": body_len}


def label_seconds(m: dict) -> np.ndarray:
    """One label per frame, decided over a 1 s window centred on it, from what the dog's body does."""
    F = len(m["fwd"])
    win = int(round(1.0 / m["dt"]))
    h_stand = np.percentile(m["height"], 80)
    lab = np.full(F, LABELS.index("other"))
    c = m["contact"].astype(float)
    for t in range(F):
        a, b = max(0, t - win // 2), min(F, t + win // 2 + 1)
        speed = np.hypot(m["fwd"][a:b].mean(), m["side"][a:b].mean())
        turn = m["yaw_rate"][a:b].mean()
        h, pitch = m["height"][a:b].mean() / h_stand, m["pitch"][a:b].mean()
        cc = c[a:b]
        airborne = (cc.sum(1) == 0).mean()
        if airborne > 0.15:                      # in the air: a gallop when fast, a jump otherwise
            lab[t] = LABELS.index("run" if speed > 0.8 else "jump")
        elif speed < 0.08 and abs(turn) < 0.3:
            if pitch > 0.35 and h < 0.85:
                lab[t] = LABELS.index("sit")
            elif h < 0.6 and abs(pitch) < 0.25:
                lab[t] = LABELS.index("lie")
            elif h > 0.85:
                lab[t] = LABELS.index("stand")
        elif speed < 0.15 and abs(turn) > 0.5:
            lab[t] = LABELS.index("turn-left" if turn > 0 else "turn-right")
        elif m["fwd"][a:b].mean() > 0.1:
            sync = lambda i, j: (cc[:, i] == cc[:, j]).mean()
            diag, lat = (sync(0, 3) + sync(1, 2)) / 2, (sync(0, 2) + sync(1, 3)) / 2   # FL+RR/FR+RL, FL+RL/FR+RR
            down = cc.sum(1).mean()
            if down > 2.6:
                lab[t] = LABELS.index("walk")
            elif diag > 0.75 and diag > lat + 0.1:
                lab[t] = LABELS.index("trot")
            elif lat > 0.75 and lat > diag + 0.1:
                lab[t] = LABELS.index("pace")
            else:
                lab[t] = LABELS.index("canter")
    return lab


def resample(x: np.ndarray, dt_in: float, dt_out: float) -> np.ndarray:
    t_in = np.arange(len(x)) * dt_in
    t_out = np.arange(0, t_in[-1], dt_out)
    x2 = x.reshape(len(x), -1).astype(float)
    return np.stack([np.interp(t_out, t_in, x2[:, k]) for k in range(x2.shape[1])], -1).reshape((len(t_out),) + x.shape[1:])


def build(dt_out: float = 0.02) -> dict:
    files = sorted(SRC.glob("*.bvh"))
    if not files:
        sys.exit(f"no BVH files in {SRC}: download MotionCapture.zip from the address in this file's docstring")
    parts = {k: [] for k in ("q", "height", "pitch", "roll", "fwd", "side", "yaw_rate", "contact", "label", "clip")}
    for i, f in enumerate(files):
        m = retarget(f)
        lab = label_seconds(m)
        for k in ("q", "height", "pitch", "roll", "fwd", "side", "yaw_rate"):
            parts[k].append(resample(m[k], m["dt"], dt_out))
        n = len(parts["q"][-1])
        idx = np.clip(np.round(np.arange(n) * dt_out / m["dt"]).astype(int), 0, len(lab) - 1)
        parts["contact"].append(m["contact"][idx])
        parts["label"].append(lab[idx])
        parts["clip"].append(np.full(n, i))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["q"] = out["q"].astype(np.float32)
    np.savez_compressed(OUT, **out, labels=np.array(LABELS), files=np.array([f.name for f in files]), dt=dt_out,
                        meta=json.dumps({"source": "Zhang, Starke, Komura, Saito, SIGGRAPH 2018 (MANN) dog mocap",
                                         "license": "CC BY-NC 4.0, Bandai Namco Studios", "joint_order": "FL FR RL RR x (hip thigh calf)"}))
    return out


def report(out: dict) -> None:
    dt = 0.02
    print(f"{len(out['q']) * dt / 60:.1f} min of dog, retargeted to the Go2 at {1 / dt:.0f} Hz -> {OUT.relative_to(REPO)}")
    for k, name in enumerate(LABELS):
        m = out["label"] == k
        if not m.any():
            print(f"  {name:11s}  none")
            continue
        print(f"  {name:11s} {m.sum() * dt:7.1f} s   forward {out['fwd'][m].mean():+.2f} m/s, sideways {out['side'][m].mean():+.2f}, "
              f"turning {out['yaw_rate'][m].mean():+.2f} rad/s, trunk height {out['height'][m].mean():.2f} m, "
              f"nose {out['pitch'][m].mean():+.2f} rad, feet down {out['contact'][m].sum(1).mean():.1f}")
    lo = np.array([LIMITS[p][0] for p in ("hip", "thigh", "calf")] * 4)
    hi = np.array([LIMITS[p][1] for p in ("hip", "thigh", "calf")] * 4)
    at_limit = ((np.abs(out["q"] - lo) < 1e-4) | (np.abs(out["q"] - hi) < 1e-4)).mean()
    print(f"  joint angles at a Go2 limit (clipped): {at_limit:.1%} of values")


if __name__ == "__main__":
    # round trip of the leg solution on random reachable feet
    rng = np.random.default_rng(0)
    q0 = np.stack([rng.uniform(-0.5, 0.5, 500), rng.uniform(0.0, 1.5, 500), rng.uniform(-2.4, -1.0, 500)], -1)
    err = max(np.abs(go2_leg_ik(go2_leg_fk(q0, s), s) - q0).max() for s in (0.0955, -0.0955))
    print(f"leg solution round trip: max joint error {err:.1e} rad")
    report(build())
