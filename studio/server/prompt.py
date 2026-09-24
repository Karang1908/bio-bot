"""Command box: turn a typed instruction into a goal and run what can run today.

A small rule-based parser, the placeholder for the intent layer (speech -> local
LLM -> goal vector) in TechnicalDesign.md section 6. Until a brain is trained,
only built-in controllers can act on goals: the drone flies to a target and the
car drives to one. Other bodies report that the goal needs a trained brain.
"""
from __future__ import annotations

import re

import mujoco

from .sim import Goal, Simulation

VERBS = [  # order matters: first match wins ("go limp" must hit relax before go)
    ("place", ["put the apple", "place the apple", "move the apple", "hide the apple", "drop the apple"]),
    ("reset", ["reset", "restart", "start over"]),
    ("relax", ["relax", "go limp", "disconnect", "turn off", "power off", "sleep"]),
    ("babble", ["babble", "explore", "wiggle", "move around", "play"]),
    ("bring", ["bring", "fetch", "pick up", "grab", "carry"]),
    ("find", ["find", "search for", "search", "look for", "locate", "where is", "where's"]),
    ("swim", ["swim", "dive", "go swimming"]),
    ("drive", ["drive", "ride", "race"]),
    ("go", ["go to", "fly to", "walk to", "move to", "head to", "run to", "climb", "go"]),
    ("hold", ["stand", "hold", "stay", "hover", "balance", "stop", "freeze", "halt", "connect", "brake"]),
]

BODY_WORDS = {
    "humanoid": ["humanoid", "human", "g1", "person", "robot"],
    "dog": ["dog", "quadruped", "go2", "puppy"],
    "drone": ["drone", "quadcopter", "x2", "copter"],
    "fly": ["fly", "insect", "bug"],
    "car": ["car", "kart", "go-kart", "vehicle"],
}
ALL_WORDS = ["everyone", "everybody", "all bodies", "all of you", "team", "all"]

# Things you can name as a target; places come from the current map.
FURNITURE = {"table": ["table"], "sofa": ["sofa", "couch"], "cabinet": ["cabinet", "cupboard"],
             "shelf": ["shelf", "shelves", "bookshelf"], "platform": ["platform"]}
PLACE_WORDS = {"home": ["home", "house", "room", "inside"], "road": ["road", "street", "track"],
               "lake": ["lake", "water", "pond", "shore"], "hills": ["hill", "hills", "hilltop", "mountain"],
               "stairs": ["stairs", "steps", "ramp"], "park": ["park", "playground", "garden"],
               "stage": ["stage", "centre", "center"],
               "city": ["city", "downtown", "town", "buildings", "skyscraper"]}
FURNITURE_GEOM = {"table": "table_top", "sofa": "sofa_seat", "cabinet": "cabinet_box",
                  "shelf": "shelf_board2", "platform": "platform"}


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def parse(text: str, bodies_present: list[str] | None = None, spots: list[str] | None = None) -> dict:
    t = text.lower().strip()
    verb = next((v for v, words in VERBS if any(_has(t, w) for w in words)), None)
    present = bodies_present or list(BODY_WORDS)
    named = [k for k, words in BODY_WORDS.items() if any(_has(t, w) for w in words)]
    # "fly to X" is the verb, not the fly body, unless the fly is named another way
    if _has(t, "fly to") and not re.search(r"\bthe fly\b|\bfly,", t):
        named = [k for k in named if k != "fly"]
    if verb == "drive" and not named:
        named = ["car"]
    everyone = not named or any(_has(t, w) for w in ALL_WORDS)
    bodies = [k for k in present if everyone or k in named]
    missing = [k for k in named if k not in present]
    target = "apple" if any(_has(t, w) for w in ("apple", "fruit")) else None
    target = target or next((k for k, words in FURNITURE.items() if any(_has(t, w) for w in words)), None)
    target = target or next((k for k, words in PLACE_WORDS.items() if any(_has(t, w) for w in words)), None)
    if verb == "swim" and target is None:
        target = "lake"
    spot = next((s for s in (spots or []) if all(_has(t, w) for w in s.split() if len(w) > 3)), None)
    return {"verb": verb, "bodies": bodies, "missing": missing, "target": target, "spot": spot}


def _target_point(sim: Simulation, target: str):
    if target == "apple":
        return sim.object_position("apple")
    info = sim.world.map
    place = next((p for p in (info.places if info else []) if p.key == target), None)
    if place is not None:
        return [place.x, place.y, 0.0]
    geom = FURNITURE_GEOM.get(target)
    if geom:
        g = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if g >= 0:
            return [float(x) for x in sim.d.geom_xpos[g]]
    return None


def execute(sim: Simulation, text: str) -> dict:
    present = list(sim.world.bodies)
    p = parse(text, present, list(sim.apple_spots))
    verb, bodies, target = p["verb"], p["bodies"], p["target"]
    labels = {k: sim.world.bodies[k].spec.label for k in bodies}
    actions: list[str] = []
    note = ""
    if p["missing"]:
        names = ", ".join(m.title() for m in p["missing"])
        note = f" ({names} isn't in this scene; add it from the body bar.)"

    if verb is None:
        return {"parsed": p, "actions": actions, "reply": "I didn't recognise an instruction. Try: 'find the apple', "
                "'car, drive to the lake', 'everyone explore', 'drone, fly to the hills', 'dog relax'." + note}
    if not bodies:
        return {"parsed": p, "actions": actions, "reply": "No body in this scene can do that." + note}

    if verb == "reset":
        sim.reset()
        return {"parsed": p, "actions": ["world reset"], "reply": "World reset."}

    if verb == "place":
        if p["spot"] is None:
            return {"parsed": p, "actions": actions,
                    "reply": f"Where should the apple go? Places: {', '.join(sim.apple_spots)}."}
        sim.place_apple(p["spot"])
        return {"parsed": p, "actions": [f"apple -> {p['spot']}"], "reply": f"Apple placed {p['spot']}."}

    if verb in ("relax", "babble", "hold"):
        mode = {"relax": "off", "babble": "babble", "hold": "hold"}[verb]
        for k in bodies:
            sim.set_mode(k, mode)
            actions.append(f"{labels[k]} -> {mode}")
        word = {"off": "relaxing", "babble": "exploring", "hold": "holding still"}[mode]
        return {"parsed": p, "actions": actions, "reply": f"{', '.join(labels.values())}: {word}." + note}

    # find / go / bring / swim / drive: goal-directed, needs a target
    if target is None:
        return {"parsed": p, "actions": actions,
                "reply": "Where to? Name a place (lake, road, hills, park, stairs, home) or the apple." + note}
    point = _target_point(sim, target)
    if point is None:
        return {"parsed": p, "actions": actions, "reply": f"There's no {target} in this scene." + note}

    executed, waiting = [], []
    for k in bodies:
        kind = sim.world.bodies[k].spec.kind
        if kind in ("aerial", "vehicle"):
            sim.go_to(k, point)
            executed.append(k)
            actions.append(f"{labels[k]} -> {target} (built-in controller)")
        else:
            waiting.append(labels[k])
    sim.goal = Goal(text=text, verb=verb, target=target, bodies=bodies,
                    status="running" if executed else "needs trained brain",
                    executed_by="built-in controllers (teachers)" if executed else None,
                    point=[round(float(x), 3) for x in point])
    parts = []
    how = {"drone": "flying", "car": "driving"}
    if executed:
        parts.append(" and ".join(f"The {k} is {how.get(k, 'heading')} to the {target}" for k in executed)
                     + " using its built-in controller.")
    if waiting:
        names = [f"the {w.lower()}" for w in waiting]
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        parts.append(f"{who[0].upper()}{who[1:]} can't {verb} on {'its' if len(names) == 1 else 'their'} own yet: "
                     f"that needs a trained brain (training stage 1). Goal recorded.")
    if verb == "swim" and executed:
        parts.append("Only walking and swimming bodies can really swim; flying over or driving to the shore is what the built-ins can do.")
    sim._event("goal", f"{verb} {target} · {', '.join(labels.values())}")
    return {"parsed": p, "actions": actions, "reply": " ".join(parts) + note}
