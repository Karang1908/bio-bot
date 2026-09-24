"""Bio-Bot Studio backend: runs the world and serves the studio.

Run:  .venv/bin/python -m uvicorn studio.server.app:app --port 8765
"""
from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import threading
import warnings
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from world.bodies import BODIES
from world.controllers import MODES
from world.free_world import build
from world.maps import MAPS

from . import prompt
from .brain_service import BrainService
from .scene import build_scene, pack_poses
from .sim import Simulation

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
POSE_HZ = 30
STATUS_HZ = 5
DEFAULT_CONFIG = {"map": "sandbox", "bodies": ["humanoid"]}

state: dict = {"version": 0}
_switch_lock = threading.Lock()
_scene_cache: dict[tuple, tuple[dict, bytes]] = {}


def _available_bodies() -> list[str]:
    return [b.key for b in BODIES if b.available]


def _make(map_key: str, bodies: list[str]):
    """Build world + scene for a config (scenes are cached; physics is always fresh)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict")
        world = build(map_key, bodies)
    ck = (map_key, tuple(world.lineup))
    if ck not in _scene_cache:
        _scene_cache[ck] = build_scene(world)
    scene, meshes_gz = _scene_cache[ck]
    return world, scene, meshes_gz


def _switch(map_key: str, bodies: list[str]) -> None:
    with _switch_lock:
        world, scene, meshes_gz = _make(map_key, bodies)
        sim = Simulation(world)
        sim.start()
        old = state.get("sim")
        state["version"] += 1
        scene = {**scene, "version": state["version"]}
        state.update(world=world, scene=scene, meshes_gz=meshes_gz, sim=sim,
                     config={"map": map_key, "bodies": world.lineup})
        if "brain" in state:
            state["brain"].attach(sim)
        if old is not None:
            old.stop()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    _switch(DEFAULT_CONFIG["map"], DEFAULT_CONFIG["bodies"])
    brain = BrainService(state["sim"])
    brain.start()
    state["brain"] = brain
    state["points_gz"] = gzip.compress(brain.points(), compresslevel=5) if brain.available else b""
    yield
    brain.stop()
    state["sim"].stop()


app = FastAPI(title="Bio-Bot Studio", lifespan=lifespan)


def sim() -> Simulation:
    return state["sim"]


def _body(key: str) -> str:
    if key not in state["world"].bodies:
        raise HTTPException(404, f"{key!r} isn't in this scene")
    return key


# ---------------------------------------------------------------------------- scene + config
class ConfigIn(BaseModel):
    map: str
    bodies: list[str]


@app.get("/api/config")
def get_config():
    return {**state["config"], "version": state["version"],
            "maps": [{"key": k, "label": {"sandbox": "Sandbox", "open": "Open world"}[k]} for k in MAPS],
            "available": _available_bodies(),
            "all": [{"key": b.key, "label": b.label, "kind": b.kind} for b in BODIES]}


@app.post("/api/config")
async def set_config(body: ConfigIn):
    if body.map not in MAPS:
        raise HTTPException(422, f"map must be one of {list(MAPS)}")
    if not body.bodies:
        raise HTTPException(422, "pick at least one body")
    bad = [b for b in body.bodies if b not in _available_bodies()]
    if bad:
        raise HTTPException(422, f"unknown or unavailable bodies: {bad}")
    await asyncio.to_thread(_switch, body.map, body.bodies)
    return get_config()


@app.get("/api/scene")
def get_scene():
    return state["scene"]


@app.get("/api/meshes.bin")
def get_meshes():
    return Response(state["meshes_gz"], media_type="application/octet-stream",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------- world
class ModeIn(BaseModel):
    mode: str


class ValueIn(BaseModel):
    value: float


class FlagIn(BaseModel):
    paused: bool


class SpotIn(BaseModel):
    spot: str


class PromptIn(BaseModel):
    text: str


class ManualIn(BaseModel):
    index: int
    value: float


class DriveIn(BaseModel):
    throttle: float
    steer: float


class NudgeIn(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


class GotoIn(BaseModel):
    place: str


def _run(fn):
    try:
        return fn()
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e).strip("'"))


@app.get("/api/status")
def get_status():
    return {**sim().status(), "version": state["version"]}


@app.post("/api/bodies/{key}/mode")
def set_mode(key: str, body: ModeIn):
    if body.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {list(MODES)}")
    sim().set_mode(_body(key), body.mode)
    return sim().status()["bodies"][key]


@app.post("/api/bodies/{key}/strength")
def set_strength(key: str, body: ValueIn):
    sim().set_strength(_body(key), body.value)
    return sim().status()["bodies"][key]


@app.post("/api/bodies/{key}/reset")
def reset_body(key: str):
    sim().reset_body(_body(key))
    return sim().status()["bodies"][key]


@app.post("/api/bodies/{key}/manual")
def manual(key: str, body: ManualIn):
    _run(lambda: sim().set_manual(_body(key), body.index, body.value))
    return {"ok": True}


@app.post("/api/bodies/{key}/drive")
def drive(key: str, body: DriveIn):
    _run(lambda: sim().set_drive(_body(key), body.throttle, body.steer))
    return {"ok": True}


@app.post("/api/bodies/{key}/nudge")
def nudge(key: str, body: NudgeIn):
    _run(lambda: sim().nudge(_body(key), body.dx, body.dy, body.dz))
    return {"ok": True}


@app.post("/api/bodies/{key}/goto")
def goto(key: str, body: GotoIn):
    info = state["world"].map
    place = next((p for p in info.places if p.key == body.place), None)
    if place is None:
        raise HTTPException(422, f"unknown place {body.place!r}")
    _run(lambda: sim().go_to(_body(key), [place.x, place.y, 0.0]))
    return {"ok": True}


@app.get("/api/bodies/{key}/inspect")
def inspect(key: str):
    return sim().inspect(_body(key))


@app.post("/api/world/reset")
def reset_world():
    sim().reset()
    return sim().status()


@app.post("/api/world/pause")
def pause(body: FlagIn):
    sim().paused = body.paused
    return {"paused": sim().paused}


@app.post("/api/world/speed")
def speed(body: ValueIn):
    sim().speed = float(min(max(body.value, 0.1), 4.0))
    return {"speed": sim().speed}


@app.get("/api/world/spots")
def spots():
    return list(sim().apple_spots)


@app.post("/api/world/apple")
def apple(body: SpotIn):
    _run(lambda: sim().place_apple(body.spot))
    return sim().status()["apple"]


@app.post("/api/prompt")
def run_prompt(body: PromptIn):
    if not body.text.strip():
        raise HTTPException(422, "empty command")
    return prompt.execute(sim(), body.text)


# ---------------------------------------------------------------------------- brain
STAGES = [
    ("0", "Build the brain", "The MaleCNS connectome running as a live network"),
    ("1", "Teach the bodies one by one", "Each body learns itself from its own senses"),
    ("2", "Teach them together", "One brain learns to use several bodies at once"),
    ("3", "Remove the teachers", "Only the brain's own sense of success remains"),
    ("4", "Keep learning alone", "On-device learning after training"),
]


class ListenIn(BaseModel):
    on: bool


def brain_service() -> BrainService:
    return state["brain"]


@app.get("/api/brain")
def brain():
    svc = brain_service()
    built = svc.available
    stages = [{"id": i, "name": n, "detail": d,
               "status": ("done" if i == "0" else "next" if i == "1" else "locked") if built
               else ("next" if i == "0" else "locked")} for i, n, d in STAGES]
    kaggle_ready = (Path.home() / ".kaggle" / "kaggle.json").exists() or (Path.home() / ".kaggle" / "access_token").exists()
    return {**svc.info(), "built": built, "trained": False, "stages": stages, "kaggle": kaggle_ready}


@app.post("/api/brain/listen")
def brain_listen(body: ListenIn):
    if not brain_service().available:
        raise HTTPException(409, "The brain isn't built yet.")
    brain_service().set_listening(body.on)
    return {"listening": brain_service().listening}


@app.get("/api/brain/points.bin")
def brain_points():
    if not brain_service().available:
        raise HTTPException(404, "The brain isn't built yet.")
    return Response(state["points_gz"], media_type="application/octet-stream",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"})


@app.post("/api/training/start")
def start_training():
    raise HTTPException(409, "Training isn't built yet. Stage 1 (teaching the bodies one by one) comes next.")


# ---------------------------------------------------------------------------- streams
@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    tick = 0
    try:
        while True:
            t, rtf, xpos, xquat = sim().poses()
            await ws.send_bytes(pack_poses(t, rtf, xpos, xquat))
            if tick % (POSE_HZ // STATUS_HZ) == 0:
                await ws.send_text(json.dumps({"type": "status", "version": state["version"], **sim().status()}))
            tick += 1
            await asyncio.sleep(1 / POSE_HZ)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws/inspect/{key}")
async def inspect_stream(ws: WebSocket, key: str):
    await ws.accept()
    try:
        while True:
            if key in state["world"].bodies:
                await ws.send_text(json.dumps({"type": "live", "key": key, "version": state["version"],
                                               **sim().live(key)}))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws/brain")
async def brain_stream(ws: WebSocket):
    await ws.accept()
    svc = brain_service()
    tick = 0
    try:
        while True:
            if svc.available:
                await ws.send_bytes(svc.frame())
            if tick % 5 == 0:
                await ws.send_text(json.dumps({"type": "brain", **svc.info()}))
            tick += 1
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        return


# ---------------------------------------------------------------------------- web app
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = WEB_DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/")
    def root():
        return JSONResponse({"detail": "Frontend not built. Run `npm run build` in studio/web, "
                                       "or use the Vite dev server on :5173."})
