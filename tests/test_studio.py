"""Studio server: command parsing and the HTTP / WebSocket API."""
import struct

import pytest
from fastapi.testclient import TestClient

from studio.server.app import app
from studio.server.prompt import parse

ALL = ["humanoid", "dog", "drone", "fly", "car"]


@pytest.mark.parametrize("text, verb, bodies, target", [
    ("find the apple", "find", ALL, "apple"),
    ("dog, go limp", "relax", ["dog"], None),
    ("everyone explore", "babble", ALL, None),
    ("drone, fly to the table", "go", ["drone"], "table"),
    ("drive to the lake", "drive", ["car"], "lake"),
    ("dog, swim", "swim", ["dog"], "lake"),
    ("humanoid, climb the stairs", "go", ["humanoid"], "stairs"),
    ("hello there", None, ALL, None),
])
def test_parse(text, verb, bodies, target):
    p = parse(text, ALL)
    assert (p["verb"], p["bodies"], p["target"]) == (verb, bodies, target)


def test_parse_reports_bodies_not_in_the_scene():
    p = parse("car, drive to the road", ["humanoid"])
    assert p["bodies"] == [] and p["missing"] == ["car"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_default_scene_is_the_sandbox_with_one_body(client):
    cfg = client.get("/api/config").json()
    assert cfg["map"] == "sandbox" and cfg["bodies"] == ["humanoid"]
    scene = client.get("/api/scene").json()
    assert [a["key"] for a in scene["agents"]] == ["humanoid"]
    assert len(client.get("/api/meshes.bin").content) == scene["meshBytes"]


def test_switch_lineup_and_map(client):
    v0 = client.get("/api/config").json()["version"]
    cfg = client.post("/api/config", json={"map": "sandbox", "bodies": ["humanoid", "dog"]}).json()
    assert cfg["bodies"] == ["humanoid", "dog"] and cfg["version"] == v0 + 1
    cfg = client.post("/api/config", json={"map": "open", "bodies": ALL}).json()
    scene = client.get("/api/scene").json()
    assert cfg["map"] == "open" and scene["map"]["key"] == "open" and len(scene["hfields"]) == 1
    assert {p["key"] for p in scene["map"]["places"]} >= {"lake", "road", "hills"}
    assert client.post("/api/config", json={"map": "moon", "bodies": ["dog"]}).status_code == 422
    assert client.post("/api/config", json={"map": "open", "bodies": []}).status_code == 422


def test_inspect_manual_drive_and_errors(client):
    client.post("/api/config", json={"map": "open", "bodies": ALL})
    info = client.get("/api/bodies/humanoid/inspect").json()
    assert len(info["actuators"]) == 29 and info["joints"]
    knee = next(a for a in info["actuators"] if "left_knee" in a["name"])
    assert client.post("/api/bodies/humanoid/manual", json={"index": knee["index"], "value": 0.8}).status_code == 200
    assert client.get("/api/status").json()["bodies"]["humanoid"]["mode"] == "manual"
    assert client.post("/api/bodies/car/drive", json={"throttle": 0.5, "steer": 0.0}).status_code == 200
    assert client.post("/api/bodies/humanoid/drive", json={"throttle": 1, "steer": 0}).status_code == 422
    assert client.post("/api/bodies/drone/nudge", json={"dz": 1.0}).status_code == 200
    assert client.post("/api/bodies/car/goto", json={"place": "lake"}).status_code == 200
    assert client.post("/api/bodies/cat/mode", json={"mode": "hold"}).status_code == 404
    assert client.post("/api/bodies/dog/mode", json={"mode": "dance"}).status_code == 422
    assert client.post("/api/world/apple", json={"spot": "the moon"}).status_code == 422
    assert client.post("/api/prompt", json={"text": "  "}).status_code == 422
    with client.websocket_connect("/ws/inspect/humanoid") as ws:
        live = ws.receive_json()
        assert live["key"] == "humanoid" and len(live["ctrl"]) == 29


def test_prompt_moves_drone_and_car_and_is_honest_about_the_rest(client):
    client.post("/api/config", json={"map": "open", "bodies": ALL})
    r = client.post("/api/prompt", json={"text": "everyone go to the lake"}).json()
    assert "drone" in r["reply"].lower() and "car" in r["reply"].lower() and "trained brain" in r["reply"].lower()
    status = client.get("/api/status").json()
    assert status["bodies"]["car"]["target"] is not None


def test_brain_api_and_training_lock(client):
    from brain.graph import CACHE
    info = client.get("/api/brain").json()
    assert info["built"] is CACHE.exists() and info["trained"] is False
    assert client.post("/api/training/start").status_code == 409
    if CACHE.exists():
        assert info["stages"][0]["status"] == "done"
        blob = client.get("/api/brain/points.bin").content
        assert len(blob) == 4 + info["drawn"] * 13
        with client.websocket_connect("/ws/brain") as ws:
            assert len(ws.receive_bytes()) == info["drawn"]


def test_stream_sends_poses_and_status(client):
    nbody = len(client.get("/api/scene").json()["bodies"])
    with client.websocket_connect("/ws") as ws:
        got_pose = got_status = False
        for _ in range(12):
            msg = ws.receive()
            if msg.get("bytes"):
                t, rtf, n = struct.unpack("<3f", msg["bytes"][:12])
                assert int(n) == nbody and len(msg["bytes"]) == 12 + nbody * 28
                got_pose = True
            elif msg.get("text"):
                got_status = True
            if got_pose and got_status:
                break
        assert got_pose and got_status
