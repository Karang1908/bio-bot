"""Registry of the bodies that can live in the free world."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MENAGERIE = REPO / "third_party" / "mujoco_menagerie"


@dataclass(frozen=True)
class BodySpec:
    key: str                  # short id, also the name prefix inside the world model
    label: str                # shown in the studio
    source: str               # menagerie model file, relative to MENAGERIE
    spawn: tuple[float, float, float]   # world x, y and extra z lift
    yaw_deg: float = 0.0
    magnification: float | None = None  # cm-g-s models only (the fly)
    visual_groups: tuple[int, ...] = (0, 1, 2)
    kind: str = "legged"      # legged | aerial | vehicle
    builder: str | None = None  # built in code instead of loaded from Menagerie
    notes: str = ""
    credit: str = ""
    tags: dict = field(default_factory=dict)

    @property
    def prefix(self) -> str:
        return f"{self.key}/"

    @property
    def path(self) -> Path:
        return MENAGERIE / self.source

    @property
    def available(self) -> bool:
        return self.builder is not None or self.path.exists()


BODIES: tuple[BodySpec, ...] = (
    BodySpec(
        key="humanoid", label="Humanoid", source="unitree_g1/g1.xml",
        spawn=(-2.4, 0.9, 0.0), yaw_deg=0,
        credit="Unitree G1 · MuJoCo Menagerie (BSD-3-Clause, Unitree Robotics)",
    ),
    BodySpec(
        key="dog", label="Dog", source="unitree_go2/go2.xml",
        spawn=(-2.4, -0.5, 0.0), yaw_deg=0,
        credit="Unitree Go2 · MuJoCo Menagerie (BSD-3-Clause, Unitree Robotics)",
    ),
    BodySpec(
        key="drone", label="Drone", source="skydio_x2/x2.xml",
        spawn=(-2.4, -1.8, 0.0), yaw_deg=0, kind="aerial",
        credit="Skydio X2 · MuJoCo Menagerie (Apache-2.0)",
    ),
    BodySpec(
        key="fly", label="Fly", source="flybody/fruitfly.xml",
        spawn=(-2.4, 2.0, 0.0), yaw_deg=0, magnification=100.0,
        notes="Real fly is ~2.5 mm; enlarged 100x with physically consistent scaling.",
        credit="flybody · Janelia / Google DeepMind · MuJoCo Menagerie (Apache-2.0)",
    ),
    BodySpec(
        key="car", label="Car", source="", builder="car",
        spawn=(-18.5, -6.0, 0.0), yaw_deg=90, kind="vehicle",
        notes="A go-kart: rear-wheel drive, front steering.",
        credit="Go-kart · built for this project",
    ),
)

BY_KEY = {b.key: b for b in BODIES}
