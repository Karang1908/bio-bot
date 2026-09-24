"""Download the third-party assets the studio needs (kept out of git).

- MuJoCo Menagerie models: unitree_g1, unitree_go2, skydio_x2, flybody
- Sky HDRI for the studio's lighting (Poly Haven, CC0)
- with --connectome: the MaleCNS v1.0 tables (~1.1 GB, Janelia FlyEM, CC-BY 4.0),
  then builds the brain graph (data/malecns/graph_v1.npz)

Usage: .venv/bin/python scripts/fetch_assets.py [--connectome]
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MENAGERIE = REPO / "third_party" / "mujoco_menagerie"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "367e3d9"  # tested 2026-09-22
MODELS = ["unitree_g1", "unitree_go2", "skydio_x2", "flybody"]
HDRI_URL = ("https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/"
            "kloofendal_48d_partly_cloudy_puresky_2k.hdr")
HDRI = REPO / "studio" / "web" / "public" / "hdri" / "sky_2k.hdr"
MALECNS_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
MALECNS_FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch_menagerie() -> None:
    if not (MENAGERIE / ".git").exists():
        MENAGERIE.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", MENAGERIE_URL, str(MENAGERIE))
    run("git", "sparse-checkout", "set", *MODELS, cwd=MENAGERIE)
    run("git", "fetch", "--depth", "1", "origin", "main", cwd=MENAGERIE)
    head = subprocess.run(["git", "rev-parse", "--short", "FETCH_HEAD"], cwd=MENAGERIE,
                          capture_output=True, text=True, check=True).stdout.strip()
    run("git", "checkout", "-q", "FETCH_HEAD", cwd=MENAGERIE)
    if not head.startswith(MENAGERIE_COMMIT):
        print(f"note: menagerie main is at {head}; this project was tested at {MENAGERIE_COMMIT}")


def fetch_hdri() -> None:
    if HDRI.exists():
        print(f"{HDRI.relative_to(REPO)} already present")
        return
    HDRI.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {HDRI_URL}")
    urllib.request.urlretrieve(HDRI_URL, HDRI)


def fetch_connectome() -> None:
    sys.path.insert(0, str(REPO))
    from brain.graph import CACHE, DATA, build
    DATA.mkdir(parents=True, exist_ok=True)
    for name in MALECNS_FILES:
        dest = DATA / name
        if dest.exists():
            print(f"{dest.relative_to(REPO)} already present")
            continue
        print(f"downloading {name}")
        urllib.request.urlretrieve(MALECNS_URL + name, dest)
    if not CACHE.exists():
        build()


if __name__ == "__main__":
    fetch_menagerie()
    fetch_hdri()
    if "--connectome" in sys.argv:
        fetch_connectome()
    missing = [m for m in MODELS if not (MENAGERIE / m).exists()]
    if missing:
        sys.exit(f"missing models: {missing}")
    print("assets ready")
