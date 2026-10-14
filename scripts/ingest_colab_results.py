#!/usr/bin/env python3
"""Ingest, unpack, and analyze Google Colab training outputs for bio-bot.

Usage:
    .venv/bin/python scripts/ingest_colab_results.py [--zip PATH] [--target-dir PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def find_zip_source() -> Path | None:
    downloads = Path.home() / "Downloads"
    candidates = list(downloads.glob("*colab*round15*.zip")) + list(downloads.glob("bio_bot_colab*.zip"))
    if candidates:
        # Return most recently modified zip
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def format_table(practice_data: list[dict]) -> str:
    headers = [p.get("when", f"eval {i}") for i, p in enumerate(practice_data)]
    cols = ["metric"] + headers

    def get_val(item, path, fmt="{:.2f}", scale=1.0):
        cur = item
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return "nan"
            cur = cur[k]
        try:
            return fmt.format(float(cur) * scale)
        except (ValueError, TypeError):
            return str(cur)

    metrics = [
        ("Owner arrival success ('come')", ["park_come", "got_there"], "{:.2%}", 1.0),
        ("Owner distance drift (start -> end, m)", None, "drift_come", 1.0),
        ("Owner aligned in final second", ["park_come", "aligned_last_s"], "{:.2%}", 1.0),
        ("Owner final facing cosine", ["park_come", "facing_end_cos"], "{:+.4f}", 1.0),
        ("Owner task falls / 13 s", ["park_come", "falls"], "{:.4f}", 1.0),
        ("Ball arrival success ('ball')", ["park_ball", "got_there"], "{:.2%}", 1.0),
        ("Ball distance drift (start -> end, m)", None, "drift_ball", 1.0),
        ("Ball aligned in final second", ["park_ball", "aligned_last_s"], "{:.2%}", 1.0),
        ("Ball final facing cosine", ["park_ball", "facing_end_cos"], "{:+.4f}", 1.0),
        ("Ball task falls / 13 s", ["park_ball", "falls"], "{:.4f}", 1.0),
        ("Mean seven movement falls / 13 s", None, "mean_falls", 1.0),
        ("Walk speed, m/s", ["walk", "walks_from_rest"], "{:+.4f}", 1.0),
        ("Pace speed, m/s", ["pace", "walks_from_rest"], "{:+.4f}", 1.0),
        ("Canter speed, m/s", ["canter", "walks_from_rest"], "{:+.4f}", 1.0),
        ("Run speed, m/s", ["run", "walks_from_rest"], "{:+.4f}", 1.0),
        ("Turn-left rate, rad/s", ["turn-left", "turn_rate"], "{:+.4f}", 1.0),
        ("Turn-right rate, rad/s", ["turn-right", "turn_rate"], "{:+.4f}", 1.0),
        ("Back speed, m/s", ["back", "walks_from_rest"], "{:+.4f}", 1.0),
        ("Sit pose pitch / trunk height", None, "sit_pose", 1.0),
        ("Lie-down pose pitch / trunk height", None, "lie_pose", 1.0),
        ("Ball-ahead balanced accuracy", ["park_perception", "ball-ahead", "balanced_accuracy"], "{:.3f}", 1.0),
        ("Owner-ahead balanced accuracy", ["park_perception", "owner-ahead", "balanced_accuracy"], "{:.3f}", 1.0),
        ("Object dodge reaction before hit", ["park_dodge", "reacted_before_arrival"], "{:.2%}", 1.0),
    ]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---:"] * len(cols)) + "|")

    for name, path, fmt, scale in metrics:
        row = [f"**{name}**" if "arrival" in name or "falls" in name else name]
        for item in practice_data:
            if fmt == "drift_come":
                s = get_val(item, ["park_come", "start_m"], "{:.3f}")
                e = get_val(item, ["park_come", "end_m"], "{:.3f}")
                try:
                    diff = (float(e) - float(s)) * 100
                    row.append(f"{s} -> {e} ({diff:+.1f} cm)")
                except ValueError:
                    row.append(f"{s} -> {e}")
            elif fmt == "drift_ball":
                s = get_val(item, ["park_ball", "start_m"], "{:.3f}")
                e = get_val(item, ["park_ball", "end_m"], "{:.3f}")
                try:
                    diff = (float(e) - float(s)) * 100
                    row.append(f"{s} -> {e} ({diff:+.1f} cm)")
                except ValueError:
                    row.append(f"{s} -> {e}")
            elif fmt == "mean_falls":
                g_falls = []
                for g in ("walk", "pace", "canter", "run", "turn-left", "turn-right", "back"):
                    val = item.get(g, {}).get("falls")
                    if val is not None:
                        g_falls.append(float(val))
                if g_falls:
                    row.append(f"{sum(g_falls) / len(g_falls):.4f}")
                else:
                    row.append("nan")
            elif fmt == "sit_pose":
                p = get_val(item, ["sit", "pitch_in_pose"], "{:+.2f}")
                h = get_val(item, ["sit", "height_in_pose"], "{:.2f}")
                row.append(f"{p} rad / {h}")
            elif fmt == "lie_pose":
                p = get_val(item, ["lie-down", "pitch_in_pose"], "{:+.2f}")
                h = get_val(item, ["lie-down", "height_in_pose"], "{:.2f}")
                row.append(f"{p} rad / {h}")
            else:
                row.append(get_val(item, path, fmt, scale))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=None, help="Path to downloaded colab results zip")
    parser.add_argument("--target-dir", type=Path, default=REPO / "results" / "baby_dog_vision_approach_round15",
                        help="Target directory to extract and verify results")
    args = parser.parse_args()

    zip_path = args.zip or find_zip_source()
    target_dir = args.target_dir

    if zip_path and zip_path.is_file():
        print(f"Found zip: {zip_path}")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        print(f"Extracted to: {target_dir}")
        # Flatten if nested under results/colab_round15/
        nested = target_dir / "results" / "colab_round15"
        if nested.exists():
            for p in nested.iterdir():
                shutil.move(str(p), str(target_dir / p.name))
            shutil.rmtree(target_dir / "results")

    if not target_dir.exists():
        sys.exit(f"Target directory {target_dir} does not exist and no zip file found.")

    practice_file = target_dir / "practice.json"
    if not practice_file.is_file():
        sys.exit(f"practice.json not found in {target_dir}")

    practice_data = json.loads(practice_file.read_text())
    print("\n" + "=" * 80)
    print("VERIFIED EVALUATION METRICS TABLE:")
    print("=" * 80)
    table = format_table(practice_data)
    print(table)
    print("=" * 80)

    # Checkpoint hashes
    ckpts = sorted(target_dir.glob("*.npz"))
    if ckpts:
        print("\nCHECKPOINT SHA256 HASHES:")
        for cp in ckpts:
            h = hashlib.sha256(cp.read_bytes()).hexdigest()
            print(f"  {cp.name}: {h}")


if __name__ == "__main__":
    main()
