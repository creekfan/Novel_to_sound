#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lib.config import VOICE_PROFILES_PATH, ROLE_BATCHES_DIR


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice_profiles", default=str(VOICE_PROFILES_PATH))
    ap.add_argument("--role_batches_dir", default=str(ROLE_BATCHES_DIR))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    vp_path = Path(args.voice_profiles)
    out_dir = Path(args.role_batches_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vp = load_json(vp_path, {"roles": {}})
    roles = vp.get("roles", {})
    if not isinstance(roles, dict):
        roles = {}

    design_rows = []
    clone_rows = []

    design_counter = 1
    clone_counter = 1

    for role_key, entry in roles.items():
        if not isinstance(entry, dict):
            continue
        role_name = str(entry.get("role", role_key))
        status = str(entry.get("voice_status", "unassigned"))
        suggested_source = str(entry.get("suggested_voice_source", ""))
        suggested_voice = str(entry.get("suggested_voice", ""))

        # Pre-create design requests only for unassigned roles that are suggested as voice_design.
        if status == "unassigned" and suggested_source == "voice_design":
            design_id = f"design_{design_counter:04d}"
            design_counter += 1
            design_rows.append(
                {
                    "design_id": design_id,
                    "role": role_name,
                    "role_key": role_key,
                    "language": "zh",
                    "prompt_text": suggested_voice or "neutral clear narrator-like voice",
                    "status": "pending",
                    "sample_wav": "",
                    "notes": "",
                }
            )

        # Add a clone template row for each role only when user explicitly wants clone later.
        # Keep file lightweight by not pre-filling all roles.

    # Keep one guidance row in clone sheet for easier user operation.
    clone_rows.append(
        {
            "clone_id": f"clone_{clone_counter:04d}",
            "role": "",
            "role_key": "",
            "ref_audio_path": "",
            "ref_text": "",
            "language": "zh",
            "status": "pending",
            "cache_prompt_path": "",
            "notes": "fill this row or append more rows when user wants cloning",
        }
    )

    design_path = out_dir / "voice_design_requests.csv"
    clone_path = out_dir / "voice_clone_requests.csv"

    if args.overwrite or not design_path.exists():
        write_csv(
            design_path,
            design_rows,
            [
                "design_id",
                "role",
                "role_key",
                "language",
                "prompt_text",
                "status",
                "sample_wav",
                "notes",
            ],
        )

    if args.overwrite or not clone_path.exists():
        write_csv(
            clone_path,
            clone_rows,
            [
                "clone_id",
                "role",
                "role_key",
                "ref_audio_path",
                "ref_text",
                "language",
                "status",
                "cache_prompt_path",
                "notes",
            ],
        )

    print(
        f"Prepared templates. design_rows={len(design_rows)} "
        f"clone_rows={len(clone_rows)} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()

