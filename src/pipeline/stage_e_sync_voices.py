#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.lib.config import ROLE_BATCHES_DIR, VOICE_PROFILES_PATH


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def role_key_from_name(role_name: str) -> str:
    return re.sub(r"\s+", "_", role_name).strip("_") or "unknown_role"


def build_name_to_key(roles_dict):
    name_to_key = {}
    for k, v in roles_dict.items():
        if not isinstance(v, dict):
            continue
        rn = v.get("role")
        if isinstance(rn, str) and rn:
            name_to_key[rn] = k
    return name_to_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role_batches_dir", default=str(ROLE_BATCHES_DIR))
    ap.add_argument("--voice_profiles", default=str(VOICE_PROFILES_PATH))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    batches_dir = Path(args.role_batches_dir)
    vp_path = Path(args.voice_profiles)

    vp = load_json(vp_path, {"roles": {}})
    roles = vp.get("roles", {})
    if not isinstance(roles, dict):
        raise SystemExit("voice_profiles roles invalid")

    name_to_key = build_name_to_key(roles)
    csv_files = sorted(batches_dir.glob("voice_pick_*.csv"))
    if not csv_files:
        raise SystemExit(f"No voice_pick_*.csv found in {batches_dir}")

    applied_design = 0
    applied_clone = 0
    skipped_empty = 0
    skipped_conflict = 0
    skipped_invalid_clone = 0
    skipped_not_found = 0

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role = (row.get("role") or "").strip()
                if not role:
                    skipped_not_found += 1
                    continue

                design_prompt = (row.get("design_prompt") or "").strip()
                design_lang = (row.get("design_language") or "zh").strip() or "zh"

                clone_audio = (row.get("clone_ref_audio_path") or "").strip()
                clone_text = (row.get("clone_ref_text") or "").strip()
                clone_lang = (row.get("clone_language") or "zh").strip() or "zh"

                has_design = bool(design_prompt)
                has_clone = bool(clone_audio or clone_text)

                if (not has_design) and (not has_clone):
                    skipped_empty += 1
                    continue
                if has_design and has_clone:
                    skipped_conflict += 1
                    continue

                role_key = name_to_key.get(role)
                if role_key is None:
                    # fallback to generated key if role names were not indexed
                    fallback = role_key_from_name(role)
                    role_key = fallback if fallback in roles else None
                if role_key is None:
                    skipped_not_found += 1
                    continue

                ent = roles.get(role_key, {})
                if not isinstance(ent, dict):
                    ent = {}

                if has_design:
                    design_id = f"design_{role_key}"
                    ent["voice_status"] = "assigned"
                    ent["voice_source"] = "design"
                    ent["voice_id"] = design_id
                    ent["design_prompt"] = design_prompt
                    ent["design_language"] = design_lang
                    applied_design += 1
                else:
                    # clone path selected; require both audio and text
                    if not clone_audio or not clone_text:
                        skipped_invalid_clone += 1
                        continue
                    clone_id = f"clone_{role_key}"
                    ent["voice_status"] = "assigned"
                    ent["voice_source"] = "clone"
                    ent["voice_id"] = clone_id
                    ent["clone_ref_audio_path"] = clone_audio
                    ent["clone_ref_text"] = clone_text
                    ent["clone_language"] = clone_lang
                    applied_clone += 1

                roles[role_key] = ent

    vp["roles"] = roles
    if not args.dry_run:
        vp_path.write_text(json.dumps(vp, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "Sync done. "
        f"applied_design={applied_design} "
        f"applied_clone={applied_clone} "
        f"skipped_empty={skipped_empty} "
        f"skipped_conflict={skipped_conflict} "
        f"skipped_invalid_clone={skipped_invalid_clone} "
        f"skipped_not_found={skipped_not_found} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()

