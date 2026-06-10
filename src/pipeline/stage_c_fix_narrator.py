#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.lib.config import CHAPTERS_JSON_DIR


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_first_narrator_index(lines):
    if not isinstance(lines, list):
        return 0
    for i, line in enumerate(lines):
        if isinstance(line, dict) and line.get("role") == "旁白":
            return i
    return 0


def main():
    ap = argparse.ArgumentParser(description="Ensure narrator exists in role_registry for chapter JSON files.")
    ap.add_argument("--in_dir", default=str(CHAPTERS_JSON_DIR))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"Input dir not found: {in_dir}")

    updated = 0
    skipped = 0
    failed = 0

    for p in sorted(in_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".json":
            continue
        if p.name == "progress.json":
            continue

        data = load_json(p)
        if not isinstance(data, dict):
            failed += 1
            continue

        lines = data.get("lines", [])
        first_idx = find_first_narrator_index(lines)

        rr = data.get("role_registry")
        if not isinstance(rr, dict):
            rr = {}
            data["role_registry"] = rr

        entry = rr.get("旁白")
        changed = False

        if not isinstance(entry, dict):
            rr["旁白"] = {
                "gender": None,
                "feature": None,
                "first_seen_line_index": first_idx,
            }
            changed = True
        else:
            # Normalize narrator entry shape to avoid future key errors.
            if "gender" not in entry:
                entry["gender"] = None
                changed = True
            if "feature" not in entry:
                entry["feature"] = None
                changed = True
            if "first_seen_line_index" not in entry:
                entry["first_seen_line_index"] = first_idx
                changed = True

        if changed:
            updated += 1
            if not args.dry_run:
                p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            skipped += 1

    print(
        f"Done. updated={updated}, skipped={skipped}, failed={failed}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()

