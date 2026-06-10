#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.lib.config import CHAPTERS_JSON_DIR, ROLE_BATCHES_DIR, VOICE_PROFILES_PATH


RX = re.compile(r"^(\d+)_.*\.json$", re.I)


def chapter_num(name: str):
    m = RX.match(name)
    return int(m.group(1)) if m else None


def dominant(counter: Counter):
    return counter.most_common(1)[0][0] if counter else None


def suggest_voice(gender: str, feature: str):
    f = (feature or "").lower()
    if gender == "女":
        if any(k in f for k in ["柔", "甜", "轻", "少"]):
            return "custom_voice", "Serena"
        return "custom_voice", "Vivian"
    if gender == "男":
        if any(k in f for k in ["老", "低", "沙"]):
            return "custom_voice", "Uncle_Fu"
        if any(k in f for k in ["少", "清", "年"]):
            return "custom_voice", "Dylan"
        return "custom_voice", "Eric"
    return "voice_design", "neutral clear narrator-like voice"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(CHAPTERS_JSON_DIR))
    ap.add_argument("--out_dir", default=str(ROLE_BATCHES_DIR))
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--voice_profiles", default=str(VOICE_PROFILES_PATH))
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    vp_path = Path(args.voice_profiles)

    if not in_dir.exists():
        raise SystemExit(f"input dir not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for p in in_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == "progress.json" or p.suffix.lower() != ".json":
            continue
        n = chapter_num(p.name)
        if n is None:
            continue
        files.append((n, p))
    files.sort(key=lambda x: x[0])
    if not files:
        raise SystemExit("no chapter json found")

    buckets = defaultdict(dict)
    global_roles = defaultdict(lambda: {"chapters": set(), "g": Counter(), "f": Counter()})

    for n, p in files:
        start = ((n - 1) // args.batch_size) * args.batch_size + 1
        end = start + args.batch_size - 1
        bkey = (start, end)

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        reg = data.get("role_registry", {})
        if not isinstance(reg, dict):
            continue

        line_count = Counter()
        lines = data.get("lines", [])
        if isinstance(lines, list):
            for line in lines:
                if isinstance(line, dict):
                    role = line.get("role")
                    if isinstance(role, str) and role:
                        line_count[role] += 1

        for role, info in reg.items():
            if not isinstance(role, str) or not role:
                continue
            if not isinstance(info, dict):
                info = {}

            o = buckets[bkey].get(role)
            if o is None:
                o = {
                    "role": role,
                    "chapter_count": 0,
                    "line_count": 0,
                    "chapters": set(),
                    "g": Counter(),
                    "f": Counter(),
                }
                buckets[bkey][role] = o

            o["chapter_count"] += 1
            o["line_count"] += int(line_count.get(role, 0))
            o["chapters"].add(n)

            g = info.get("gender")
            f = info.get("feature")
            if isinstance(g, str) and g:
                o["g"][g] += 1
            if isinstance(f, str) and f:
                o["f"][f] += 1

            gr = global_roles[role]
            gr["chapters"].add(n)
            if isinstance(g, str) and g:
                gr["g"][g] += 1
            if isinstance(f, str) and f:
                gr["f"][f] += 1

    index = {"batch_size": args.batch_size, "batches": []}

    for bkey in sorted(buckets.keys()):
        start, end = bkey
        roles = []
        for role, o in buckets[bkey].items():
            dg = dominant(o["g"])
            df = dominant(o["f"])
            src, sv = suggest_voice(dg, df)
            roles.append(
                {
                    "role": role,
                    "chapter_count": o["chapter_count"],
                    "line_count": o["line_count"],
                    "chapters": sorted(o["chapters"]),
                    "dominant_gender": dg,
                    "dominant_feature": df,
                    "suggested_voice_source": src,
                    "suggested_voice": sv,
                }
            )
        roles.sort(key=lambda x: (-x["chapter_count"], -x["line_count"], x["role"]))

        json_name = f"roles_{start:04d}_{end:04d}.json"
        csv_name = f"voice_pick_{start:04d}_{end:04d}.csv"

        (out_dir / json_name).write_text(
            json.dumps(
                {"range": {"start": start, "end": end}, "role_count": len(roles), "roles": roles},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # New schema:
        # - remove suggested_* columns from CSV
        # - user fills design_* OR clone_* to choose voice source
        csv_out_name = csv_name
        csv_out_path = out_dir / csv_out_name
        try:
            with csv_out_path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "role",
                        "chapter_count",
                        "line_count",
                        "dominant_gender",
                        "dominant_feature",
                        "design_prompt",
                        "design_language",
                        "clone_ref_audio_path",
                        "clone_ref_text",
                        "clone_language",
                        "selection_notes",
                    ]
                )
                for r in roles:
                    w.writerow(
                        [
                            r["role"],
                            r["chapter_count"],
                            r["line_count"],
                            r["dominant_gender"] or "",
                            r["dominant_feature"] or "",
                            "",
                            "zh",
                            "",
                            "",
                            "zh",
                            "",
                        ]
                    )
        except PermissionError:
            csv_out_name = csv_name.replace(".csv", "_new.csv")
            csv_out_path = out_dir / csv_out_name
            with csv_out_path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "role",
                        "chapter_count",
                        "line_count",
                        "dominant_gender",
                        "dominant_feature",
                        "design_prompt",
                        "design_language",
                        "clone_ref_audio_path",
                        "clone_ref_text",
                        "clone_language",
                        "selection_notes",
                    ]
                )
                for r in roles:
                    w.writerow(
                        [
                            r["role"],
                            r["chapter_count"],
                            r["line_count"],
                            r["dominant_gender"] or "",
                            r["dominant_feature"] or "",
                            "",
                            "zh",
                            "",
                            "",
                            "zh",
                            "",
                        ]
                    )

        index["batches"].append(
            {
                "start": start,
                "end": end,
                "roles_json": json_name,
                "voice_pick_csv": csv_out_name,
                "role_count": len(roles),
            }
        )

    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    vp = {"roles": {}}
    if vp_path.exists():
        try:
            vp = json.loads(vp_path.read_text(encoding="utf-8"))
        except Exception:
            vp = {"roles": {}}
    if not isinstance(vp, dict):
        vp = {"roles": {}}
    if not isinstance(vp.get("roles"), dict):
        vp["roles"] = {}

    for role, gr in global_roles.items():
        rid = re.sub(r"\s+", "_", role).strip("_") or "unknown_role"
        dg = dominant(gr["g"])
        df = dominant(gr["f"])
        src, sv = suggest_voice(dg, df)

        ent = vp["roles"].get(rid, {})
        if not isinstance(ent, dict):
            ent = {}
        ent.setdefault("role", role)
        ent.setdefault("voice_status", "unassigned")
        ent["chapter_count"] = len(gr["chapters"])
        ent["chapters"] = sorted(gr["chapters"])
        ent["dominant_gender"] = dg
        ent["dominant_feature"] = df
        ent.setdefault("voice_source", src)
        ent.setdefault("voice_id", "")
        ent.setdefault("suggested_voice_source", src)
        ent.setdefault("suggested_voice", sv)
        ent.setdefault("locked", True)
        vp["roles"][rid] = ent

    vp_path.write_text(json.dumps(vp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. batches={len(index['batches'])} roles={len(global_roles)}")


if __name__ == "__main__":
    main()
