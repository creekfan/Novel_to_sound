#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.lib.config import QWEN3_TTS_ROOT

sys.path.insert(0, str(QWEN3_TTS_ROOT))
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

from src.lib.config import CHAPTERS_JSON_DIR, ROLE_BATCHES_DIR, OUTPUT_DIR


CHAPTER_RX = re.compile(r"^(\d+)_.*\.json$", re.I)


def safe_name(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    s = re.sub(r"\s+", "_", s)
    return s or "unknown"


def map_language(lang: Optional[str], default_lang: str) -> str:
    x = (lang or "").strip().lower()
    if not x:
        return default_lang
    if x in {"zh", "zh-cn", "cn", "chinese"}:
        return "Chinese"
    if x in {"en", "en-us", "english"}:
        return "English"
    if x in {"auto"}:
        return "Auto"
    return lang.strip()


def normalize_gender(value: Optional[str]) -> str:
    x = (value or "").strip().lower()
    if x in {"男", "male", "m", "man", "boy"}:
        return "男"
    if x in {"女", "female", "f", "woman", "girl"}:
        return "女"
    return "中性"


def build_gender_lock_text(gender: str) -> str:
    if gender == "男":
        return "性别锁定：男声。必须使用自然稳定的男性音色，禁止女性音色、女声共振和女声气质。"
    if gender == "女":
        return "性别锁定：女声。必须使用自然稳定的女性音色，禁止男性音色、男声共振和男声气质。"
    return "性别锁定：中性。保持自然中性音色，不偏男性或女性极端特征。"


def build_design_instruct(role_cfg: Dict[str, Any], enable_gender_lock: bool) -> str:
    parts: List[str] = []
    gender = normalize_gender(role_cfg.get("dominant_gender"))
    feature = (role_cfg.get("dominant_feature") or "").strip()
    prompt = (role_cfg.get("design_prompt") or "").strip()

    if enable_gender_lock:
        parts.append(build_gender_lock_text(gender))
    if feature:
        parts.append(f"角色特征：{feature}。")
    if prompt:
        parts.append(prompt)
    else:
        parts.append("保持角色说话风格稳定、吐字清晰。")
    return " ".join(parts).strip()


def chapter_num_from_name(name: str) -> Optional[int]:
    m = CHAPTER_RX.match(name)
    if not m:
        return None
    return int(m.group(1))


def build_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    hh = ms // 3600000
    mm = (ms % 3600000) // 60000
    ss = (ms % 60000) // 1000
    ms2 = ms % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms2:03d}"


def write_srt(path: Path, items: List[Dict[str, Any]]) -> None:
    out: List[str] = []
    for i, x in enumerate(items, start=1):
        out.append(str(i))
        out.append(f"{build_ts(x['start'])} --> {build_ts(x['end'])}")
        out.append(f"{x['role']}: {x['text']}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def to_dtype(dtype_name: str, device: str) -> torch.dtype:
    x = dtype_name.strip().lower()
    if x == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if x == "bfloat16":
        return torch.bfloat16
    if x == "float16":
        return torch.float16
    return torch.float32


def pick_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def make_signature(role_cfg: Dict[str, Any]) -> str:
    clone_abs = role_cfg.get("clone_ref_audio_abs", "")
    clone_stat = role_cfg.get("clone_ref_audio_stat", "")
    parts = [
        role_cfg.get("source", ""),
        role_cfg.get("dominant_gender", ""),
        role_cfg.get("dominant_feature", ""),
        role_cfg.get("design_prompt", ""),
        role_cfg.get("design_language", ""),
        role_cfg.get("clone_ref_audio_path", ""),
        clone_abs,
        clone_stat,
        role_cfg.get("clone_ref_text", ""),
        role_cfg.get("clone_language", ""),
    ]
    text = "||".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_prompt_payload(path: Path) -> Tuple[List[VoiceClonePromptItem], Dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(f"Invalid prompt payload: {path}")

    items_out: List[VoiceClonePromptItem] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        ref_code = item.get("ref_code")
        ref_spk = item.get("ref_spk_embedding")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items_out.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(item.get("x_vector_only_mode", False)),
                icl_mode=bool(item.get("icl_mode", True)),
                ref_text=item.get("ref_text"),
            )
        )

    if not items_out:
        raise ValueError(f"Empty prompt payload: {path}")

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return items_out, meta


def save_prompt_payload(path: Path, items: List[VoiceClonePromptItem], meta: Dict[str, Any]) -> None:
    payload_items: List[Dict[str, Any]] = []
    for it in items:
        d = asdict(it)
        if d.get("ref_code") is not None and torch.is_tensor(d["ref_code"]):
            d["ref_code"] = d["ref_code"].detach().cpu()
        if torch.is_tensor(d.get("ref_spk_embedding")):
            d["ref_spk_embedding"] = d["ref_spk_embedding"].detach().cpu()
        payload_items.append(d)
    torch.save({"items": payload_items, "meta": meta}, path)


def resolve_clone_audio(raw: str, csv_dir: Path) -> Tuple[str, str, str]:
    ref = raw.strip()
    if not ref:
        return "", "", ""

    p = Path(ref)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((csv_dir / p).resolve())
        candidates.append(p)

    for c in candidates:
        try:
            if c.exists():
                st = c.stat()
                stat = f"{st.st_size}:{int(st.st_mtime)}"
                return ref, str(c.resolve()), stat
        except OSError:
            continue
    return ref, "", ""


def load_voice_pick(csv_path: Path, default_lang: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    roles: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "role" not in reader.fieldnames:
            raise ValueError("voice_pick csv missing required column: role")

        for row in reader:
            role = (row.get("role") or "").strip()
            if not role:
                continue

            design_prompt = (row.get("design_prompt") or "").strip()
            design_language = map_language(row.get("design_language"), default_lang)
            dominant_gender = normalize_gender(row.get("dominant_gender") or row.get("gender"))
            dominant_feature = (row.get("dominant_feature") or row.get("feature") or "").strip()
            clone_ref_audio_path = (row.get("clone_ref_audio_path") or "").strip()
            clone_ref_text = (row.get("clone_ref_text") or "").strip()
            clone_language = map_language(row.get("clone_language"), default_lang)

            has_design = bool(design_prompt)
            has_clone_any = bool(clone_ref_audio_path or clone_ref_text)
            has_clone_full = bool(clone_ref_audio_path and clone_ref_text)

            source = "unset"
            if has_design and has_clone_any:
                source = "conflict"
                warnings.append(f"[skip][conflict] role={role} has both design and clone fields.")
            elif has_design:
                source = "design"
            elif has_clone_any:
                if has_clone_full:
                    source = "clone"
                else:
                    source = "invalid_clone"
                    warnings.append(f"[skip][invalid_clone] role={role} clone needs both audio path and text.")

            clone_raw, clone_abs, clone_stat = resolve_clone_audio(clone_ref_audio_path, csv_path.parent)
            if source == "clone" and not clone_abs:
                warnings.append(
                    f"[warn][clone_ref_missing] role={role} audio not found now: {clone_ref_audio_path} (will pass raw path)."
                )

            role_cfg = {
                "role": role,
                "source": source,
                "dominant_gender": dominant_gender,
                "dominant_feature": dominant_feature,
                "design_prompt": design_prompt,
                "design_language": design_language,
                "clone_ref_audio_path": clone_raw or clone_ref_audio_path,
                "clone_ref_audio_abs": clone_abs,
                "clone_ref_audio_stat": clone_stat,
                "clone_ref_text": clone_ref_text,
                "clone_language": clone_language,
            }
            role_cfg["signature"] = make_signature(role_cfg)
            roles[role] = role_cfg

    return roles, warnings


def iter_chapters(chapter_dir: Path, chapter_start: int, chapter_end: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for p in chapter_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        if p.name == "progress.json":
            continue
        n = chapter_num_from_name(p.name)
        if n is None or not (chapter_start <= n <= chapter_end):
            continue
        out.append({"num": n, "path": p})

    out.sort(key=lambda x: x["num"])
    return out


def load_chapter_lines(chapter_path: Path, default_lang: str) -> Tuple[str, List[Dict[str, Any]]]:
    data = json.loads(chapter_path.read_text(encoding="utf-8"))
    chapter_title = str(data.get("chapter_title") or chapter_path.stem)
    raw_lines = data.get("lines") or []

    lines_out: List[Dict[str, Any]] = []
    if not isinstance(raw_lines, list):
        return chapter_title, lines_out

    for line in raw_lines:
        if not isinstance(line, dict):
            continue
        role = str(line.get("role") or line.get("speaker") or "旁白").strip() or "旁白"
        text = str(line.get("dialogue") or line.get("text") or "").strip()
        if not text:
            continue
        language = map_language(line.get("language"), default_lang)
        lines_out.append({"role": role, "text": text, "language": language})
    return chapter_title, lines_out


def choose_fallback_role(roles_cfg: Dict[str, Dict[str, Any]]) -> str:
    valid = [r for r, cfg in roles_cfg.items() if cfg.get("source") in {"design", "clone"}]
    if not valid:
        raise ValueError("No valid role voice config in csv (need design_prompt OR full clone fields).")

    for key in ("旁白", "narrator", "Narrator"):
        if key in valid:
            return key
    return valid[0]


def ensure_role_prompt(
    role_cfg: Dict[str, Any],
    role_prompt_path: Path,
    role_ref_wav_path: Path,
    base_model: Qwen3TTSModel,
    design_model_ref: Dict[str, Optional[Qwen3TTSModel]],
    design_model_name: str,
    device: str,
    dtype: torch.dtype,
    regen_prompts: bool,
    ref_text_template: str,
    design_temperature: float,
    design_top_p: float,
    enable_gender_lock: bool,
) -> Tuple[Optional[List[VoiceClonePromptItem]], str]:
    source = role_cfg.get("source")
    role = role_cfg.get("role", "")
    signature = role_cfg.get("signature", "")
    expected_meta = {"role": role, "source": source, "signature": signature}

    if role_prompt_path.exists() and not regen_prompts:
        try:
            cached_items, cached_meta = load_prompt_payload(role_prompt_path)
            if (
                cached_meta.get("role") == expected_meta["role"]
                and cached_meta.get("source") == expected_meta["source"]
                and cached_meta.get("signature") == expected_meta["signature"]
            ):
                return cached_items, "cache"
        except Exception:
            pass

    if source == "design":
        if design_model_ref.get("model") is None:
            design_model_ref["model"] = Qwen3TTSModel.from_pretrained(
                design_model_name,
                device_map=device,
                dtype=dtype,
            )
        design_model = design_model_ref["model"]
        if design_model is None:
            return None, "failed"

        ref_text = ref_text_template.format(
            role=role,
            gender=role_cfg.get("dominant_gender", "中性"),
            feature=role_cfg.get("dominant_feature", ""),
        )
        design_instruct = build_design_instruct(role_cfg, enable_gender_lock=enable_gender_lock)
        wavs, sr = design_model.generate_voice_design(
            text=ref_text,
            language=role_cfg.get("design_language", "Chinese"),
            instruct=design_instruct,
            non_streaming_mode=True,
            temperature=design_temperature,
            top_p=design_top_p,
        )
        ref_wav = wavs[0]
        sf.write(str(role_ref_wav_path), ref_wav, sr)
        prompt_items = base_model.create_voice_clone_prompt(
            ref_audio=(ref_wav, sr),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        save_prompt_payload(
            role_prompt_path,
            prompt_items,
            {
                **expected_meta,
                "language": role_cfg.get("design_language", "Chinese"),
                "design_prompt": role_cfg.get("design_prompt", ""),
                "design_instruct": design_instruct,
                "ref_text": ref_text,
            },
        )
        return prompt_items, "generated_design"

    if source == "clone":
        clone_audio = role_cfg.get("clone_ref_audio_abs") or role_cfg.get("clone_ref_audio_path")
        clone_text = role_cfg.get("clone_ref_text")
        if not clone_audio or not clone_text:
            return None, "invalid_clone"
        prompt_items = base_model.create_voice_clone_prompt(
            ref_audio=clone_audio,
            ref_text=clone_text,
            x_vector_only_mode=False,
        )
        save_prompt_payload(
            role_prompt_path,
            prompt_items,
            {
                **expected_meta,
                "clone_ref_audio_path": role_cfg.get("clone_ref_audio_path"),
                "clone_ref_audio_abs": role_cfg.get("clone_ref_audio_abs"),
                "clone_ref_text": clone_text,
                "clone_language": role_cfg.get("clone_language", "Chinese"),
            },
        )
        return prompt_items, "generated_clone"

    return None, "skipped"


def synthesize_chapter(
    chapter_name: str,
    lines: List[Dict[str, Any]],
    role_prompts: Dict[str, List[VoiceClonePromptItem]],
    role_cfg_map: Dict[str, Dict[str, Any]],
    fallback_role: str,
    base_model: Qwen3TTSModel,
    out_wav: Path,
    out_srt: Path,
    out_segments_json: Path,
    gap_ms: int,
    default_lang: str,
    temperature: float,
    top_p: float,
    save_line_wavs: bool,
    batch_size: int,
    fail_on_missing_role: bool,
) -> Dict[str, Any]:
    chunks: List[np.ndarray] = []
    srt_items: List[Dict[str, Any]] = []
    sample_rate: Optional[int] = None
    t = 0.0
    fallback_hits = 0

    line_dir = out_wav.parent / "lines"
    if save_line_wavs:
        line_dir.mkdir(parents=True, exist_ok=True)

    total = len(lines)
    batch_size = max(1, int(batch_size))
    idx_base = 1
    for s in range(0, total, batch_size):
        batch = lines[s : s + batch_size]
        batch_texts: List[str] = []
        batch_langs: List[str] = []
        batch_prompts: List[VoiceClonePromptItem] = []
        batch_meta: List[Tuple[str, str]] = []

        for line in batch:
            role = line["role"]
            text = line["text"]
            prompt_items = role_prompts.get(role)
            role_cfg = role_cfg_map.get(role)

            if prompt_items is None:
                if fail_on_missing_role:
                    raise ValueError(f"Missing voice config for role={role} in chapter={chapter_name}")
                prompt_items = role_prompts[fallback_role]
                role_cfg = role_cfg_map.get(fallback_role)
                fallback_hits += 1

            role_default_lang = default_lang
            if role_cfg:
                if role_cfg.get("source") == "design":
                    role_default_lang = role_cfg.get("design_language", default_lang)
                elif role_cfg.get("source") == "clone":
                    role_default_lang = role_cfg.get("clone_language", default_lang)
            language = map_language(line.get("language"), role_default_lang)

            batch_texts.append(text)
            batch_langs.append(language)
            # create_voice_clone_prompt currently produces 1 prompt item for a role reference.
            batch_prompts.append(prompt_items[0])
            batch_meta.append((role, text))

        wavs, sr = base_model.generate_voice_clone(
            text=batch_texts,
            language=batch_langs,
            voice_clone_prompt=batch_prompts,
            non_streaming_mode=True,
            temperature=temperature,
            top_p=top_p,
        )

        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError(f"Sample rate mismatch in chapter={chapter_name}: {sample_rate} vs {sr}")

        for j, wav in enumerate(wavs):
            role, text = batch_meta[j]
            wav = wav.astype(np.float32)
            idx = idx_base + j

            start = t
            duration = len(wav) / float(sample_rate)
            end = start + duration
            srt_items.append(
                {
                    "index": idx,
                    "role": role,
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )

            if save_line_wavs:
                line_path = line_dir / f"{idx:04d}_{safe_name(role)}.wav"
                sf.write(str(line_path), wav, sample_rate)

            chunks.append(wav)
            gap = np.zeros(int(sample_rate * gap_ms / 1000.0), dtype=np.float32)
            chunks.append(gap)
            t = end + (gap_ms / 1000.0)

        idx_base += len(batch)

    if sample_rate is None:
        raise ValueError(f"No valid lines in chapter: {chapter_name}")

    merged = np.concatenate(chunks, axis=0) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(str(out_wav), merged, sample_rate)
    write_srt(out_srt, srt_items)
    out_segments_json.write_text(
        json.dumps(
            {
                "chapter": chapter_name,
                "sample_rate": sample_rate,
                "line_count": len(srt_items),
                "segments": srt_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "line_count": len(srt_items),
        "sample_rate": sample_rate,
        "duration_sec": round(len(merged) / float(sample_rate), 3),
        "fallback_hits": fallback_hits,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate chapter-level TTS audio + subtitles from voice_pick csv (design/clone schema)."
    )
    p.add_argument("--voice-pick-csv", default=str(ROLE_BATCHES_DIR / "voice_pick_0001_0100.csv"))
    p.add_argument("--chapters-dir", default=str(CHAPTERS_JSON_DIR))
    p.add_argument("--chapter-start", type=int, default=1)
    p.add_argument("--chapter-end", type=int, default=100)
    p.add_argument("--out-dir", default=str(OUTPUT_DIR / "tts_outputs_0001_0100"))
    p.add_argument("--design-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    p.add_argument("--base-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--default-language", default="Chinese")
    p.add_argument("--ref-text-template", default="{role}，这是{gender}角色音色参考文本，用于整章复用。")
    p.add_argument("--gap-ms", type=int, default=250)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--design-temperature", type=float, default=0.55, help="Sampling temperature for voice design stage.")
    p.add_argument("--design-top_p", type=float, default=0.9, help="Sampling top_p for voice design stage.")
    p.add_argument("--disable-gender-lock", action="store_true", help="Disable strict gender lock for design roles.")
    p.add_argument("--batch-size", type=int, default=4, help="Lines per generation call inside one chapter.")
    p.add_argument("--fail-on-missing-role", action="store_true", help="Fail instead of fallback when a role has no voice config.")
    p.add_argument("--regen-prompts", action="store_true")
    p.add_argument("--save-line-wavs", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing chapter wav/srt files.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    csv_path = Path(args.voice_pick_csv).resolve()
    chapters_dir = Path(args.chapters_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    prompts_dir = out_dir / "voice_prompts"
    refs_dir = out_dir / "voice_refs"
    chapters_out = out_dir / "chapters"
    for d in (prompts_dir, refs_dir, chapters_out):
        d.mkdir(parents=True, exist_ok=True)

    roles_cfg, warnings = load_voice_pick(csv_path, args.default_language)
    fallback_role = choose_fallback_role(roles_cfg)
    chapters = iter_chapters(chapters_dir, args.chapter_start, args.chapter_end)
    if not chapters:
        raise SystemExit("No chapter json files found in requested range.")

    chapter_payloads: List[Dict[str, Any]] = []
    roles_needed: set = set()
    for ch in chapters:
        chapter_title, lines = load_chapter_lines(ch["path"], args.default_language)
        if not lines:
            continue
        for line in lines:
            roles_needed.add(line["role"])
        chapter_payloads.append(
            {
                "num": ch["num"],
                "path": ch["path"],
                "chapter_title": chapter_title,
                "lines": lines,
            }
        )
    if not chapter_payloads:
        raise SystemExit("No valid chapter lines found in requested range.")

    roles_to_prepare = {
        role
        for role in roles_needed
        if role in roles_cfg and roles_cfg[role].get("source") in {"design", "clone"}
    }
    roles_to_prepare.add(fallback_role)
    missing_roles = sorted(
        role
        for role in roles_needed
        if role not in roles_cfg or roles_cfg[role].get("source") not in {"design", "clone"}
    )

    for w in warnings:
        print(w)
    print(
        f"[Info] csv_roles={len(roles_cfg)} fallback_role={fallback_role} "
        f"chapters={len(chapter_payloads)} range={args.chapter_start}-{args.chapter_end} "
        f"roles_in_chapters={len(roles_needed)} roles_to_prepare={len(roles_to_prepare)} "
        f"missing_roles={len(missing_roles)}"
    )
    if missing_roles:
        preview = ", ".join(missing_roles[:15])
        if len(missing_roles) > 15:
            preview += ", ..."
        print(f"[Warn] roles without valid voice config (will fallback unless --fail-on-missing-role): {preview}")

    if args.dry_run:
        print("[Info] dry_run=True, no model loading and no synthesis executed.")
        return

    device = pick_device(args.device)
    dtype = to_dtype(args.dtype, device)
    print(f"[Info] device={device} dtype={dtype}")
    print("[Info] loading base model...")
    base_model = Qwen3TTSModel.from_pretrained(
        args.base_model,
        device_map=device,
        dtype=dtype,
    )
    design_model_ref: Dict[str, Optional[Qwen3TTSModel]] = {"model": None}

    role_prompts: Dict[str, List[VoiceClonePromptItem]] = {}
    for role in sorted(roles_to_prepare):
        cfg = roles_cfg.get(role)
        if cfg is None:
            continue
        if cfg.get("source") not in {"design", "clone"}:
            continue
        role_file = safe_name(role)
        role_prompt_path = prompts_dir / f"{role_file}.pt"
        role_ref_wav_path = refs_dir / f"{role_file}.wav"
        items, status = ensure_role_prompt(
            role_cfg=cfg,
            role_prompt_path=role_prompt_path,
            role_ref_wav_path=role_ref_wav_path,
            base_model=base_model,
            design_model_ref=design_model_ref,
            design_model_name=args.design_model,
            device=device,
            dtype=dtype,
            regen_prompts=bool(args.regen_prompts),
            ref_text_template=args.ref_text_template,
            design_temperature=args.design_temperature,
            design_top_p=args.design_top_p,
            enable_gender_lock=not bool(args.disable_gender_lock),
        )
        print(f"[Role] {role} -> {status}")
        if items is not None:
            role_prompts[role] = items

    if fallback_role not in role_prompts:
        if role_prompts:
            fallback_role = next(iter(role_prompts.keys()))
            print(f"[Warn] configured fallback not ready, switched to: {fallback_role}")
        else:
            raise SystemExit("No role prompts available. Check csv config or clone refs.")

    manifest: List[Dict[str, Any]] = []
    for ch in chapter_payloads:
        n = ch["num"]
        p = ch["path"]
        chapter_title = ch["chapter_title"]
        lines = ch["lines"]

        chapter_key = f"{n:04d}_{safe_name(chapter_title)}"
        chapter_folder = chapters_out / chapter_key
        chapter_folder.mkdir(parents=True, exist_ok=True)
        out_wav = chapter_folder / f"{chapter_key}.wav"
        out_srt = chapter_folder / f"{chapter_key}.srt"
        out_segments_json = chapter_folder / f"{chapter_key}.segments.json"
        if (not args.overwrite) and out_wav.exists() and out_srt.exists() and out_segments_json.exists():
            print(f"[Skip] already exists: {chapter_key}")
            manifest.append(
                {
                    "chapter_num": n,
                    "chapter_title": chapter_title,
                    "source_json": str(p),
                    "wav": str(out_wav),
                    "srt": str(out_srt),
                    "segments_json": str(out_segments_json),
                    "skipped_existing": True,
                }
            )
            continue

        print(f"[Chapter] {p.name} lines={len(lines)}")
        stats = synthesize_chapter(
            chapter_name=chapter_title,
            lines=lines,
            role_prompts=role_prompts,
            role_cfg_map=roles_cfg,
            fallback_role=fallback_role,
            base_model=base_model,
            out_wav=out_wav,
            out_srt=out_srt,
            out_segments_json=out_segments_json,
            gap_ms=args.gap_ms,
            default_lang=args.default_language,
            temperature=args.temperature,
            top_p=args.top_p,
            save_line_wavs=bool(args.save_line_wavs),
            batch_size=args.batch_size,
            fail_on_missing_role=bool(args.fail_on_missing_role),
        )
        manifest.append(
            {
                "chapter_num": n,
                "chapter_title": chapter_title,
                "source_json": str(p),
                "wav": str(out_wav),
                "srt": str(out_srt),
                "segments_json": str(out_segments_json),
                **stats,
            }
        )
        print(f"[Done] wav={out_wav.name} srt={out_srt.name}")

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "voice_pick_csv": str(csv_path),
                "chapters_dir": str(chapters_dir),
                "range": {"start": args.chapter_start, "end": args.chapter_end},
                "chapter_count": len(manifest),
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[All Done] chapters_generated={len(manifest)} output={out_dir}")


if __name__ == "__main__":
    main()
