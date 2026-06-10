import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch


def bootstrap_repo() -> None:
    repo_root = Path(os.environ.get("QWEN3_TTS_ROOT", r"D:\Qwen3-TTS")).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


bootstrap_repo()

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


def clamp_score(v: Optional[str], default: int = 50) -> int:
    if v is None or str(v).strip() == "":
        return default
    try:
        x = int(float(v))
    except ValueError:
        return default
    return max(0, min(100, x))


def norm_gender(gender: Optional[str]) -> str:
    if gender is None:
        return "中性"
    g = str(gender).strip().lower()
    if g in {"男", "male", "m", "man"}:
        return "男"
    if g in {"女", "female", "f", "woman"}:
        return "女"
    return "中性"


def score_level(score: int, labels: Tuple[str, str, str, str, str]) -> str:
    if score <= 20:
        return labels[0]
    if score <= 40:
        return labels[1]
    if score <= 60:
        return labels[2]
    if score <= 80:
        return labels[3]
    return labels[4]


def build_voice_prompt(row: Dict[str, str]) -> str:
    gender = norm_gender(row.get("gender"))
    age = clamp_score(row.get("age"), default=50)
    pitch = clamp_score(row.get("pitch"), default=50)
    speed = clamp_score(row.get("speed"), default=50)
    brightness = clamp_score(row.get("brightness"), default=50)
    breath = clamp_score(row.get("breathiness"), default=50)
    rough = clamp_score(row.get("roughness"), default=50)
    emotion = clamp_score(row.get("emotion"), default=50)
    charm = clamp_score(row.get("charm"), default=50)
    align = clamp_score(row.get("alignment"), default=50)
    style = (row.get("style") or "").strip()

    age_text = "少年感" if age <= 33 else ("中年感" if age <= 66 else "老年感")
    align_text = "正派气质" if align <= 33 else ("中立普通气质" if align <= 66 else "反派压迫气质")

    pitch_text = score_level(pitch, ("音调很低", "音调偏低", "音调中等", "音调偏高", "音调很高"))
    speed_text = score_level(speed, ("语速很慢", "语速偏慢", "语速中等", "语速偏快", "语速很快"))
    bright_text = score_level(
        brightness, ("音色偏暗", "音色稍暗", "音色中性", "音色明亮", "音色非常明亮")
    )
    breath_text = score_level(
        breath, ("气声极少", "气声偏少", "气声适中", "气声偏多", "气声明显")
    )
    rough_text = score_level(
        rough, ("非常平滑", "较平滑", "轻微颗粒", "明显颗粒", "沙哑粗粝")
    )
    emo_text = score_level(
        emotion, ("情绪克制", "情绪偏弱", "情绪中等", "情绪饱满", "情绪爆发力强")
    )
    if charm >= 70 and gender == "女":
        charm_text = "带有魅惑引导感"
    elif charm >= 70:
        charm_text = "带有强烈蛊惑感"
    elif charm <= 30:
        charm_text = "风格朴素，不刻意卖弄"
    else:
        charm_text = "风格自然"

    parts = [
        f"{gender}声线，{age_text}，{align_text}",
        pitch_text,
        speed_text,
        bright_text,
        breath_text,
        rough_text,
        emo_text,
        charm_text,
        "吐字清晰，保持角色稳定一致。",
    ]
    if style:
        parts.append(f"附加特征：{style}")
    return "，".join(parts)


def safe_name(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    s = re.sub(r"\s+", "_", s)
    return s[:80] if s else "unknown"


def load_role_profiles(csv_path: Path) -> Dict[str, Dict[str, str]]:
    roles: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "role" not in reader.fieldnames:
            raise ValueError("CSV must include a 'role' column.")
        for row in reader:
            role = (row.get("role") or "").strip()
            if not role:
                continue
            roles[role] = row
    if not roles:
        raise ValueError("No valid role rows found in CSV.")
    return roles


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def normalize_line(line: dict) -> Optional[dict]:
    role = (line.get("role") or line.get("speaker") or "旁白").strip()
    text = (line.get("dialogue") or line.get("text") or "").strip()
    if not text:
        return None
    language = (line.get("language") or "Chinese").strip()
    return {"role": role, "text": text, "language": language}


def load_chapters(input_path: Path) -> Dict[str, List[dict]]:
    chapters: Dict[str, List[dict]] = {}

    def add_lines(ch_name: str, raw_lines: List[dict]) -> None:
        normed = []
        for x in raw_lines:
            n = normalize_line(x)
            if n is not None:
                normed.append(n)
        if normed:
            chapters[ch_name] = normed

    if input_path.is_dir():
        files = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in {".json", ".jsonl"}],
            key=lambda p: p.name,
        )
        for p in files:
            chapter_name = p.stem
            data = load_jsonl(p) if p.suffix.lower() == ".jsonl" else load_json(p)
            if isinstance(data, dict) and isinstance(data.get("lines"), list):
                add_lines(str(data.get("chapter") or chapter_name), data["lines"])
            elif isinstance(data, list):
                add_lines(chapter_name, data)
    else:
        data = load_jsonl(input_path) if input_path.suffix.lower() == ".jsonl" else load_json(input_path)
        if isinstance(data, dict):
            if isinstance(data.get("chapters"), list):
                for ch in data["chapters"]:
                    if isinstance(ch, dict) and isinstance(ch.get("lines"), list):
                        add_lines(str(ch.get("chapter") or f"chapter_{len(chapters)+1:03d}"), ch["lines"])
            elif isinstance(data.get("lines"), list):
                add_lines(str(data.get("chapter") or "chapter_001"), data["lines"])
        elif isinstance(data, list):
            if data and isinstance(data[0], dict) and "chapter" in data[0]:
                grouped: Dict[str, List[dict]] = {}
                for item in data:
                    chapter_name = str(item.get("chapter") or "chapter_001")
                    grouped.setdefault(chapter_name, []).append(item)
                for ch_name, lines in grouped.items():
                    add_lines(ch_name, lines)
            else:
                add_lines("chapter_001", data)

    if not chapters:
        raise ValueError("No chapter lines found from input.")
    return chapters


def get_device_and_dtype(force_device: Optional[str]) -> Tuple[str, torch.dtype]:
    if force_device:
        device = force_device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return device, dtype


def build_srt_ts(sec: float) -> str:
    total_ms = int(round(sec * 1000))
    hh = total_ms // 3600000
    mm = (total_ms % 3600000) // 60000
    ss = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(path: Path, entries: List[dict]) -> None:
    lines: List[str] = []
    for idx, e in enumerate(entries, start=1):
        lines.append(str(idx))
        lines.append(f"{build_srt_ts(e['start'])} --> {build_srt_ts(e['end'])}")
        lines.append(f"{e['role']}: {e['text']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_prompt_payload(path: Path, items: List[VoiceClonePromptItem], meta: dict) -> None:
    payload_items = []
    for it in items:
        d = asdict(it)
        if d["ref_code"] is not None and torch.is_tensor(d["ref_code"]):
            d["ref_code"] = d["ref_code"].detach().cpu()
        if torch.is_tensor(d["ref_spk_embedding"]):
            d["ref_spk_embedding"] = d["ref_spk_embedding"].detach().cpu()
        payload_items.append(d)
    torch.save({"items": payload_items, "meta": meta}, path)


def load_prompt_payload(path: Path) -> List[VoiceClonePromptItem]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(f"Invalid prompt payload: {path}")
    out: List[VoiceClonePromptItem] = []
    for d in payload["items"]:
        ref_code = d.get("ref_code")
        ref_spk = d.get("ref_spk_embedding")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        out.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", True)),
                ref_text=d.get("ref_text"),
            )
        )
    if not out:
        raise ValueError(f"Empty prompt items in payload: {path}")
    return out


def ensure_role_prompt(
    role: str,
    profile: Dict[str, str],
    prompt_path: Path,
    ref_wav_path: Path,
    design_model: Qwen3TTSModel,
    base_model: Qwen3TTSModel,
    ref_text: str,
    regen: bool,
    temperature: float,
    top_p: float,
) -> List[VoiceClonePromptItem]:
    if prompt_path.exists() and not regen:
        return load_prompt_payload(prompt_path)

    voice_prompt = build_voice_prompt(profile)
    language = (profile.get("language") or "Chinese").strip()

    wavs, sr = design_model.generate_voice_design(
        text=ref_text,
        language=language,
        instruct=voice_prompt,
        temperature=temperature,
        top_p=top_p,
    )
    ref_wav = wavs[0]
    sf.write(str(ref_wav_path), ref_wav, sr)

    items = base_model.create_voice_clone_prompt(
        ref_audio=(ref_wav, sr),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    save_prompt_payload(
        prompt_path,
        items,
        meta={
            "role": role,
            "voice_prompt": voice_prompt,
            "language": language,
            "ref_text": ref_text,
        },
    )
    return items


def choose_fallback_role(roles: Dict[str, Dict[str, str]]) -> str:
    for key in ("旁白", "narrator", "Narrator"):
        if key in roles:
            return key
    return next(iter(roles.keys()))


def synthesize_chapter(
    chapter_name: str,
    lines: List[dict],
    role_prompts: Dict[str, List[VoiceClonePromptItem]],
    fallback_role: str,
    base_model: Qwen3TTSModel,
    out_wav: Path,
    out_srt: Path,
    gap_ms: int,
    temperature: float,
    top_p: float,
    line_wavs_dir: Optional[Path] = None,
) -> None:
    chunks: List[np.ndarray] = []
    srt_entries: List[dict] = []
    sample_rate: Optional[int] = None
    t = 0.0

    if line_wavs_dir is not None:
        line_wavs_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(lines, start=1):
        role = item["role"]
        text = item["text"]
        language = item["language"]
        prompt_items = role_prompts.get(role) or role_prompts[fallback_role]

        wavs, sr = base_model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
            temperature=temperature,
            top_p=top_p,
        )
        wav = wavs[0].astype(np.float32)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError(f"Sample rate mismatch in {chapter_name}: {sample_rate} vs {sr}")

        start = t
        dur = len(wav) / float(sample_rate)
        end = start + dur
        srt_entries.append({"role": role, "text": text, "start": start, "end": end})

        chunks.append(wav)
        if line_wavs_dir is not None:
            line_file = line_wavs_dir / f"{i:04d}_{safe_name(role)}.wav"
            sf.write(str(line_file), wav, sample_rate)

        gap = np.zeros(int(sample_rate * gap_ms / 1000.0), dtype=np.float32)
        chunks.append(gap)
        t = end + (gap_ms / 1000.0)

    if sample_rate is None:
        raise ValueError(f"No valid lines for chapter: {chapter_name}")

    full = np.concatenate(chunks, axis=0) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(str(out_wav), full, sample_rate)
    write_srt(out_srt, srt_entries)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate chapter-level audio + SRT from role CSV (0-100 sliders) and chapter JSON."
    )
    p.add_argument("--roles-csv", required=True, help="Role config CSV path (must include role column).")
    p.add_argument("--chapters", required=True, help="Chapter input path (.json/.jsonl file or a directory).")
    p.add_argument("--out-dir", default="outputs_chapter_tts", help="Output directory.")
    p.add_argument("--design-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", help="VoiceDesign model id/path.")
    p.add_argument("--base-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Base model id/path.")
    p.add_argument("--device", default=None, help="Device map, e.g. cuda:0 or cpu.")
    p.add_argument("--ref-text", default="你好，这是角色音色参考文本，用于后续整章复用。", help="Reference text for prompt building.")
    p.add_argument("--gap-ms", type=int, default=300, help="Silence gap between lines in merged chapter wav.")
    p.add_argument("--regen-prompts", action="store_true", help="Regenerate role prompt cache even if .pt exists.")
    p.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p.")
    p.add_argument("--save-line-wavs", action="store_true", help="Save per-line wav files under each chapter folder.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    roles_csv = Path(args.roles_csv).resolve()
    chapters_path = Path(args.chapters).resolve()
    out_dir = Path(args.out_dir).resolve()
    prompt_dir = out_dir / "voice_prompts"
    ref_dir = out_dir / "voice_refs"
    chapter_dir = out_dir / "chapters"

    prompt_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    chapter_dir.mkdir(parents=True, exist_ok=True)

    roles = load_role_profiles(roles_csv)
    chapters = load_chapters(chapters_path)
    fallback_role = choose_fallback_role(roles)

    device, dtype = get_device_and_dtype(args.device)
    print(f"[Info] device={device} dtype={dtype}")
    print("[Info] Loading models...")
    design_model = Qwen3TTSModel.from_pretrained(args.design_model, device_map=device, dtype=dtype)
    base_model = Qwen3TTSModel.from_pretrained(args.base_model, device_map=device, dtype=dtype)

    role_prompts: Dict[str, List[VoiceClonePromptItem]] = {}
    for role, profile in roles.items():
        safe_role = safe_name(role)
        prompt_path = prompt_dir / f"{safe_role}.pt"
        ref_path = ref_dir / f"{safe_role}.wav"
        print(f"[Info] preparing role prompt: {role}")
        role_prompts[role] = ensure_role_prompt(
            role=role,
            profile=profile,
            prompt_path=prompt_path,
            ref_wav_path=ref_path,
            design_model=design_model,
            base_model=base_model,
            ref_text=args.ref_text,
            regen=bool(args.regen_prompts),
            temperature=args.temperature,
            top_p=args.top_p,
        )

    if fallback_role not in role_prompts:
        fallback_role = next(iter(role_prompts.keys()))
    print(f"[Info] fallback role = {fallback_role}")

    for ch_name, lines in chapters.items():
        ch_safe = safe_name(ch_name)
        ch_folder = chapter_dir / ch_safe
        ch_folder.mkdir(parents=True, exist_ok=True)
        out_wav = ch_folder / f"{ch_safe}.wav"
        out_srt = ch_folder / f"{ch_safe}.srt"
        line_wavs = ch_folder / "lines" if args.save_line_wavs else None
        print(f"[Info] synthesizing chapter: {ch_name} ({len(lines)} lines)")
        synthesize_chapter(
            chapter_name=ch_name,
            lines=lines,
            role_prompts=role_prompts,
            fallback_role=fallback_role,
            base_model=base_model,
            out_wav=out_wav,
            out_srt=out_srt,
            gap_ms=args.gap_ms,
            temperature=args.temperature,
            top_p=args.top_p,
            line_wavs_dir=line_wavs,
        )
        print(f"[Done] {out_wav}")
        print(f"[Done] {out_srt}")


if __name__ == "__main__":
    main()
