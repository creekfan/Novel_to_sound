import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch


def bootstrap_repo() -> None:
    repo_root = Path(os.environ.get("QWEN3_TTS_ROOT", r"D:\Qwen3-TTS")).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


bootstrap_repo()

from qwen_tts import Qwen3TTSModel


MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def build_instruct(gender: str) -> str:
    g = gender.strip().lower()
    if g in {"男", "male", "m"}:
        return "男性声音。"
    if g in {"女", "female", "f"}:
        return "女性声音。"
    raise ValueError("gender 只支持: 男/女 (或 male/female)")


def main() -> None:
    parser = argparse.ArgumentParser(description="最简声音设计脚本：gender + prompt(音色) + text(内容)")
    parser.add_argument("--gender", required=True, help="男 或 女")
    parser.add_argument("--prompt", required=True, help="声音设计描述（音色 prompt）")
    parser.add_argument("--text", required=True, help="测试朗读文本")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度，越大变化越强")
    parser.add_argument("--top_p", type=float, default=1.0, help="核采样阈值，通常 0.8~1.0")
    parser.add_argument("--out", default=None, help="输出 wav 文件名，不填则自动带时间戳")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=dtype,
    )

    instruct = f"{build_instruct(args.gender)} {args.prompt}"
    out_path = args.out or f"voice_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    wavs, sr = tts.generate_voice_design(
        text=args.text,
        language="Chinese",
        instruct=instruct,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    sf.write(out_path, wavs[0], sr)
    with open(out_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    print(f"instruct: {instruct}")
    print(f"saved: {out_path}")
    print(f"md5: {md5}")


if __name__ == "__main__":
    main()
