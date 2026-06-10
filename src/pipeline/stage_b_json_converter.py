# -*- coding: utf-8 -*-
"""
Stage B: Batch convert chapter txt -> structured JSON via LLM.
Resume/pause by chapter using progress.json.

Usage:
  python src/pipeline/stage_b_json_converter.py
  python src/pipeline/stage_b_json_converter.py --limit 30
  python src/pipeline/stage_b_json_converter.py --start 10
  python src/pipeline/stage_b_json_converter.py --force

Set DASHSCOPE_API_KEY environment variable or use --api_key.
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.lib.config import DATA_DIR, CHAPTER_DIR, CHAPTERS_JSON_DIR, DASHSCOPE_BASE_URL, LLM_MODEL, PROJECT_ROOT

from openai import OpenAI


# =========================
# “结构化拆分器”prompt
# =========================
SYSTEM_PROMPT = """你是“小说章节结构化拆分器”。你的任务是把输入章节文本转换为结构化 JSON。只输出 JSON，不输出任何解释、注释或 Markdown。

必须遵守：
1. 不改写原文：lines[i].dialogue 必须是原文连续片段（可去首尾空白，不可润色/拼接非连续文本）。
2. 无证据不臆造：无法确定时填 null 或 []。
3. 输出字段固定，禁止新增或删除字段。
4. lines 顺序必须与原文出现顺序一致。
5. 注意上下文关系，不要将说话的角色弄错。

输出结构固定为：
{
  "chapter_title": string|null,
  "lines": [
    {
      "role": string,
      "gender": "男"|"女"|null,
      "dialogue": string,
      "emotion": string|null,
      "feature": string|null,
      "evidence": {
        "role_quotes": [string],
        "gender_quotes": [string],
        "emotion_quotes": [string],
        "feature_quotes": [string]
      },
      "notes": string|null
    }
  ],
  "role_registry": {
    "<角色名或占位名或旁白>": {
      "gender": "男"|"女"|null,
      "feature": string|null,
      "first_seen_line_index": integer
    }
  }
}

判定规则：
- 对话/明确心理独白单独成行；叙述用 role="旁白"。
- chapter_title：若开头有“第X章 ...”则填原文标题，否则 null。
- role/gender/emotion/feature 若非 null，必须在对应 quotes 中给出短证据；否则改为 null 且 quotes=[]。
- feature 仅表示长期基础音色特征（如“低沉/清亮/少年感/沙哑”），不是瞬时情绪。
- 同一角色优先沿用既有 feature；若有新线索可写入 notes。

输出前自检：
- 顶层仅有 chapter_title、lines、role_registry。
- 每个 line 都包含 role、gender、dialogue、emotion、feature、evidence、notes。
- JSON 必须严格可解析。
"""



FIX_JSON_PROMPT = """你将收到一段“疑似 JSON”的文本。
只做一件事：把它修复成“严格可解析的 JSON”。
禁止：新增字段、删改字段语义、改写任何 dialogue 文本内容。
只输出修复后的 JSON 本体（以 { 开始，以 } 结束）。"""


def list_chapter_files(in_dir: Path) -> List[Path]:
    rx = re.compile(r"^(?P<num>\d{3})_(?P<rest>.*)\.txt$", re.IGNORECASE)
    items = []
    for p in in_dir.iterdir():
        if not p.is_file():
            continue
        m = rx.match(p.name)
        if not m:
            continue
        if p.name.lower() == "000_preamble.txt":
            continue
        items.append((int(m.group("num")), p.name, p))

    # 先按数字，再按文件名稳定排序
    items.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in items]




def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # progress.json / 其它json损坏时：备份并重置
        try:
            backup = path.with_suffix(path.suffix + ".bad")
            backup.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        except Exception:
            pass
        return default



def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def call_model_return_json(
    client: OpenAI,
    model: str,
    chapter_text: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    user_input = f"""请按规则把以下章节拆分为 JSON（只输出 JSON）：
<<<TEXT>>>
{chapter_text}
<<<END>>>"""

    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=temperature,
        **kwargs,
    )

    raw = (resp.choices[0].message.content or "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FIX_JSON_PROMPT},
                {"role": "user", "content": raw},
            ],
            temperature=0.0,
        )
        fixed = (resp2.choices[0].message.content or "").strip()
        return json.loads(fixed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(CHAPTER_DIR))
    ap.add_argument("--out_dir", default=str(CHAPTERS_JSON_DIR))
    ap.add_argument("--progress", default=str(CHAPTERS_JSON_DIR / "progress.json"))
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--base_url", default=DASHSCOPE_BASE_URL)
    ap.add_argument("--api_key", default="", help="DashScope API key; also reads env DASHSCOPE_API_KEY / OPENAI_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=0, help="0 表示不传")
    ap.add_argument("--start", type=int, default=None, help="从第 N 个文件开始（0-based，覆盖断点）")
    ap.add_argument("--limit", type=int, default=None, help="最多处理多少章（用于按章暂停）")
    ap.add_argument("--force", action="store_true", help="强制重跑，即使输出 json 已存在")
    ap.add_argument("--sleep", type=float, default=0.0, help="每章间隔秒数")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    progress_path = Path(args.progress)

    if not in_dir.exists():
        raise SystemExit(f"Input dir not found: {in_dir}")

    files = list_chapter_files(in_dir)
    if not files:
        raise SystemExit(f"No files matched: {in_dir}\\<NNN>_*.txt (e.g. 001_*.txt)")

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Provide --api_key or set env DASHSCOPE_API_KEY / OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    progress = load_json(
        progress_path,
        default={
            "last_done_index": -1,
            "last_done_filename": None,
            "done_count": 0,
            "updated_at": None,
        },
    )

    if args.start is not None:
        start_index = args.start
    else:
        last_name = progress.get("last_done_filename")
        if last_name:
            name_to_index = {p.name: i for i, p in enumerate(files)}
            if last_name in name_to_index:
                start_index = name_to_index[last_name] + 1
            else:
                # 进度文件里的文件名不在当前列表（列表变了/文件被改名/删了）
                start_index = 0
        else:
            start_index = 0

    todo = files[start_index:]
    if args.limit is not None:
        todo = todo[: args.limit]

    if not todo:
        print("Nothing to do.")
        return

    max_tokens = None if args.max_tokens == 0 else args.max_tokens

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, txt_path in enumerate(todo, start=start_index):
        out_path = out_dir / (txt_path.stem + ".json")

        if out_path.exists() and not args.force:
            progress["last_done_index"] = idx
            progress["last_done_filename"] = txt_path.name
            progress["done_count"] = int(progress.get("done_count", 0)) + 1
            progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_json(progress_path, progress)
            print(f"[SKIP] {txt_path.name} -> exists")
            continue

        try:
            chapter_text = txt_path.read_text(encoding="utf-8")
            data = call_model_return_json(
                client=client,
                model=args.model,
                chapter_text=chapter_text,
                temperature=args.temperature,
                max_tokens=max_tokens,
            )
            save_json(out_path, data)

            progress["last_done_index"] = idx
            progress["last_done_filename"] = txt_path.name
            progress["done_count"] = int(progress.get("done_count", 0)) + 1
            progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_json(progress_path, progress)

            print(f"[OK] {txt_path.name} -> {out_path.name}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        except KeyboardInterrupt:
            print("\nInterrupted. You can rerun to resume.")
            break
        except Exception as e:
            err_path = out_dir / (txt_path.stem + ".error.txt")
            err_path.write_text(str(e), encoding="utf-8")
            print(f"[ERROR] {txt_path.name} -> {e}")
            print(f"        wrote: {err_path}")
            break


if __name__ == "__main__":
    main()
