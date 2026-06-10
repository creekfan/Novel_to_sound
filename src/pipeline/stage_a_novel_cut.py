#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage A: Split a novel .txt into per-chapter files by headings like 第*章.

Examples of chapter headings supported:
- 第1章 / 第 1 章 / 第十二章 / 第12章：标题 / 第12章-标题

Usage:
  python src/pipeline/stage_a_novel_cut.py data/novel/novel.txt -o data/chapter
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


CHAPTER_RE = re.compile(
    r"""(?mx)               # m: multiline, x: verbose
    ^\s*                    # line start + optional spaces
    第\s*                   # '第'
    (?P<num>[0-9一二三四五六七八九十百千万两零〇]+)  # chapter number (digits or Chinese numerals)
    \s*章                   # '章'
    (?P<title>[^\n\r]*)     # rest of the line as title (optional)
    \s*$                    # optional trailing spaces till line end
    """
)


def sanitize_filename(name: str) -> str:
    # Windows-forbidden: \ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = name.strip()
    return name if name else "untitled"


def find_chapter_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Return list of (start_idx, end_idx, heading_line) spans for each chapter.
    end_idx is exclusive and points to next chapter start or EOF.
    """
    matches = list(CHAPTER_RE.finditer(text))
    spans: List[Tuple[int, int, str]] = []
    if not matches:
        return spans

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading_line = m.group(0).strip()
        spans.append((start, end, heading_line))
    return spans


def make_output_name(index: int, heading_line: str) -> str:
    """
    Create filename like: 001_第十二章_标题.txt
    """
    # Use full heading but keep it safe
    base = sanitize_filename(heading_line)
    return f"{index:03d}_{base}.txt"


def split_file(input_path: Path, out_dir: Path, keep_preamble: bool) -> None:
    text = input_path.read_text(encoding="utf-8")

    spans = find_chapter_spans(text)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not spans:
        # No chapters found; write whole file as 000_full.txt
        (out_dir / "000_full.txt").write_text(text, encoding="utf-8")
        print("No chapter headings matched. Wrote: 000_full.txt")
        return

    # Optional preamble (text before first chapter)
    preamble = text[:spans[0][0]]
    if keep_preamble and preamble.strip():
        (out_dir / "000_preamble.txt").write_text(preamble, encoding="utf-8")
        print("Wrote: 000_preamble.txt")

    for idx, (start, end, heading) in enumerate(spans, start=1):
        chunk = text[start:end]
        filename = make_output_name(idx, heading)
        (out_dir / filename).write_text(chunk, encoding="utf-8")
        print(f"Wrote: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Split txt file into per-chapter files by headings like 第*章")
    parser.add_argument("input", type=str, help="Input txt file path")
    parser.add_argument("-o", "--out", type=str, default="chapters_out", help="Output directory")
    parser.add_argument(
        "--keep-preamble",
        action="store_true",
        help="If set, save text before first chapter into 000_preamble.txt",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    if not input_path.exists() or not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    split_file(input_path, out_dir, keep_preamble=args.keep_preamble)


if __name__ == "__main__":
    main()