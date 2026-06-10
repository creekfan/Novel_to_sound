#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for Stage B: run json_converter in chunks.
Replaces the broken run_json_batches_100.bat.

Usage:
  python scripts/run_json_batches.py --batch-size 100
  python scripts/run_json_batches.py --batch-size 50 --start 0
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_B_SCRIPT = PROJECT_ROOT / "src" / "pipeline" / "stage_b_json_converter.py"


def main():
    ap = argparse.ArgumentParser(description="Run Stage B JSON converter in batches")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    args = ap.parse_args()

    batch = args.batch_size
    start = args.start

    while True:
        cmd = [
            args.python,
            str(STAGE_B_SCRIPT),
            "--start", str(start),
            "--limit", str(batch),
        ]
        print(f"Running batch: start={start} limit={batch}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"Stopped at start={start} (exit code {result.returncode})")
            break
        start += batch


if __name__ == "__main__":
    main()
