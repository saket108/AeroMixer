#!/usr/bin/env python3
"""Compatibility wrapper for archived baseline benchmark script.

Primary active entrypoint is scripts/pipeline.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("archive") / "run_baseline_benchmarks.py"
    print("[deprecated] scripts/run_baseline_benchmarks.py moved to scripts/archive/.")
    print("[hint] Use scripts/pipeline.py for day-to-day train/eval runs.")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
