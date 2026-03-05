#!/usr/bin/env python3
"""Compatibility wrapper for archived IoF tau ablation script.

Primary active entrypoint is scripts/pipeline.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("archive") / "run_iof_tau_ablation.py"
    print("[deprecated] scripts/run_iof_tau_ablation.py moved to scripts/archive/.")
    print("[hint] Use scripts/pipeline.py for day-to-day train/eval runs.")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
