#!/usr/bin/env python3
"""Validate Docker build/inference contract for AeroMixer.

This script checks Docker availability, optionally builds the image, and
runs a contract command (default: pipeline --help) inside the container.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class StepResult:
    name: str
    ok: bool
    command: str
    return_code: int
    stdout_tail: str
    stderr_tail: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> StepResult:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        return StepResult(
            name=cmd[0] if cmd else "unknown",
            ok=(p.returncode == 0),
            command=" ".join(shlex.quote(x) for x in cmd),
            return_code=int(p.returncode),
            stdout_tail=(p.stdout or "")[-4000:],
            stderr_tail=(p.stderr or "")[-4000:],
        )
    except FileNotFoundError as exc:
        return StepResult(
            name=cmd[0] if cmd else "unknown",
            ok=False,
            command=" ".join(shlex.quote(x) for x in cmd),
            return_code=127,
            stdout_tail="",
            stderr_tail=str(exc),
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate Docker inference contract.")
    p.add_argument("--image", default="aeromixer:latest")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument(
        "--container-cmd",
        default="python scripts/pipeline.py --help",
        help="Command executed inside container.",
    )
    p.add_argument(
        "--report-out",
        default="output/docker/docker_validation_report.json",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    root = _repo_root()
    steps: list[StepResult] = []

    docker_check = _run(["docker", "--version"], cwd=root)
    steps.append(
        StepResult(
            name="docker_available",
            ok=docker_check.ok,
            command=docker_check.command,
            return_code=docker_check.return_code,
            stdout_tail=docker_check.stdout_tail,
            stderr_tail=docker_check.stderr_tail,
        )
    )
    if not docker_check.ok:
        report = {
            "created_at_utc": _now_iso(),
            "ok": False,
            "reason": "docker_not_available",
            "steps": [asdict(s) for s in steps],
        }
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Docker not available. Report: {out}")
        return 2

    if not args.skip_build:
        build_step = _run(["docker", "build", "-t", args.image, "."], cwd=root)
        steps.append(
            StepResult(
                name="docker_build",
                ok=build_step.ok,
                command=build_step.command,
                return_code=build_step.return_code,
                stdout_tail=build_step.stdout_tail,
                stderr_tail=build_step.stderr_tail,
            )
        )
        if not build_step.ok:
            report = {
                "created_at_utc": _now_iso(),
                "ok": False,
                "reason": "docker_build_failed",
                "steps": [asdict(s) for s in steps],
            }
            out = Path(args.report_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Docker build failed. Report: {out}")
            return 2

    run_cmd = [
        "docker",
        "run",
        "--rm",
        args.image,
        "sh",
        "-lc",
        args.container_cmd,
    ]
    run_step = _run(run_cmd, cwd=root)
    steps.append(
        StepResult(
            name="docker_run_contract",
            ok=run_step.ok,
            command=run_step.command,
            return_code=run_step.return_code,
            stdout_tail=run_step.stdout_tail,
            stderr_tail=run_step.stderr_tail,
        )
    )

    ok = all(s.ok for s in steps)
    report = {
        "created_at_utc": _now_iso(),
        "ok": ok,
        "image": args.image,
        "container_cmd": args.container_cmd,
        "steps": [asdict(s) for s in steps],
    }
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Docker validation report: {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
