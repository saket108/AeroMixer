#!/usr/bin/env python3
"""Release discipline utilities for AeroMixer.

Commands:
- check: validate release prerequisites
- tag: create annotated git tag (optionally dry-run)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass
class CheckItem:
    name: str
    ok: bool
    details: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return int(p.returncode), p.stdout.strip(), p.stderr.strip()


def _check_clean_worktree(root: Path) -> CheckItem:
    rc, out, _ = _run(["git", "status", "--porcelain"], root)
    if rc != 0:
        return CheckItem("clean_worktree", False, "failed to query git status")
    if out:
        return CheckItem("clean_worktree", False, "working tree has uncommitted changes")
    return CheckItem("clean_worktree", True, "working tree is clean")


def _check_required_files(root: Path) -> CheckItem:
    required = [
        "CHANGELOG.md",
        "RELEASE_CHECKLIST.md",
        "requirements_lock.txt",
        "requirements_lock_colab.txt",
        "requirements_lock_colab_minimal.txt",
        "scripts/pipeline.py",
        "scripts/internal/train_any_dataset.py",
        "scripts/internal/build_tiled_yolo_dataset.py",
        "scripts/colab_bootstrap.sh",
        "config_files/presets/lite.yaml",
        "config_files/presets/full.yaml",
        "config_files/presets/prod.yaml",
        "Dockerfile",
    ]
    missing = [p for p in required if not (root / p).exists()]
    if missing:
        return CheckItem("required_files", False, f"missing: {missing}")
    return CheckItem("required_files", True, "all required files present")


def _check_changelog(root: Path, version: str) -> CheckItem:
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        return CheckItem("changelog", False, "CHANGELOG.md not found")
    text = changelog.read_text(encoding="utf-8", errors="ignore")
    if "## [Unreleased]" not in text:
        return CheckItem("changelog", False, "missing [Unreleased] section")
    if f"## [{version}]" in text or f"## [v{version}]" in text:
        return CheckItem("changelog", True, f"release section {version} found")
    return CheckItem("changelog", False, f"release section {version} not found")


def _check_tag_absent(root: Path, version: str) -> CheckItem:
    tag = f"v{version}"
    rc, out, _ = _run(["git", "tag", "--list", tag], root)
    if rc != 0:
        return CheckItem("tag_absent", False, "failed to list tags")
    if out.strip():
        return CheckItem("tag_absent", False, f"tag {tag} already exists")
    return CheckItem("tag_absent", True, f"tag {tag} does not exist yet")


def _prepare_changelog_section(root: Path, version: str, date_str: str | None) -> Path:
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        raise FileNotFoundError("CHANGELOG.md not found")

    text = changelog.read_text(encoding="utf-8", errors="ignore")
    heading = f"## [{version}]"
    if heading in text or f"## [v{version}]" in text:
        return changelog

    lines = text.splitlines(keepends=True)
    idx_unreleased = None
    for i, line in enumerate(lines):
        if line.strip() == "## [Unreleased]":
            idx_unreleased = i
            break
    if idx_unreleased is None:
        raise RuntimeError("Missing '## [Unreleased]' section in CHANGELOG.md")

    idx_insert = len(lines)
    for i in range(idx_unreleased + 1, len(lines)):
        if lines[i].startswith("## ["):
            idx_insert = i
            break

    rel_date = date_str or datetime.now(timezone.utc).date().isoformat()
    block = [
        "\n",
        f"## [{version}] - {rel_date}\n",
        "\n",
        "### Added\n",
        "- TODO\n",
        "\n",
    ]
    lines[idx_insert:idx_insert] = block
    changelog.write_text("".join(lines), encoding="utf-8")
    return changelog


def _cmd_check(args: argparse.Namespace) -> int:
    root = _repo_root()
    items = [
        _check_clean_worktree(root),
        _check_required_files(root),
        _check_changelog(root, args.version),
        _check_tag_absent(root, args.version),
    ]
    ok = all(i.ok for i in items)
    report = {
        "ok": ok,
        "version": args.version,
        "checks": [asdict(i) for i in items],
    }
    print(json.dumps(report, indent=2))
    if args.report_out:
        out = Path(args.report_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {out}")
    return 0 if ok else 2


def _cmd_prepare(args: argparse.Namespace) -> int:
    root = _repo_root()
    path = _prepare_changelog_section(root, args.version, args.date)
    print(f"Prepared changelog for version {args.version}: {path}")
    return 0


def _cmd_tag(args: argparse.Namespace) -> int:
    root = _repo_root()
    tag = f"v{args.version}"
    msg = args.message or f"Release {tag}"

    rc_check = _cmd_check(argparse.Namespace(version=args.version, report_out=None))
    if rc_check != 0 and not args.force:
        print("Release checks failed. Use --force to tag anyway.")
        return rc_check

    cmd = ["git", "tag", "-a", tag, "-m", msg]
    print(">> " + " ".join(cmd))
    if args.dry_run:
        return 0
    rc, out, err = _run(cmd, root)
    if out:
        print(out)
    if err:
        print(err)
    return rc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Release tooling for AeroMixer.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="Run release readiness checks.")
    c.add_argument("--version", required=True, help="Version without v-prefix (e.g. 0.5.0).")
    c.add_argument("--report-out", default=None, help="Optional JSON report output path.")
    c.set_defaults(fn=_cmd_check)

    prep = sub.add_parser("prepare", help="Insert release section in CHANGELOG.md.")
    prep.add_argument("--version", required=True, help="Version without v-prefix (e.g. 0.5.0).")
    prep.add_argument("--date", default=None, help="Optional release date (YYYY-MM-DD).")
    prep.set_defaults(fn=_cmd_prepare)

    t = sub.add_parser("tag", help="Create an annotated git tag.")
    t.add_argument("--version", required=True, help="Version without v-prefix (e.g. 0.5.0).")
    t.add_argument("--message", default=None, help="Annotated tag message.")
    t.add_argument("--force", action="store_true", help="Tag even if checks fail.")
    t.add_argument("--dry-run", action="store_true", help="Print command only.")
    t.set_defaults(fn=_cmd_tag)
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not VERSION_RE.match(args.version):
        raise SystemExit("Invalid version format. Use semantic version x.y.z")
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
