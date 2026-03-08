"""Export an Excel workbook summarizing AeroMixer change history.

The report is based on local git history and the current working tree.
It includes:
- commit-level summary
- per-file commit changes
- file-level aggregate summary
- current uncommitted changes
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.internal.export_experiment_workbook import discover_runs, write_xlsx


def run_git(repo_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def maybe_int(value: str) -> int | None:
    value = str(value).strip()
    if value == "-" or not value:
        return None
    return int(value)


def parse_commit_history(repo_root: Path) -> list[dict[str, Any]]:
    raw = run_git(
        repo_root,
        [
            "log",
            "--date=iso-strict",
            "--numstat",
            "--format=%x1e%H%x1f%h%x1f%ad%x1f%an%x1f%ae%x1f%s",
        ],
    )

    commits: list[dict[str, Any]] = []
    for chunk in raw.split("\x1e"):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        header = lines[0].split("\x1f")
        if len(header) != 6:
            continue
        commit_hash, short_hash, commit_date, author_name, author_email, subject = (
            header
        )
        files = []
        additions = 0
        deletions = 0
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, deleted, path = parts
            added_num = maybe_int(added)
            deleted_num = maybe_int(deleted)
            if added_num is not None:
                additions += added_num
            if deleted_num is not None:
                deletions += deleted_num
            files.append(
                {
                    "path": path,
                    "added": added_num,
                    "deleted": deleted_num,
                    "added_raw": added,
                    "deleted_raw": deleted,
                }
            )

        commits.append(
            {
                "commit": commit_hash,
                "short_commit": short_hash,
                "date": commit_date,
                "date_dt": datetime.fromisoformat(commit_date),
                "author_name": author_name,
                "author_email": author_email,
                "subject": subject,
                "files": files,
                "files_changed": len(files),
                "additions": additions,
                "deletions": deletions,
            }
        )
    return commits


def parse_current_worktree(repo_root: Path) -> list[dict[str, Any]]:
    status_raw = run_git(repo_root, ["status", "--porcelain=v1"])
    diff_numstat_raw = run_git(repo_root, ["diff", "--numstat", "HEAD"])

    diff_map: dict[str, tuple[str, str]] = {}
    for line in diff_numstat_raw.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        diff_map[parts[2]] = (parts[0], parts[1])

    rows = []
    for line in status_raw.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:]
        added_raw, deleted_raw = diff_map.get(path, ("", ""))
        rows.append(
            {
                "status": status,
                "path": path,
                "added": maybe_int(added_raw),
                "deleted": maybe_int(deleted_raw),
                "added_raw": added_raw,
                "deleted_raw": deleted_raw,
            }
        )
    return rows


def parse_run_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H.%M.%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def build_file_summary(commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "commits_touched": 0,
            "total_added": 0,
            "total_deleted": 0,
            "last_commit": "",
            "last_date": "",
            "last_subject": "",
        }
    )

    for commit in commits:
        for file_change in commit["files"]:
            path = file_change["path"]
            row = summary[path]
            row["commits_touched"] += 1
            row["total_added"] += file_change["added"] or 0
            row["total_deleted"] += file_change["deleted"] or 0
            if not row["last_commit"]:
                row["last_commit"] = commit["short_commit"]
                row["last_date"] = commit["date"]
                row["last_subject"] = commit["subject"]

    return [
        {"path": path, **values}
        for path, values in sorted(
            summary.items(),
            key=lambda item: (
                -item[1]["commits_touched"],
                -(item[1]["total_added"] + item[1]["total_deleted"]),
                item[0],
            ),
        )
    ]


MAJOR_CHANGE_RULES = [
    {
        "match": "Configure AeroMixer for aircraft dataset training",
        "title": "Dataset Direction Set",
        "category": "product",
        "summary": "Configured the repo around aircraft-defect training and the new target dataset.",
        "impact": "Established the domain-specific baseline for subsequent detector work.",
    },
    {
        "match": "Convert AeroMixer from video action detection to image object detection",
        "title": "Image Detection Pivot",
        "category": "architecture",
        "summary": "Moved AeroMixer away from video action detection and into image object detection.",
        "impact": "Changed the project’s core task and made image detection the main runtime path.",
    },
    {
        "match": "feat: finalize image-text multimodal pipeline fixes",
        "title": "Multimodal Pipeline Stabilized",
        "category": "multimodal",
        "summary": "Fixed the image+text pipeline so multimodal training and inference worked more reliably.",
        "impact": "Made multimodal detection usable instead of experimental-only.",
    },
    {
        "match": "Professionalize pipeline workflow and archive research scripts",
        "title": "Pipeline Cleanup Started",
        "category": "tooling",
        "summary": "Separated the active workflow from older research scripts and streamlined the main pipeline.",
        "impact": "Reduced project clutter and made the supported path clearer.",
    },
    {
        "match": "Add production preset and guardrails to pipeline",
        "title": "Production Preset Added",
        "category": "tooling",
        "summary": "Added production-oriented defaults and safety checks to the pipeline.",
        "impact": "Improved reproducibility and reduced accidental bad runs.",
    },
    {
        "match": "Add fail-fast dataset validation and pipeline integration",
        "title": "Dataset Validation Integrated",
        "category": "data",
        "summary": "Added fail-fast dataset validation directly into the pipeline flow.",
        "impact": "Caught broken datasets earlier and made runs more trustworthy.",
    },
    {
        "match": "Add small-object tiling workflow to pipeline",
        "title": "Small-Object Tiling Added",
        "category": "small-objects",
        "summary": "Added tiling support aimed at small defect detection.",
        "impact": "Improved the project’s strategy for tiny aircraft defects.",
    },
    {
        "match": "Professionalize tooling: canonical pipeline, presets, release/deploy, CI/tests",
        "title": "Tooling Professionalized",
        "category": "tooling",
        "summary": "Standardized presets, CI, release tooling, and the canonical pipeline entrypoint.",
        "impact": "Made the repo look and behave more like a maintained product.",
    },
    {
        "match": "Improve image training robustness: balanced sampling, class-weighted loss, aug",
        "title": "Training Robustness Improved",
        "category": "training",
        "summary": "Added balanced sampling, class-weighted loss, and stronger augmentation.",
        "impact": "Improved long-tail training behavior and robustness for uneven defect classes.",
    },
    {
        "match": "Add AP50:95 reporting and wire full/prod evaluation defaults",
        "title": "Evaluation Upgraded",
        "category": "evaluation",
        "summary": "Added AP50:95 reporting and stronger evaluation defaults for full and prod presets.",
        "impact": "Made benchmarks more credible and closer to modern detector reporting.",
    },
    {
        "match": "Improve AeroMixer image detection defaults",
        "title": "Image Detection Defaults Improved",
        "category": "detection",
        "summary": "Improved image detection defaults, stronger recipe settings, and more reliable evaluation behavior.",
        "impact": "Raised the baseline quality for serious detection runs.",
    },
    {
        "match": "Refocus AeroMixer on AeroLite detector runtime",
        "title": "AeroLite Runtime Refocus",
        "category": "architecture",
        "summary": "Cleaned the repo around the AeroLite detector family, removed legacy public surface, and clarified architecture.",
        "impact": "Turned the repo from a mixed research fork into a cleaner detector project.",
    },
    {
        "match": "Align CI with AeroLite runtime",
        "title": "CI Realigned",
        "category": "ci",
        "summary": "Updated CI to match the cleaned AeroLite runtime after legacy CLIP paths were removed.",
        "impact": "Stopped stale workflow failures caused by removed dependencies.",
    },
    {
        "match": "Format repo to satisfy CI black gate",
        "title": "Formatting Gate Fixed",
        "category": "ci",
        "summary": "Reformatted the repo so the active codebase passed the CI formatting gate.",
        "impact": "Brought the cleaned runtime to a green CI state.",
    },
]


def is_result_run(run: dict[str, Any]) -> bool:
    return run.get("status") in {"evaluated", "trained-only"}


def build_major_changes(
    commits: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    worktree_rows: list[dict[str, Any]],
) -> list[list[Any]]:
    rows = [
        [
            "order",
            "status",
            "category",
            "commit_or_tag",
            "date",
            "title",
            "summary",
            "impact",
            "result_link_mode",
            "linked_runs",
            "best_run",
            "best_map50",
            "best_small_ap",
            "result_runs",
            "key_files",
        ]
    ]

    order = 1
    matched_commits: list[dict[str, Any]] = []
    for rule in MAJOR_CHANGE_RULES:
        commit = None
        for item in commits:
            if rule["match"] in item["subject"]:
                commit = item
                break
        if commit is None:
            continue
        matched_commits.append({"rule": rule, "commit": commit})

    matched_commits.sort(key=lambda item: item["commit"]["date_dt"])
    run_rows = []
    for run in runs:
        run_dt = parse_run_datetime(run.get("created_at_utc"))
        run_rows.append({"run": run, "run_dt": run_dt})

    for idx, item in enumerate(matched_commits):
        rule = item["rule"]
        commit = item["commit"]
        key_files = ", ".join(change["path"] for change in commit["files"][:6])
        next_commit_dt = (
            matched_commits[idx + 1]["commit"]["date_dt"]
            if idx + 1 < len(matched_commits)
            else None
        )

        exact_runs = [
            row["run"]
            for row in run_rows
            if row["run"].get("git_commit") == commit["commit"]
            and is_result_run(row["run"])
        ]
        window_runs = []
        if not exact_runs:
            for row in run_rows:
                run = row["run"]
                run_dt = row["run_dt"]
                if not is_result_run(run):
                    continue
                if run_dt is None:
                    continue
                if run_dt < commit["date_dt"]:
                    continue
                if next_commit_dt is not None and run_dt >= next_commit_dt:
                    continue
                window_runs.append(run)

        linked_runs = exact_runs or window_runs
        link_mode = "exact" if exact_runs else ("window" if window_runs else "")
        evaluated_runs = [
            run for run in linked_runs if run.get("status") == "evaluated"
        ]
        best_run = None
        if evaluated_runs:
            best_run = max(
                evaluated_runs, key=lambda run: float(run.get("map50") or -1)
            )
        elif linked_runs:
            best_run = linked_runs[0]

        result_runs = ", ".join(
            f"{run.get('run_name')} ({run.get('map50')})" for run in linked_runs[:4]
        )
        rows.append(
            [
                order,
                "committed",
                rule["category"],
                commit["short_commit"],
                commit["date"],
                rule["title"],
                rule["summary"],
                rule["impact"],
                link_mode,
                len(linked_runs),
                best_run.get("run_name") if best_run else "",
                best_run.get("map50") if best_run else "",
                best_run.get("small_ap") if best_run else "",
                result_runs,
                key_files,
            ]
        )
        order += 1

    if worktree_rows:
        worktree_paths = [row["path"] for row in worktree_rows]
        rows.append(
            [
                order,
                "wip",
                "architecture",
                "worktree",
                datetime.now(timezone.utc).isoformat(),
                "Prompt-Conditioned Scale Routing",
                "Added a new uncommitted novelty path: prompt-conditioned scale routing and prompt-adaptive queries for AeroLite image+text detection, plus workbook exporters.",
                "Introduces a differentiating multimodal detection idea beyond the previous cleanup-only work.",
                "",
                0,
                "",
                "",
                "",
                "",
                ", ".join(worktree_paths[:8]),
            ]
        )

    return rows


def build_change_results(
    commits: list[dict[str, Any]], runs: list[dict[str, Any]]
) -> list[list[Any]]:
    rows = [
        [
            "commit",
            "short_commit",
            "date",
            "subject",
            "change_title",
            "result_link_mode",
            "run_name",
            "run_created_at",
            "run_status",
            "dataset_name",
            "preset",
            "backbone",
            "map50",
            "map5095",
            "small_ap",
            "notes",
        ]
    ]

    matched_commits: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rule in MAJOR_CHANGE_RULES:
        for item in commits:
            if rule["match"] in item["subject"]:
                matched_commits.append((rule, item))
                break
    matched_commits.sort(key=lambda item: item[1]["date_dt"])

    run_rows = [
        {"run": run, "run_dt": parse_run_datetime(run.get("created_at_utc"))}
        for run in runs
    ]

    for idx, (rule, commit) in enumerate(matched_commits):
        next_commit_dt = (
            matched_commits[idx + 1][1]["date_dt"]
            if idx + 1 < len(matched_commits)
            else None
        )
        exact = [
            row["run"]
            for row in run_rows
            if row["run"].get("git_commit") == commit["commit"]
            and is_result_run(row["run"])
        ]
        linked = exact
        link_mode = "exact"
        if not linked:
            linked = []
            link_mode = "window"
            for row in run_rows:
                run = row["run"]
                run_dt = row["run_dt"]
                if not is_result_run(run) or run_dt is None:
                    continue
                if run_dt < commit["date_dt"]:
                    continue
                if next_commit_dt is not None and run_dt >= next_commit_dt:
                    continue
                linked.append(run)
        if not linked:
            rows.append(
                [
                    commit["commit"],
                    commit["short_commit"],
                    commit["date"],
                    commit["subject"],
                    rule["title"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "No recorded local result artifact linked to this change.",
                ]
            )
            continue
        for run in linked:
            rows.append(
                [
                    commit["commit"],
                    commit["short_commit"],
                    commit["date"],
                    commit["subject"],
                    rule["title"],
                    link_mode,
                    run.get("run_name"),
                    run.get("created_at_utc"),
                    run.get("status"),
                    run.get("dataset_name"),
                    run.get("preset"),
                    run.get("backbone"),
                    run.get("map50"),
                    run.get("map5095"),
                    run.get("small_ap"),
                    "",
                ]
            )
    return rows


def build_sheets(
    repo_root: Path,
    commits: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    file_summary: list[dict[str, Any]],
    worktree_rows: list[dict[str, Any]],
) -> dict[str, list[list[Any]]]:
    head_commit = run_git(repo_root, ["rev-parse", "HEAD"]).strip()
    branch = run_git(repo_root, ["branch", "--show-current"]).strip()

    major_change_rows = build_major_changes(commits, runs, worktree_rows)
    change_results_rows = build_change_results(commits, runs)
    commit_rows = [
        [
            "commit",
            "short_commit",
            "date",
            "author_name",
            "author_email",
            "subject",
            "files_changed",
            "additions",
            "deletions",
        ]
    ]
    commit_file_rows = [
        [
            "commit",
            "short_commit",
            "date",
            "subject",
            "path",
            "added",
            "deleted",
            "added_raw",
            "deleted_raw",
        ]
    ]
    file_summary_rows = [
        [
            "path",
            "commits_touched",
            "total_added",
            "total_deleted",
            "last_commit",
            "last_date",
            "last_subject",
        ]
    ]
    worktree_sheet = [
        [
            "status",
            "path",
            "added",
            "deleted",
            "added_raw",
            "deleted_raw",
        ]
    ]
    notes_rows = [
        ["generated_at_utc", datetime.now(timezone.utc).isoformat()],
        ["repo_root", str(repo_root.resolve())],
        ["branch", branch],
        ["head_commit", head_commit],
        ["total_commits", len(commits)],
        ["current_worktree_rows", len(worktree_rows)],
        [
            "note",
            "Commit sheets come from git history. CurrentWorktree captures local uncommitted files at export time.",
        ],
    ]

    for commit in commits:
        commit_rows.append(
            [
                commit["commit"],
                commit["short_commit"],
                commit["date"],
                commit["author_name"],
                commit["author_email"],
                commit["subject"],
                commit["files_changed"],
                commit["additions"],
                commit["deletions"],
            ]
        )
        for file_change in commit["files"]:
            commit_file_rows.append(
                [
                    commit["commit"],
                    commit["short_commit"],
                    commit["date"],
                    commit["subject"],
                    file_change["path"],
                    file_change["added"],
                    file_change["deleted"],
                    file_change["added_raw"],
                    file_change["deleted_raw"],
                ]
            )

    for row in file_summary:
        file_summary_rows.append(
            [
                row["path"],
                row["commits_touched"],
                row["total_added"],
                row["total_deleted"],
                row["last_commit"],
                row["last_date"],
                row["last_subject"],
            ]
        )

    for row in worktree_rows:
        worktree_sheet.append(
            [
                row["status"],
                row["path"],
                row["added"],
                row["deleted"],
                row["added_raw"],
                row["deleted_raw"],
            ]
        )

    return {
        "MajorChanges": major_change_rows,
        "ChangeResults": change_results_rows,
        "CommitSummary": commit_rows,
        "CommitFiles": commit_file_rows,
        "FileSummary": file_summary_rows,
        "CurrentWorktree": worktree_sheet,
        "Notes": notes_rows,
    }


def write_csv_summary(rows: list[list[Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Git repository root.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/reports/aeromixer_changes.xlsx"),
        help="Path to the generated .xlsx workbook.",
    )
    parser.add_argument(
        "--csv-summary",
        type=Path,
        default=Path("output/reports/aeromixer_changes_summary.csv"),
        help="Path to the generated CSV commit summary.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    commits = parse_commit_history(repo_root)
    runs = discover_runs((repo_root / "output").resolve())
    worktree_rows = parse_current_worktree(repo_root)
    file_summary = build_file_summary(commits)
    sheets = build_sheets(repo_root, commits, runs, file_summary, worktree_rows)

    write_xlsx(sheets, args.report.resolve())
    write_csv_summary(sheets["CommitSummary"], args.csv_summary.resolve())

    print(f"Commits exported: {len(commits)}")
    print(f"Workbook: {args.report.resolve()}")
    print(f"CSV summary: {args.csv_summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
