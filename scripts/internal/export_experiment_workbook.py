"""Export a workbook summarizing local AeroMixer experiment artifacts.

The exporter scans the workspace `output/` tree for:
- pipeline/inference manifests
- evaluation logs (`result_image.log`)
- training summaries (`train_metrics_final.json`)
- timestamped root logs (`YYYY-MM-DD_HH.MM.SS.log`)

It then writes a dependency-free `.xlsx` workbook with summary sheets.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


TIMESTAMP_LOG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}\.log$")
NP_FLOAT_RE = re.compile(r"np\.float64\(([^)]+)\)")
SPACE_RE = re.compile(r"\s+")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def maybe_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return value
        try:
            return int(raw)
        except Exception:
            pass
        try:
            return float(raw)
        except Exception:
            return value
    return value


def infer_run_root(path: Path) -> Path | None:
    if path.name in {
        "pipeline_manifest.json",
        "inference_manifest.json",
        "dataset_validation.json",
    }:
        return path.parent
    if path.name == "train_metrics_final.json":
        return path.parent.parent
    if path.name == "result_image.log":
        if len(path.parents) >= 3:
            return path.parents[2]
        return None
    if TIMESTAMP_LOG_RE.match(path.name):
        return path.parent
    return None


def extract_literal_dict(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    cleaned = NP_FLOAT_RE.sub(r"\1", text[start : end + 1])
    cleaned = cleaned.replace("nan", "None")
    try:
        value = ast.literal_eval(cleaned)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def extract_eval_metrics_from_result_log(path: Path) -> dict[str, Any]:
    return extract_literal_dict(read_text(path))


def find_first(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def parse_root_log(path: Path) -> dict[str, Any]:
    text = read_text(path)
    fields: dict[str, Any] = {
        "source_log": str(path),
        "log_timestamp": path.stem.replace("_", " "),
    }

    fields["config_file"] = find_first(
        [r"Namespace\(config_file='([^']+)'", r"Config file:\s*(.+)$"],
        text,
    )
    fields["output_dir"] = find_first(
        [r"Output dir:\s*(.+)$", r"OUTPUT_DIR:\s*(.+)$"], text
    )
    fields["dataset_source"] = find_first(
        [r"data_dir=([^,\n]+)", r"PATH_TO_DATA_DIR:\s*(.+)$"],
        text,
    )
    fields["annotation_format"] = find_first(
        [r"annotation_format=([^,\n]+)", r"ANNOTATION_FORMAT:\s*(.+)$"],
        text,
    )
    fields["epochs"] = maybe_number(
        find_first([r"epochs=(\d+)", r"MAX_EPOCH:\s*([0-9.]+)$"], text)
    )
    fields["batch_size"] = maybe_number(
        find_first([r"batch=(\d+)", r"IMAGES_PER_BATCH:\s*([0-9.]+)$"], text)
    )
    fields["backbone"] = find_first(
        [r"backbone=([^,\n]+)", r'CONV_BODY:\s*"?(.*?)"?\s*$'],
        text,
    )
    fields["text_encoder"] = find_first([r"TEXT_ENCODER:\s*(.+)$"], text)

    last_iter = None
    max_iter = None
    last_loss = None
    last_ce = None
    last_bbox = None
    last_giou = None
    for match in re.finditer(r"iter:\s*(\d+)/(\d+).*?total_loss:\s*([0-9.]+)", text):
        last_iter = maybe_number(match.group(1))
        max_iter = maybe_number(match.group(2))
        last_loss = maybe_number(match.group(3))
    for metric_name, bucket in [
        ("loss_ce", "last_loss_ce"),
        ("loss_bbox", "last_loss_bbox"),
        ("loss_giou", "last_loss_giou"),
    ]:
        metric_matches = re.findall(rf"{metric_name}:\s*([0-9.]+)", text)
        if metric_matches:
            fields[bucket] = maybe_number(metric_matches[-1])
    if last_iter is not None:
        fields["final_iteration"] = last_iter
    if max_iter is not None:
        fields["max_iter"] = max_iter
    if last_loss is not None:
        fields["total_loss"] = last_loss
    if "last_loss_ce" in fields:
        last_ce = fields["last_loss_ce"]
    if "last_loss_bbox" in fields:
        last_bbox = fields["last_loss_bbox"]
    if "last_loss_giou" in fields:
        last_giou = fields["last_loss_giou"]
    if last_ce is not None:
        fields["loss_ce"] = last_ce
    if last_bbox is not None:
        fields["loss_bbox"] = last_bbox
    if last_giou is not None:
        fields["loss_giou"] = last_giou

    total_training_time = find_first([r"Total training time:\s*([^\n]+)$"], text)
    if total_training_time is not None:
        fields["training_time"] = total_training_time

    eval_anchor = text.rfind("Evaluation results")
    if eval_anchor >= 0:
        eval_text = text[eval_anchor:]
        metrics = extract_literal_dict(eval_text)
        if metrics:
            fields["eval_metrics"] = metrics

    return fields


def merge_if_missing(dst: dict[str, Any], src: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if dst.get(key) in (None, "", []):
            value = src.get(key)
            if value not in (None, "", []):
                dst[key] = value


def parse_eval_metrics(
    root: Path, manifest: dict[str, Any] | None
) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    if manifest and isinstance(manifest.get("eval_metrics"), dict):
        for dataset_name, payload in manifest["eval_metrics"].items():
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            eval_rows.append(
                {
                    "dataset": dataset_name,
                    "metrics": metrics,
                    "source": (
                        payload.get("log_file") if isinstance(payload, dict) else ""
                    ),
                    "map50": (
                        payload.get(
                            "map50", metrics.get("PascalBoxes_Precision/mAP@0.5IOU")
                        )
                        if isinstance(payload, dict)
                        else metrics.get("PascalBoxes_Precision/mAP@0.5IOU")
                    ),
                    "map5095": (
                        payload.get(
                            "map5095",
                            metrics.get("PascalBoxes_Precision/mAP@0.5:0.95IOU"),
                        )
                        if isinstance(payload, dict)
                        else metrics.get("PascalBoxes_Precision/mAP@0.5:0.95IOU")
                    ),
                    "small_ap": (
                        payload.get("small_ap", metrics.get("SmallObject/AP@0.5IOU"))
                        if isinstance(payload, dict)
                        else metrics.get("SmallObject/AP@0.5IOU")
                    ),
                }
            )

    if eval_rows:
        return eval_rows

    for result_log in sorted(root.glob("inference/*/result_image.log")):
        metrics = extract_eval_metrics_from_result_log(result_log)
        if not metrics:
            continue
        eval_rows.append(
            {
                "dataset": result_log.parent.name,
                "metrics": metrics,
                "source": str(result_log),
                "map50": metrics.get("PascalBoxes_Precision/mAP@0.5IOU"),
                "map5095": metrics.get("PascalBoxes_Precision/mAP@0.5:0.95IOU"),
                "small_ap": metrics.get("SmallObject/AP@0.5IOU"),
            }
        )

    return eval_rows


def build_run_record(root: Path, output_root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_name": root.relative_to(output_root).as_posix(),
        "root_path": str(root.resolve()),
        "created_at_utc": "",
        "git_commit": "",
        "mode": "",
        "preset": "",
        "config_file": "",
        "dataset_source": "",
        "dataset_name": "",
        "annotation_format": "",
        "num_classes": "",
        "epochs": "",
        "batch_size": "",
        "num_workers": "",
        "backbone": "",
        "text_encoder": "",
        "dry_run": "",
        "tiling_enabled": "",
        "tile_size": "",
        "tile_overlap": "",
        "threshold_tuning": "",
        "threshold_grid": "",
        "validation_ok": "",
        "eval_dataset_count": 0,
        "status": "artifact-only",
        "artifact_count": 0,
        "top_log_count": 0,
    }

    manifest = None
    for manifest_name in ("pipeline_manifest.json", "inference_manifest.json"):
        manifest_path = root / manifest_name
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            record["manifest_path"] = str(manifest_path)
            break

    if manifest:
        record["created_at_utc"] = manifest.get("created_at_utc", "")
        record["git_commit"] = manifest.get("git_commit", "")
        record["mode"] = manifest.get("mode", "")
        record["preset"] = manifest.get("preset", "")
        record["config_file"] = manifest.get("config_file", "")
        record["dataset_source"] = manifest.get("dataset_source", "")
        dataset_plan = manifest.get("dataset_plan", {}) or {}
        record["dataset_name"] = (
            ",".join(manifest.get("eval_metrics", {}).keys())
            if manifest.get("eval_metrics")
            else ""
        )
        record["annotation_format"] = dataset_plan.get("annotation_format", "")
        record["num_classes"] = dataset_plan.get("num_classes", "")
        record["epochs"] = manifest.get("epochs", "")
        record["batch_size"] = manifest.get("batch_size", "")
        record["num_workers"] = manifest.get("num_workers", "")
        record["dry_run"] = manifest.get("dry_run", "")
        tiling = manifest.get("tiling", {}) or {}
        record["tiling_enabled"] = tiling.get("enabled", "")
        record["tile_size"] = tiling.get("tile_size", "")
        record["tile_overlap"] = tiling.get("tile_overlap", "")
        threshold_tuning = manifest.get("threshold_tuning", {}) or {}
        record["threshold_tuning"] = threshold_tuning.get("enabled", "")
        record["threshold_grid"] = threshold_tuning.get("threshold_grid", "")
        validation = manifest.get("validation", {}) or {}
        record["validation_ok"] = validation.get("ok", "")

    top_logs = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and TIMESTAMP_LOG_RE.match(path.name)
    )
    record["top_log_count"] = len(top_logs)
    if top_logs:
        parsed_log = parse_root_log(top_logs[-1])
        merge_if_missing(
            record,
            parsed_log,
            [
                "config_file",
                "dataset_source",
                "annotation_format",
                "epochs",
                "batch_size",
                "backbone",
                "text_encoder",
            ],
        )
        if not record["created_at_utc"]:
            record["created_at_utc"] = top_logs[0].stem.replace("_", " ")
        if not record.get("output_dir"):
            record["output_dir"] = parsed_log.get("output_dir", "")
        for key in (
            "final_iteration",
            "max_iter",
            "total_loss",
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "training_time",
        ):
            if parsed_log.get(key) not in (None, ""):
                record[key] = parsed_log[key]
        if not record["dataset_name"] and parsed_log.get("dataset_source"):
            record["dataset_name"] = Path(str(parsed_log["dataset_source"])).name
        if "eval_metrics" in parsed_log and not manifest:
            manifest = {
                "eval_metrics": {
                    record["dataset_name"]
                    or "eval": {"metrics": parsed_log["eval_metrics"]}
                }
            }

    train_metrics_path = root / "inference" / "train_metrics_final.json"
    if train_metrics_path.exists():
        train_metrics = load_json(train_metrics_path)
        record["train_metrics_path"] = str(train_metrics_path)
        for key in (
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "total_loss",
            "time",
            "data",
            "max_iter",
            "final_iteration",
        ):
            if key in train_metrics:
                record[key] = train_metrics[key]

    eval_rows = parse_eval_metrics(root, manifest)
    record["eval_rows"] = eval_rows
    record["eval_dataset_count"] = len(eval_rows)
    if len(eval_rows) == 1 and not record["dataset_name"]:
        record["dataset_name"] = eval_rows[0]["dataset"]
    if eval_rows:
        primary = eval_rows[0]
        metrics = primary["metrics"]
        record["map50"] = primary.get(
            "map50", metrics.get("PascalBoxes_Precision/mAP@0.5IOU")
        )
        record["map5095"] = primary.get(
            "map5095", metrics.get("PascalBoxes_Precision/mAP@0.5:0.95IOU")
        )
        record["small_ap"] = primary.get(
            "small_ap", metrics.get("SmallObject/AP@0.5IOU")
        )
        record["medium_ap"] = metrics.get("Area/medium/mAP@0.5IOU")
        record["large_ap"] = metrics.get("Area/large/mAP@0.5IOU")
        record["status"] = "evaluated"
    elif any(
        record.get(key) not in (None, "", 0)
        for key in ("total_loss", "final_iteration", "max_iter")
    ):
        record["status"] = "trained-only"

    if record["dry_run"] is True:
        record["status"] = "dry-run"

    artifact_files = []
    for rel in [
        "pipeline_manifest.json",
        "inference_manifest.json",
        "dataset_validation.json",
        "inference/train_metrics_final.json",
    ]:
        path = root / rel
        if path.exists():
            artifact_files.append(str(path.resolve()))
    artifact_files.extend(str(path.resolve()) for path in top_logs)
    artifact_files.extend(
        str(path.resolve()) for path in root.glob("inference/*/result_image.log")
    )
    record["artifact_count"] = len(artifact_files)
    record["artifact_files"] = artifact_files

    return record


def discover_runs(output_root: Path) -> list[dict[str, Any]]:
    roots: set[Path] = set()
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        root = infer_run_root(path)
        if root is not None and root.exists():
            roots.add(root)
    return [build_run_record(root, output_root) for root in sorted(roots)]


def build_sheets(runs: list[dict[str, Any]]) -> dict[str, list[list[Any]]]:
    experiments_rows = [
        [
            "run_name",
            "root_path",
            "created_at_utc",
            "status",
            "preset",
            "mode",
            "config_file",
            "dataset_name",
            "dataset_source",
            "annotation_format",
            "num_classes",
            "epochs",
            "batch_size",
            "num_workers",
            "backbone",
            "text_encoder",
            "dry_run",
            "tiling_enabled",
            "tile_size",
            "tile_overlap",
            "threshold_tuning",
            "threshold_grid",
            "validation_ok",
            "map50",
            "map5095",
            "small_ap",
            "medium_ap",
            "large_ap",
            "total_loss",
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "final_iteration",
            "max_iter",
            "git_commit",
            "artifact_count",
            "top_log_count",
        ]
    ]
    eval_summary_rows = [
        [
            "run_name",
            "dataset",
            "source",
            "map50",
            "map5095",
            "small_ap",
            "medium_ap",
            "large_ap",
            "small_num_det",
            "medium_num_det",
            "large_num_det",
        ]
    ]
    eval_metrics_rows = [["run_name", "dataset", "metric", "value", "source"]]
    category_rows = [["run_name", "dataset", "category", "ap50", "source"]]
    train_rows = [
        [
            "run_name",
            "train_metrics_path",
            "total_loss",
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "time",
            "data",
            "final_iteration",
            "max_iter",
            "training_time",
        ]
    ]
    artifacts_rows = [["run_name", "artifact_path"]]
    notes_rows = [
        ["generated_at_utc", datetime.now(timezone.utc).isoformat()],
        ["workspace_root", str(Path.cwd().resolve())],
        ["output_root", str((Path.cwd() / "output").resolve())],
        ["included_runs", len(runs)],
        [
            "note",
            "Only artifacts present in the local workspace were exported. Chat-only Colab logs or runs not saved under output/ are not included.",
        ],
    ]

    for run in runs:
        experiments_rows.append(
            [
                run.get("run_name"),
                run.get("root_path"),
                run.get("created_at_utc"),
                run.get("status"),
                run.get("preset"),
                run.get("mode"),
                run.get("config_file"),
                run.get("dataset_name"),
                run.get("dataset_source"),
                run.get("annotation_format"),
                run.get("num_classes"),
                run.get("epochs"),
                run.get("batch_size"),
                run.get("num_workers"),
                run.get("backbone"),
                run.get("text_encoder"),
                run.get("dry_run"),
                run.get("tiling_enabled"),
                run.get("tile_size"),
                run.get("tile_overlap"),
                run.get("threshold_tuning"),
                run.get("threshold_grid"),
                run.get("validation_ok"),
                run.get("map50"),
                run.get("map5095"),
                run.get("small_ap"),
                run.get("medium_ap"),
                run.get("large_ap"),
                run.get("total_loss"),
                run.get("loss_ce"),
                run.get("loss_bbox"),
                run.get("loss_giou"),
                run.get("final_iteration"),
                run.get("max_iter"),
                run.get("git_commit"),
                run.get("artifact_count"),
                run.get("top_log_count"),
            ]
        )

        if any(
            run.get(key) not in (None, "", [])
            for key in ("train_metrics_path", "total_loss", "final_iteration")
        ):
            train_rows.append(
                [
                    run.get("run_name"),
                    run.get("train_metrics_path"),
                    run.get("total_loss"),
                    run.get("loss_ce"),
                    run.get("loss_bbox"),
                    run.get("loss_giou"),
                    run.get("time"),
                    run.get("data"),
                    run.get("final_iteration"),
                    run.get("max_iter"),
                    run.get("training_time"),
                ]
            )

        for artifact_path in run.get("artifact_files", []):
            artifacts_rows.append([run.get("run_name"), artifact_path])

        for eval_row in run.get("eval_rows", []):
            metrics = eval_row.get("metrics", {}) or {}
            eval_summary_rows.append(
                [
                    run.get("run_name"),
                    eval_row.get("dataset"),
                    eval_row.get("source"),
                    eval_row.get(
                        "map50", metrics.get("PascalBoxes_Precision/mAP@0.5IOU")
                    ),
                    eval_row.get(
                        "map5095", metrics.get("PascalBoxes_Precision/mAP@0.5:0.95IOU")
                    ),
                    eval_row.get("small_ap", metrics.get("SmallObject/AP@0.5IOU")),
                    metrics.get("Area/medium/mAP@0.5IOU"),
                    metrics.get("Area/large/mAP@0.5IOU"),
                    metrics.get("Area/small/num_det"),
                    metrics.get("Area/medium/num_det"),
                    metrics.get("Area/large/num_det"),
                ]
            )
            for key, value in sorted(metrics.items()):
                eval_metrics_rows.append(
                    [
                        run.get("run_name"),
                        eval_row.get("dataset"),
                        key,
                        value,
                        eval_row.get("source"),
                    ]
                )
                prefix = "PascalBoxes_PerformanceByCategory/AP@0.5IOU/"
                if key.startswith(prefix):
                    category_rows.append(
                        [
                            run.get("run_name"),
                            eval_row.get("dataset"),
                            key[len(prefix) :],
                            value,
                            eval_row.get("source"),
                        ]
                    )

    return {
        "Experiments": experiments_rows,
        "EvalSummary": eval_summary_rows,
        "EvalMetrics": eval_metrics_rows,
        "CategoryAP": category_rows,
        "TrainSummary": train_rows,
        "Artifacts": artifacts_rows,
        "Notes": notes_rows,
    }


def excel_col(index: int) -> str:
    letters = []
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def xml_cell(cell_ref: str, value: Any, header: bool = False) -> str:
    style = ' s="1"' if header else ""
    if value is None:
        return f'<c r="{cell_ref}"{style}/>'
    if isinstance(value, bool):
        return (
            f'<c r="{cell_ref}" t="inlineStr"{style}><is><t>{str(value)}</t></is></c>'
        )
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f'<c r="{cell_ref}"{style}><v>{value}</v></c>'
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = escape(text)
    preserve = ' xml:space="preserve"' if text != text.strip() or "\n" in text else ""
    return (
        f'<c r="{cell_ref}" t="inlineStr"{style}><is><t{preserve}>{text}</t></is></c>'
    )


def make_sheet_xml(rows: list[list[Any]]) -> str:
    row_xml = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            cell_ref = f"{excel_col(col_idx)}{row_idx}"
            cells.append(xml_cell(cell_ref, value, header=(row_idx == 1)))
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews>'
        "<sheetData>" + "".join(row_xml) + "</sheetData></worksheet>"
    )


def write_xlsx(sheets: dict[str, list[list[Any]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_items = list(sheets.items())

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(1, len(sheet_items) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx, (name, _) in enumerate(sheet_items, start=1):
        workbook_xml.append(
            f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    style_rel_id = len(sheet_items) + 1
    workbook_rels.append(
        f'<Relationship Id="rId{style_rel_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_xml.append("</sheets></workbook>")
    workbook_rels.append("</Relationships>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        "</fonts>"
        '<fills count="2">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        "</fills>"
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    now = datetime.now(timezone.utc).isoformat()
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        "<dc:title>AeroMixer Experiment Report</dc:title>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )

    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Codex</Application>"
        "</Properties>"
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", "".join(workbook_xml))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        for idx, (_, rows) in enumerate(sheet_items, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", make_sheet_xml(rows))


def write_csv_summary(experiments_rows: list[list[Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(experiments_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Workspace output root to scan.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/reports/aeromixer_experiments.xlsx"),
        help="Path to the generated .xlsx workbook.",
    )
    parser.add_argument(
        "--csv-summary",
        type=Path,
        default=Path("output/reports/aeromixer_experiments_summary.csv"),
        help="Path to the generated CSV summary sheet.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    runs = discover_runs(output_root)
    sheets = build_sheets(runs)
    write_xlsx(sheets, args.report.resolve())
    write_csv_summary(sheets["Experiments"], args.csv_summary.resolve())

    print(f"Runs exported: {len(runs)}")
    print(f"Workbook: {args.report.resolve()}")
    print(f"CSV summary: {args.csv_summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
