#!/usr/bin/env python3
"""
Build closed/open vocabulary files from generic detection annotations.

Supported annotation inputs:
- TXT lines with label as last token
  (works for both image and video txt formats used in this repo)
- JSON list or {"annotations": [...]} with one of:
  label | class | class_id | category_id
  and optional "categories" for id->name mapping.
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build closed/open vocab json files.")
    parser.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        help="One or more annotation files (.txt/.json).",
    )
    parser.add_argument(
        "--out-closed",
        default="vocab_closed.json",
        help="Output path for closed vocabulary json.",
    )
    parser.add_argument(
        "--out-open",
        default="vocab_open.json",
        help="Output path for open vocabulary json.",
    )
    parser.add_argument(
        "--out-combined",
        default="",
        help="Optional combined json with {'closed':..., 'open':...}.",
    )
    parser.add_argument(
        "--out-unseen",
        default="",
        help="Optional txt file for unseen class names.",
    )
    parser.add_argument(
        "--closed-ratio",
        type=float,
        default=0.8,
        help="Ratio of classes to keep in closed set when --closed-list is not provided.",
    )
    parser.add_argument(
        "--closed-list",
        default="",
        help="Optional txt file with one class name per line for closed set.",
    )
    parser.add_argument(
        "--prompt-template",
        default="{label}",
        help="Caption template. Example: 'a person is {label}'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling closed classes by ratio.",
    )
    return parser.parse_args()


def _safe_str(value):
    return str(value).strip()


def _extract_labels_from_txt(path):
    labels = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label = _safe_str(parts[-1])
            if label:
                labels.add(label)
    return labels


def _extract_labels_from_json(path):
    labels = set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_name_by_id = {}
    if isinstance(data, dict) and isinstance(data.get("categories"), list):
        for cat in data["categories"]:
            if not isinstance(cat, dict):
                continue
            cat_id = _safe_str(cat.get("id", ""))
            cat_name = _safe_str(cat.get("name", cat_id))
            if cat_id:
                category_name_by_id[cat_id] = cat_name

    annotations = data.get("annotations") if isinstance(data, dict) else data
    if not isinstance(annotations, list):
        return labels

    for rec in annotations:
        if not isinstance(rec, dict):
            continue
        raw_label = (
            rec.get("label")
            if rec.get("label") is not None
            else rec.get("class")
            if rec.get("class") is not None
            else rec.get("class_id")
            if rec.get("class_id") is not None
            else rec.get("category_id")
        )
        if raw_label is None:
            continue
        label = _safe_str(raw_label)
        label = category_name_by_id.get(label, label)
        if label:
            labels.add(label)
    return labels


def _read_closed_list(path):
    wanted = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wanted.append(line)
    return wanted


def _build_vocab_entries(labels, template):
    data = {}
    for label in labels:
        caption = template.format(label=label)
        data[label] = {"caption": caption}
    return data


def main():
    args = parse_args()

    labels_all = set()
    for ann_path in args.annotations:
        p = Path(ann_path)
        if not p.exists():
            raise FileNotFoundError(f"Annotation file not found: {p}")
        if p.suffix.lower() == ".txt":
            labels_all.update(_extract_labels_from_txt(p))
        elif p.suffix.lower() == ".json":
            labels_all.update(_extract_labels_from_json(p))
        else:
            raise ValueError(f"Unsupported annotation extension: {p.suffix} ({p})")

    labels_all = sorted(labels_all)
    if len(labels_all) == 0:
        raise RuntimeError("No class labels found in provided annotation files.")

    if args.closed_list:
        requested = _read_closed_list(args.closed_list)
        closed = [x for x in requested if x in labels_all]
        missing = [x for x in requested if x not in labels_all]
        if len(closed) == 0:
            raise RuntimeError("Closed list did not match any label from annotations.")
        if missing:
            print(f"[warn] {len(missing)} labels from closed-list were not found in annotations.")
    else:
        if not (0.0 < args.closed_ratio <= 1.0):
            raise ValueError("--closed-ratio must be in (0, 1].")
        if len(labels_all) == 1:
            closed = list(labels_all)
        elif args.closed_ratio >= 1.0:
            closed = list(labels_all)
        else:
            rng = random.Random(args.seed)
            pool = list(labels_all)
            rng.shuffle(pool)
            n_closed = int(round(len(pool) * args.closed_ratio))
            n_closed = max(1, min(len(pool) - 1, n_closed))
            closed = sorted(pool[:n_closed])

    open_set = list(labels_all)
    unseen = sorted([x for x in open_set if x not in set(closed)])

    closed_json = _build_vocab_entries(closed, args.prompt_template)
    open_json = _build_vocab_entries(open_set, args.prompt_template)

    out_closed = Path(args.out_closed)
    out_open = Path(args.out_open)
    out_closed.parent.mkdir(parents=True, exist_ok=True)
    out_open.parent.mkdir(parents=True, exist_ok=True)

    with open(out_closed, "w", encoding="utf-8") as f:
        json.dump(closed_json, f, indent=2, ensure_ascii=False)
    with open(out_open, "w", encoding="utf-8") as f:
        json.dump(open_json, f, indent=2, ensure_ascii=False)

    if args.out_combined:
        out_combined = Path(args.out_combined)
        out_combined.parent.mkdir(parents=True, exist_ok=True)
        with open(out_combined, "w", encoding="utf-8") as f:
            json.dump({"closed": closed_json, "open": open_json}, f, indent=2, ensure_ascii=False)

    if args.out_unseen:
        out_unseen = Path(args.out_unseen)
        out_unseen.parent.mkdir(parents=True, exist_ok=True)
        with open(out_unseen, "w", encoding="utf-8") as f:
            for name in unseen:
                f.write(f"{name}\n")

    print(f"[done] classes total: {len(open_set)}")
    print(f"[done] closed classes: {len(closed)}")
    print(f"[done] unseen classes: {len(unseen)}")
    print(f"[done] wrote closed vocab: {out_closed}")
    print(f"[done] wrote open vocab: {out_open}")


if __name__ == "__main__":
    main()
