import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset")
PRESET_TO_CONFIG = {
    "lite": ROOT / "config_files" / "presets" / "lite.yaml",
    "full": ROOT / "config_files" / "presets" / "full.yaml",
    "prod": ROOT / "config_files" / "presets" / "prod.yaml",
}


def _default_workers():
    return 0 if os.name == "nt" else 2


def _default_dataset():
    if DEFAULT_DATASET.exists():
        return str(DEFAULT_DATASET)
    return ""


def _require_dataset(path_str):
    dataset = Path(path_str or "").expanduser()
    if not dataset:
        raise SystemExit(
            "Dataset path is required. Use --data or create " f"{DEFAULT_DATASET}."
        )
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")
    return str(dataset)


def _run(command):
    printable = " ".join(str(part) for part in command)
    print(f">> {printable}")
    subprocess.run(command, cwd=str(ROOT), check=True)


def _build_pipeline_command(args, mode):
    extra_opts = []
    command = [
        sys.executable,
        str(ROOT / "scripts" / "pipeline.py"),
        "--mode",
        str(mode),
        "--data",
        _require_dataset(args.data),
        "--preset",
        args.preset,
        "--output-dir",
        args.output_dir,
        "--num-workers",
        str(args.num_workers),
    ]
    if hasattr(args, "epochs"):
        command.extend(["--epochs", str(args.epochs)])
    if hasattr(args, "batch_size"):
        command.extend(["--batch-size", str(args.batch_size)])
    if hasattr(args, "annotation_format") and args.annotation_format:
        extra_opts.extend(["DATA.ANNOTATION_FORMAT", args.annotation_format])

    if hasattr(args, "tile_size") and int(args.tile_size) > 0:
        command.extend(
            [
                "--tile-size",
                str(args.tile_size),
                "--tile-overlap",
                str(args.tile_overlap),
                "--tile-min-cover",
                str(args.tile_min_cover),
            ]
        )
    if getattr(args, "tune_thresholds", False):
        command.extend(
            [
                "--tune-thresholds",
                "--threshold-grid",
                args.threshold_grid,
            ]
        )
    if getattr(args, "skip_val_in_train", False):
        command.append("--skip-val-in-train")
    if getattr(args, "resume", False):
        command.append("--resume")
    if str(mode) == "train":
        command.append("--skip-final-test")
    if getattr(args, "extra_opts", None):
        extra_opts.extend(args.extra_opts)
    if extra_opts:
        command.extend(["--extra-opts", *extra_opts])
    return command


def _build_visualize_command(args):
    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    output_path = (
        Path(args.output)
        if args.output
        else ROOT / "output" / "visuals" / f"{image_path.stem}_pred.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(ROOT / "demo_image.py"),
        "--config-file",
        str(PRESET_TO_CONFIG[args.preset]),
        "--dataset-root",
        _require_dataset(args.data),
        "--image",
        str(image_path),
        "--output",
        str(output_path),
        "--device",
        args.device,
        "--topk",
        str(args.topk),
    ]
    if args.ckpt:
        command.extend(["--ckpt", args.ckpt])
    if args.score_threshold is not None:
        command.extend(["--score-threshold", str(args.score_threshold)])
    return command


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Simple AeroMixer wrapper: smoke, train, eval, or visualize."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Quick health-check training run.")
    smoke.add_argument("--data", default=_default_dataset())
    smoke.add_argument("--preset", choices=["lite", "full", "prod"], default="lite")
    smoke.add_argument("--output-dir", default="output/aero_smoke")
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--batch-size", type=int, default=1)
    smoke.add_argument("--num-workers", type=int, default=_default_workers())
    smoke.add_argument(
        "--annotation-format",
        choices=["auto", "yolo", "custom_json"],
        default="auto",
    )
    smoke.add_argument("--tile-size", type=int, default=0)
    smoke.add_argument("--tile-overlap", type=float, default=0.25)
    smoke.add_argument("--tile-min-cover", type=float, default=0.35)
    smoke.add_argument("--skip-val-in-train", action="store_true")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[])

    train = sub.add_parser(
        "train", help="Main training run with epoch validation only."
    )
    train.add_argument("--data", default=_default_dataset())
    train.add_argument("--preset", choices=["lite", "full", "prod"], default="prod")
    train.add_argument("--output-dir", default="output/aero_train")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--num-workers", type=int, default=_default_workers())
    train.add_argument(
        "--annotation-format",
        choices=["auto", "yolo", "custom_json"],
        default="custom_json",
    )
    train.add_argument("--tile-size", type=int, default=0)
    train.add_argument("--tile-overlap", type=float, default=0.25)
    train.add_argument("--tile-min-cover", type=float, default=0.35)
    train.add_argument("--skip-val-in-train", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[])

    eval_parser = sub.add_parser(
        "eval", help="Final test evaluation for an existing training output."
    )
    eval_parser.add_argument("--data", default=_default_dataset())
    eval_parser.add_argument(
        "--preset", choices=["lite", "full", "prod"], default="prod"
    )
    eval_parser.add_argument("--output-dir", default="output/aero_train")
    eval_parser.add_argument("--batch-size", type=int, default=2)
    eval_parser.add_argument("--num-workers", type=int, default=_default_workers())
    eval_parser.add_argument(
        "--annotation-format",
        choices=["auto", "yolo", "custom_json"],
        default="custom_json",
    )
    eval_parser.add_argument("--tile-size", type=int, default=0)
    eval_parser.add_argument("--tile-overlap", type=float, default=0.25)
    eval_parser.add_argument("--tile-min-cover", type=float, default=0.35)
    eval_parser.add_argument(
        "--tune-thresholds", dest="tune_thresholds", action="store_true"
    )
    eval_parser.add_argument(
        "--no-tune-thresholds", dest="tune_thresholds", action="store_false"
    )
    eval_parser.set_defaults(tune_thresholds=True)
    eval_parser.add_argument("--threshold-grid", default="0.05,0.1,0.2,0.3")
    eval_parser.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[])

    vis = sub.add_parser("vis", help="Visualize one image with boxes and JSON text.")
    vis.add_argument("--data", default=_default_dataset())
    vis.add_argument("--preset", choices=["lite", "full", "prod"], default="prod")
    vis.add_argument("--image", required=True)
    vis.add_argument("--ckpt", default="output/aero_train/checkpoints/model_final.pth")
    vis.add_argument("--output", default=None)
    vis.add_argument("--device", default="cuda")
    vis.add_argument("--topk", type=int, default=5)
    vis.add_argument("--score-threshold", type=float, default=None)

    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "smoke":
        _run(_build_pipeline_command(args, mode="train"))
        return
    if args.command == "train":
        _run(_build_pipeline_command(args, mode="train"))
        return
    if args.command == "eval":
        _run(_build_pipeline_command(args, mode="eval"))
        return
    if args.command == "vis":
        _run(_build_visualize_command(args))
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
