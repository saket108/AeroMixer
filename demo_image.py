import argparse
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch

import alphaction.dataset.datasets.utils as utils
from alphaction.config import auto_sync_dataset_class_counts, cfg, uses_text_branch
from alphaction.dataset.datasets.cv2_transform import PreprocessWithBoxes
from alphaction.dataset.datasets.image_dataset import ImageDataset
from alphaction.engine.inference import _normalize_batch_outputs
from alphaction.modeling.detector import build_detection_model
from alphaction.modeling.runtime import configure_text_encoder
from alphaction.utils.checkpoint import ActionCheckpointer
from alphaction.utils.visualize import annotate


def _normalize_path(path):
    return str(path).replace("\\", "/")


def _match_dataset_sample(dataset, image_path):
    image_abs = os.path.normcase(os.path.abspath(str(image_path)))
    image_base = os.path.basename(str(image_path)).lower()

    for sample in getattr(dataset, "samples", []):
        image_rel = str(sample.get("image_rel", ""))
        sample_abs = os.path.normcase(
            os.path.abspath(os.path.join(dataset.image_dir, image_rel))
        )
        if sample_abs == image_abs:
            return sample
        if os.path.basename(image_rel).lower() == image_base:
            return sample
    return None


def _load_dataset_context(config, image_path):
    first_context = None

    for split in ("train", "val", "test"):
        try:
            dataset = ImageDataset(config, split)
        except Exception:
            continue

        vocabulary = None
        if getattr(dataset, "text_input", None) is not None:
            vocabulary = dataset.text_input.get("closed", None)

        sample = _match_dataset_sample(dataset, image_path)
        context = {
            "split": split,
            "class_names": list(getattr(dataset, "class_names", [])),
            "vocabulary": vocabulary,
            "annotation_extras": (
                list(sample.get("annotation_extras", []))
                if isinstance(sample, dict)
                and isinstance(sample.get("annotation_extras", None), list)
                else []
            ),
            "image_rel": (
                str(sample.get("image_rel", "")) if isinstance(sample, dict) else ""
            ),
        }

        if first_context is None:
            first_context = context
        if sample is not None:
            return context

    return first_context


def _make_preprocess(config):
    dataset_cfg = SimpleNamespace(
        BGR=False,
        TRAIN_USE_COLOR_AUGMENTATION=False,
        TRAIN_PCA_JITTER_ONLY=False,
        TRAIN_PCA_EIGVAL=[0.225, 0.224, 0.229],
        TRAIN_PCA_EIGVEC=[
            [0.467, 0.485, 0.456],
            [0.229, 0.224, 0.225],
            [0.197, 0.199, 0.201],
        ],
        TEST_FORCE_FLIP=False,
    )
    return PreprocessWithBoxes("test", config.DATA, dataset_cfg)


def _resolve_checkpoint(config, explicit_ckpt):
    if explicit_ckpt:
        return explicit_ckpt
    if config.MODEL.WEIGHT:
        return os.path.join(config.OUTPUT_DIR, config.MODEL.WEIGHT)
    return None


def _collect_vocab(model, config, dataset_context):
    if dataset_context and dataset_context.get("vocabulary"):
        configure_text_encoder(model, dataset_context["vocabulary"])
    if (
        uses_text_branch(config)
        and hasattr(model, "backbone")
        and hasattr(model.backbone, "text_encoder")
        and hasattr(model.backbone.text_encoder, "text_data")
    ):
        try:
            return list(model.backbone.text_encoder.text_data.keys())
        except Exception:
            return None
    return None


def _run_single_image_inference(model, config, image_path, device):
    preproc = _make_preprocess(config)
    imgs = utils.retry_load_images([image_path], backend="cv2")
    imgs_proc, _ = preproc.process(imgs, boxes=None)
    imgs_packed = utils.pack_pathway_output(
        config, imgs_proc, pathways=config.MODEL.BACKBONE.PATHWAYS
    )
    if config.MODEL.BACKBONE.PATHWAYS == 1:
        primary_input = imgs_packed[0][None].to(device)
        secondary_input = None
    else:
        primary_input = imgs_packed[0][None].to(device)
        secondary_input = imgs_packed[1][None].to(device)

    height, width = primary_input.shape[-2:]
    whwh = torch.tensor([[width, height, width, height]], dtype=torch.float32).to(
        device
    )

    with torch.no_grad():
        outputs = model(primary_input, secondary_input, whwh)

    return (
        imgs[0],
        _normalize_batch_outputs(
            outputs,
            batch_size=1,
            num_classes=max(1, int(getattr(config.MODEL.STM, "OBJECT_CLASSES", 1))),
            device=device,
        )[0],
    )


def _wrap_lines(text, width=46):
    text = str(text).strip()
    if not text:
        return []
    return textwrap.wrap(text, width=width) or [text]


def _format_json_description_lines(annotation_extras, max_items=6):
    lines = []
    for idx, item in enumerate(annotation_extras[:max_items], start=1):
        if not isinstance(item, dict):
            continue
        category_name = str(item.get("category_name", "object")).strip() or "object"
        zone = str(item.get("zone_estimation", "")).strip()
        header = f"{idx}. {category_name}"
        if zone:
            header += f" [{zone}]"
        lines.append(header)

        damage_metrics = item.get("damage_metrics", None)
        if isinstance(damage_metrics, dict):
            severity = damage_metrics.get("raw_severity_score", None)
            if severity is not None:
                try:
                    lines.append(f"   raw_severity={float(severity):.3f}")
                except Exception:
                    pass

        description = item.get("description", None)
        for wrapped in _wrap_lines(description, width=48):
            lines.append(f"   {wrapped}")
    return lines


def _draw_info_panel(
    image_bgr,
    image_path,
    dataset_context,
    prediction_lines,
    panel_width=440,
):
    image_height = int(image_bgr.shape[0])
    panel = (
        255 * torch.ones((image_height, int(panel_width), 3), dtype=torch.uint8).numpy()
    )

    x = 16
    y = 28
    line_h = 20

    def add_line(text, color=(20, 20, 20), scale=0.5, thickness=1):
        nonlocal y
        if y > image_height - 12:
            return
        cv2.putText(
            panel,
            str(text),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

    add_line("AeroMixer Prediction", color=(10, 10, 10), scale=0.7, thickness=2)
    add_line(f"image: {os.path.basename(str(image_path))}")
    if dataset_context:
        add_line(f"dataset split: {dataset_context.get('split', '')}")
        image_rel = str(dataset_context.get("image_rel", "")).strip()
        if image_rel:
            add_line(f"dataset rel: {image_rel}")

    y += 8
    add_line("Predictions", color=(0, 80, 0), scale=0.6, thickness=2)
    if prediction_lines:
        for line in prediction_lines:
            for wrapped in _wrap_lines(line, width=44):
                add_line(wrapped)
    else:
        add_line("No detections above threshold.")

    y += 8
    add_line("JSON Descriptions", color=(120, 40, 0), scale=0.6, thickness=2)
    description_lines = _format_json_description_lines(
        dataset_context.get("annotation_extras", []) if dataset_context else []
    )
    if description_lines:
        for line in description_lines:
            add_line(line)
    else:
        add_line("No JSON description found for this image.")

    return cv2.hconcat([image_bgr, panel])


def main():
    parser = argparse.ArgumentParser(
        description="Run single-image AeroMixer inference with optional JSON descriptions."
    )
    parser.add_argument("--config-file", required=True, help="config file")
    parser.add_argument("--image", required=True, help="path to input image")
    parser.add_argument(
        "--output", default="output/image_demo.png", help="annotated output path"
    )
    parser.add_argument(
        "--ckpt", default=None, help="checkpoint file (overrides cfg.MODEL.WEIGHT)"
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root used to load class names and JSON descriptions.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override score threshold for visualization.",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=440,
        help="Width of the right-side metadata panel in pixels.",
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Maximum detections to render."
    )
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument(
        "opts", help="Override config options", nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.dataset_root:
        cfg.DATA.PATH_TO_DATA_DIR = args.dataset_root
        cfg.DATA.FRAME_DIR = ""
    auto_sync_dataset_class_counts(cfg)
    cfg.freeze()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(
        args.device
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else "cpu"
    )

    model = build_detection_model(cfg)
    model.to(device)
    model.eval()

    ckpt_file = _resolve_checkpoint(cfg, args.ckpt)
    if ckpt_file and os.path.exists(ckpt_file):
        checkpointer = ActionCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(ckpt_file)
    else:
        print(
            "Warning: no checkpoint found — running with randomly initialized weights"
        )

    dataset_context = (
        _load_dataset_context(cfg, image_path) if args.dataset_root else None
    )
    vocab = _collect_vocab(model, cfg, dataset_context)

    original_bgr, (boxes_norm, score_matrix) = _run_single_image_inference(
        model, cfg, str(image_path), device
    )
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    if score_matrix.numel() == 0 or boxes_norm.numel() == 0:
        annotated_bgr = original_bgr.copy()
        prediction_lines = []
    else:
        best_scores, best_ids = score_matrix.max(dim=-1)
        score_threshold = (
            float(args.score_threshold)
            if args.score_threshold is not None
            else float(getattr(cfg.MODEL.STM, "SCORE_THRESHOLD", 0.05))
        )
        keep = torch.where(best_scores >= score_threshold)[0]
        if keep.numel() == 0 and best_scores.numel() > 0:
            keep = torch.topk(
                best_scores, k=min(int(args.topk), int(best_scores.numel()))
            ).indices

        keep_scores = best_scores[keep]
        keep_ids = best_ids[keep]
        keep_boxes = boxes_norm[keep]
        order = torch.argsort(keep_scores, descending=True)
        keep_scores = keep_scores[order][: int(args.topk)]
        keep_ids = keep_ids[order][: int(args.topk)]
        keep_boxes = keep_boxes[order][: int(args.topk)]

        phrases = []
        prediction_lines = []
        for det_idx, (score, class_id) in enumerate(
            zip(keep_scores.tolist(), keep_ids.tolist()), start=1
        ):
            if vocab is not None and int(class_id) < len(vocab):
                label = str(vocab[int(class_id)])
            else:
                label = f"class_{int(class_id)}"
            phrases.append(f"{label} {float(score):.2f}")
            prediction_lines.append(f"{det_idx}. {label} ({float(score):.3f})")

        annotated_bgr = annotate(
            image_rgb,
            keep_boxes,
            normalized=True,
            phrases=phrases,
            is_xyxy=True,
            set_text_color="white",
        )

    report_bgr = _draw_info_panel(
        annotated_bgr,
        image_path=image_path,
        dataset_context=dataset_context,
        prediction_lines=prediction_lines,
        panel_width=int(args.panel_width),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), report_bgr)
    print(f"Saved annotated image to {output_path}")
    if dataset_context and dataset_context.get("annotation_extras"):
        print(
            f"Loaded {len(dataset_context['annotation_extras'])} JSON annotation descriptions from split {dataset_context.get('split')}."
        )


if __name__ == "__main__":
    main()
