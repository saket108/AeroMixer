import os
import argparse
import cv2
import numpy as np
import torch

from alphaction.config import cfg
from alphaction.modeling.detector import build_detection_model
from alphaction.utils.checkpoint import ActionCheckpointer
from alphaction.dataset.datasets.cv2_transform import PreprocessWithBoxes
import alphaction.dataset.datasets.utils as dataset_utils


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def _resolve_checkpoint_path(args_ckpt: str, cfg_obj) -> str:
    if args_ckpt:
        return args_ckpt
    weight = str(getattr(cfg_obj.MODEL, "WEIGHT", "")).strip()
    if not weight:
        return ""
    if os.path.isabs(weight):
        return weight
    return os.path.join(cfg_obj.OUTPUT_DIR, weight)


def _get_sequence_indices(center_pos: int, video_len: int, num_frames: int, sample_rate: int):
    half = (num_frames // 2) * sample_rate
    start = center_pos - half
    indices = []
    for i in range(num_frames):
        pos = start + i * sample_rate
        pos = max(0, min(video_len - 1, pos))
        indices.append(pos)
    return indices


def _draw_predictions(frame_bgr, boxes_norm, scores, topk=1, multilabel=False, vocab=None):
    h, w = frame_bgr.shape[:2]
    output = frame_bgr.copy()

    if boxes_norm is None or boxes_norm.numel() == 0:
        return output

    if multilabel:
        probs = torch.sigmoid(scores)
    else:
        probs = torch.softmax(scores, dim=-1)

    k = min(max(1, topk), probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

    for det_idx, box in enumerate(boxes_norm):
        x1 = int(float(box[0]) * w)
        y1 = int(float(box[1]) * h)
        x2 = int(float(box[2]) * w)
        y2 = int(float(box[3]) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        cls_id = int(top_ids[det_idx, 0].item())
        score = float(top_probs[det_idx, 0].item())
        cls_name = vocab[cls_id] if vocab is not None and cls_id < len(vocab) else f"class_{cls_id}"
        label = f"{cls_name} {score:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def main():
    parser = argparse.ArgumentParser(description="Run AeroMixer inference on a video and save annotated output.")
    parser.add_argument("--config-file", required=True, help="path to config yaml")
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--output", default="output/video_demo.mp4", help="output annotated video path")
    parser.add_argument("--ckpt", default=None, help="checkpoint path (overrides cfg.MODEL.WEIGHT)")
    parser.add_argument("--topk", type=int, default=1, help="top-k per detection (display uses top-1)")
    parser.add_argument("--frame-stride", type=int, default=1, help="run inference every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="optional cap on processed frames (0 = all)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="extra cfg options")
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video not found: {args.video}")

    device = _resolve_device(args.device)

    model = build_detection_model(cfg)
    model.to(device)
    model.eval()

    ckpt_path = _resolve_checkpoint_path(args.ckpt, cfg)
    if ckpt_path and os.path.exists(ckpt_path):
        checkpointer = ActionCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("Warning: checkpoint not found. Running with current model weights.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if args.max_frames > 0 and len(frames) >= args.max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError("No frames decoded from input video.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output writer: {args.output}")

    preproc = PreprocessWithBoxes("test", cfg.DATA, cfg.IMAGES)

    vocab = None
    if cfg.DATA.OPEN_VOCABULARY and hasattr(model.backbone, "text_encoder"):
        text_data = getattr(model.backbone.text_encoder, "text_data", None)
        if isinstance(text_data, dict):
            vocab = list(text_data.keys())

    num_frames = max(1, int(cfg.DATA.NUM_FRAMES))
    sample_rate = max(1, int(cfg.DATA.SAMPLING_RATE))
    pathways = int(cfg.MODEL.BACKBONE.PATHWAYS)

    with torch.no_grad():
        for center_idx in range(len(frames)):
            if center_idx % max(1, args.frame_stride) != 0:
                writer.write(frames[center_idx])
                continue

            seq_idx = _get_sequence_indices(center_idx, len(frames), num_frames, sample_rate)
            clip_imgs = [frames[i] for i in seq_idx]

            imgs_proc, _ = preproc.process(clip_imgs, boxes=None)
            imgs_packed = dataset_utils.pack_pathway_output(cfg, imgs_proc, pathways=pathways)

            if pathways == 1:
                primary_inputs = imgs_packed[0][None].to(device)
                secondary_inputs = None
            else:
                primary_inputs = imgs_packed[0][None].to(device)
                secondary_inputs = imgs_packed[1][None].to(device)

            ph, pw = primary_inputs.shape[-2:]
            whwh = torch.tensor([[pw, ph, pw, ph]], dtype=torch.float32, device=device)

            action_score_list, box_list = model(primary_inputs, secondary_inputs, whwh)
            boxes_norm = box_list[0].detach().cpu() if len(box_list) > 0 else None
            scores = action_score_list[0].detach().cpu() if len(action_score_list) > 0 else None

            vis_frame = frames[center_idx]
            if boxes_norm is not None and scores is not None:
                vis_frame = _draw_predictions(
                    vis_frame,
                    boxes_norm,
                    scores,
                    topk=args.topk,
                    multilabel=bool(cfg.MODEL.MULTI_LABEL_ACTION),
                    vocab=vocab,
                )

            writer.write(vis_frame)

            if (center_idx + 1) % 20 == 0 or center_idx + 1 == len(frames):
                print(f"Processed {center_idx + 1}/{len(frames)} frames")

    writer.release()
    print(f"Saved annotated video to: {args.output}")


if __name__ == "__main__":
    main()
