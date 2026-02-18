from typing import List

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

try:
    import supervision as sv
except Exception:  # pragma: no cover
    sv = None

try:
    import imageio
except Exception:  # pragma: no cover
    imageio = None


def _to_xyxy_array(boxes, normalized, image_shape, is_xyxy):
    h, w = image_shape[:2]

    if torch.is_tensor(boxes):
        box_tensor = boxes.detach().cpu().to(dtype=torch.float32)
    else:
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32)

    if box_tensor.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    box_tensor = box_tensor.reshape(-1, 4)
    if normalized:
        scale = torch.tensor([w, h, w, h], dtype=box_tensor.dtype)
        box_tensor = box_tensor * scale

    if not is_xyxy:
        box_tensor = box_convert(boxes=box_tensor, in_fmt="cxcywh", out_fmt="xyxy")

    return box_tensor.numpy()


def _fallback_annotate(image_source, xyxy, labels, color, set_text_color):
    annotated = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    box_color = color if isinstance(color, tuple) and len(color) == 3 else (0, 255, 0)
    text_color = (255, 255, 255) if set_text_color == "white" else (0, 0, 0)

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        if i < len(labels) and labels[i]:
            y = max(12, y1 - 6)
            cv2.putText(
                annotated,
                str(labels[i]),
                (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    normalized=True,
    logits=None,
    phrases: List[str] = None,
    is_xyxy=False,
    color=None,
    text_padding=10,
    set_text_color="black",
) -> np.ndarray:
    if phrases is None:
        phrases = []

    xyxy = _to_xyxy_array(boxes, normalized=normalized, image_shape=image_source.shape, is_xyxy=is_xyxy)

    if logits is not None and len(phrases) == len(xyxy):
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().tolist()
        labels = [f"{phrase} {score:.2f}" for phrase, score in zip(phrases, logits)]
    else:
        labels = list(phrases)

    if sv is None:
        return _fallback_annotate(image_source, xyxy, labels, color, set_text_color)

    detections = sv.Detections(xyxy=xyxy)
    if color is None or not isinstance(color, tuple):
        svcolor = sv.ColorPalette.default()
    else:
        svcolor = sv.Color(*color)

    text_color = sv.Color.white() if set_text_color == "white" else sv.Color.black()
    box_annotator = sv.BoxAnnotator(color=svcolor, text_padding=text_padding, text_color=text_color)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


def video_to_gif(video, giffile, fps=5.0, toBGR=False):
    if imageio is None:
        raise ImportError("imageio is required for video_to_gif. Install it with `pip install imageio`.")
    assert giffile.endswith('.gif')
    with imageio.get_writer(giffile, mode='I', duration=1.0 / fps, loop=0) as writer:
        for frame in video:
            frame_vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if toBGR else np.copy(frame)
            writer.append_data(frame_vis)
