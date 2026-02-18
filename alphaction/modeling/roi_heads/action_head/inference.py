import torch
from torch import nn
import torch.nn.functional as F

from alphaction.structures.bounding_box import BoxList


class PostProcessor(nn.Module):
    """Convert action logits to BoxList outputs for image+text inference."""

    def __init__(self, multi_label=False, softmax_classes=None):
        super(PostProcessor, self).__init__()
        self.multi_label = bool(multi_label)
        self.softmax_classes = softmax_classes

    def _to_probability(self, class_logits):
        split = self.softmax_classes
        if isinstance(split, int) and 0 < split < class_logits.size(1):
            pose_prob = F.softmax(class_logits[:, :split], dim=-1)
            interaction_prob = torch.sigmoid(class_logits[:, split:])
            return torch.cat([pose_prob, interaction_prob], dim=1)
        if self.multi_label:
            return torch.sigmoid(class_logits)
        return F.softmax(class_logits, dim=-1)

    def _extract_box_data(self, box_item):
        if isinstance(box_item, BoxList):
            return box_item.bbox, box_item.size

        bbox = None
        image_shape = None
        if hasattr(box_item, "bbox"):
            bbox = box_item.bbox
            image_shape = getattr(box_item, "size", None)
        elif torch.is_tensor(box_item):
            bbox = box_item
        else:
            bbox = torch.as_tensor(box_item, dtype=torch.float32)

        bbox = bbox.to(dtype=torch.float32)
        if bbox.numel() == 0:
            bbox = bbox.reshape(0, 4)
        else:
            bbox = bbox.reshape(-1, 4)

        return bbox, image_shape

    def _infer_image_shape(self, bbox):
        if bbox.numel() == 0:
            return (1, 1)
        max_xy = torch.ceil(bbox[:, 2:4].max(dim=0).values)
        width = int(max(1, max_xy[0].item() + 1))
        height = int(max(1, max_xy[1].item() + 1))
        return (width, height)

    def forward(self, x, boxes):
        if torch.is_tensor(x):
            class_logits = x
        elif isinstance(x, (list, tuple)) and len(x) > 0:
            class_logits = x[0]
        else:
            raise ValueError("PostProcessor expects logits tensor or tuple/list with logits.")

        action_prob = self._to_probability(class_logits)

        counts = [len(box) for box in boxes]
        expected = sum(counts)
        if action_prob.size(0) != expected:
            raise ValueError(
                f"Mismatch between logits rows ({action_prob.size(0)}) and proposal boxes ({expected})."
            )
        action_prob = action_prob.split(counts, dim=0)

        results = []
        for prob, box_item in zip(action_prob, boxes):
            box_tensor, image_shape = self._extract_box_data(box_item)
            if image_shape is None:
                image_shape = self._infer_image_shape(box_tensor)
            boxlist = BoxList(box_tensor, image_shape, mode="xyxy")
            boxlist.add_field("scores", prob)
            boxlist.add_field("action_scores", prob)
            results.append(boxlist)
        return results


def make_roi_action_post_processor(cfg):
    multi_label = bool(getattr(cfg.MODEL, "MULTI_LABEL_ACTION", False))
    roi_cfg = getattr(cfg.MODEL, "ROI_ACTION_HEAD", None)
    split = None if roi_cfg is None else getattr(roi_cfg, "NUM_PERSON_MOVEMENT_CLASSES", None)
    split = int(split) if isinstance(split, (int, float)) and split > 0 else None
    return PostProcessor(multi_label=multi_label, softmax_classes=split)
