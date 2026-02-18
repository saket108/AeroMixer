import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

from .box_ops import generalized_box_iou


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _get_target_boxes(target):
    if "boxes_xyxy" in target:
        return target["boxes_xyxy"]
    return target["boxes"]


# ---------------------------------------------------------
# HUNGARIAN MATCHER (DETR STYLE)
# ---------------------------------------------------------

class HungarianMatcher(nn.Module):

    def __init__(self, cfg=None, cost_class=1, cost_bbox=5, cost_giou=2, **kwargs):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):

        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids_list = [v["labels"] for v in targets]
        tgt_box_list = [_get_target_boxes(v) for v in targets]

        tgt_ids = torch.cat(tgt_ids_list, dim=0)
        tgt_bbox = torch.cat(tgt_box_list, dim=0)

        if tgt_bbox.numel() == 0:
            return [
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
                for _ in range(bs)
            ]

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(boxes) for boxes in tgt_box_list]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


# ---------------------------------------------------------
# DETECTION LOSS
# ---------------------------------------------------------

class SetCriterion(nn.Module):

    def __init__(self, cfg=None, num_classes=1, matcher=None, eos_coef=0.1, losses=None, use_focal=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = losses if losses is not None else ["labels", "boxes"]

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # ---------------- CLASSIFICATION ----------------

    def loss_labels(self, outputs, targets, indices, num_boxes):

        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    # ---------------- BOX LOSS ----------------

    def loss_boxes(self, outputs, targets, indices, num_boxes):

        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([_get_target_boxes(t)[i] for t, (_, i) in zip(targets, indices)], dim=0)

        if src_boxes.numel() == 0 or target_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {
                "loss_bbox": zero,
                "loss_giou": zero,
            }

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        loss_giou = loss_giou.sum() / num_boxes

        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

    # ---------------- UTILS ----------------

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # ---------------- FORWARD ----------------

    def forward(self, outputs, targets):

        outputs_wo_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_wo_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs_wo_aux["pred_logits"].device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == "labels":
                losses.update(self.loss_labels(outputs_wo_aux, targets, indices, num_boxes))
            elif loss == "boxes":
                losses.update(self.loss_boxes(outputs_wo_aux, targets, indices, num_boxes))

        return losses
