import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import sigmoid_focal_loss
from scipy.optimize import linear_sum_assignment

from . import box_ops
from .misc import accuracy, get_world_size, is_dist_avail_and_initialized
from .box_ops import generalized_box_iou


# -------------------------------------------------------------
# Helper
# -------------------------------------------------------------
def _to_class_indices(target):
    """
    Converts labels to class indices.

    Supports:
    - class index tensor [N]
    - one hot tensor [N,C]
    """
    if target.ndim == 2:
        return torch.argmax(target, dim=1)
    return target.long().view(-1)


# -------------------------------------------------------------
# Loss
# -------------------------------------------------------------
class SetCriterion(nn.Module):

    def __init__(self, cfg, num_classes, matcher, eos_coef, losses, use_focal=False):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = losses

        self.multi_label_action = cfg.MODEL.MULTI_LABEL_ACTION
        self.use_focal = use_focal
        self.focal_gamma = cfg.MODEL.STM.FS_GAMMA

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # ---------------------------------------------------------
    # Classification loss (object)
    # ---------------------------------------------------------
    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):

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
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    # ---------------------------------------------------------
    # Action classification loss (FIXED UNIVERSAL VERSION)
    # ---------------------------------------------------------
    def action_cls_loss(self, output_action, targets, indices, idx):

        target_actions = torch.cat([t["action_labels"][J] for t, (_, J) in zip(targets, indices)])
        action = output_action[idx]

        # ---- Multi-label AVA ----
        if self.multi_label_action:
            return F.binary_cross_entropy_with_logits(action, target_actions.float())

        # ---- One-hot image dataset ----
        if target_actions.ndim == 2:
            target_indices = torch.argmax(target_actions, dim=1)
        else:
            target_indices = target_actions.long()

        if self.focal_gamma > 0:
            ce = F.cross_entropy(action, target_indices, reduction="none")
            p = F.softmax(action, dim=-1)
            pt = p[torch.arange(len(target_indices), device=p.device), target_indices]
            return (ce * ((1 - pt) ** self.focal_gamma)).mean()

        return F.cross_entropy(action, target_indices)

    # ---------------------------------------------------------
    # Boxes loss
    # ---------------------------------------------------------
    def loss_boxes(self, outputs, targets, indices, num_boxes):

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes_xyxy"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        return {
            "loss_giou": loss_giou.sum() / num_boxes,
            "loss_bbox": loss_bbox.sum() / num_boxes,
        }

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, outputs, targets):

        outputs_wo_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_wo_aux, targets)

        idx = self._get_src_permutation_idx(indices)
        losses = {}

        if "pred_actions" in outputs_wo_aux:
            losses["loss_action"] = self.action_cls_loss(outputs_wo_aux["pred_actions"], targets, indices, idx)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        for loss in self.losses:
            if loss == "labels":
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            if loss == "boxes":
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        return losses

    # ---------------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


# -------------------------------------------------------------
# Matcher (unchanged logic, cleaned)
# -------------------------------------------------------------
class HungarianMatcher(nn.Module):

    def __init__(self, cfg, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):

        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes_xyxy"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i), torch.as_tensor(j)) for i, j in indices]
