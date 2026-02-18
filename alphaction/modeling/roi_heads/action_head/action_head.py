import torch
import torch.nn.functional as F

from .roi_action_feature_extractor import make_roi_action_feature_extractor
from .inference import make_roi_action_post_processor


def _safe_cat(tensors, dim=0):
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


def _collect_numeric_tensors(value, bucket):
    if value is None:
        return
    if torch.is_tensor(value):
        if value.numel() > 0:
            bucket.append(value)
        return
    if isinstance(value, dict):
        for child in value.values():
            _collect_numeric_tensors(child, bucket)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _collect_numeric_tensors(child, bucket)
        return
    try:
        tensor = torch.as_tensor(value)
    except Exception:
        return
    if tensor.numel() > 0:
        bucket.append(tensor)


class ROIActionHead(torch.nn.Module):
    """ROI action head for image+text multimodal detection."""

    def __init__(self, cfg, dim_in):
        super(ROIActionHead, self).__init__()
        self.cfg = cfg

        roi_cfg = getattr(cfg.MODEL, "ROI_ACTION_HEAD", None)
        self.max_proposals = int(getattr(roi_cfg, "PROPOSAL_PER_CLIP", 0))
        self.test_ext = getattr(cfg.TEST, "EXTEND_SCALE", [1.0, 1.0])
        self.multi_label = bool(getattr(cfg.MODEL, "MULTI_LABEL_ACTION", False))

        self.feature_extractor = make_roi_action_feature_extractor(cfg, dim_in)

        roi_num_classes = getattr(roi_cfg, "NUM_CLASSES", None)
        if roi_num_classes is None:
            roi_num_classes = int(getattr(cfg.MODEL.STM, "ACTION_CLASSES", 1))
        self.num_classes = max(1, int(roi_num_classes))

        self.visual_classifier = torch.nn.Linear(self.feature_extractor.dim_out, self.num_classes)
        self.text_proj = torch.nn.Linear(
            self.feature_extractor.dim_out,
            self.feature_extractor.dim_out,
            bias=False,
        )

        self.post_processor = make_roi_action_post_processor(cfg)

    def _subsample_proposals(self, boxes):
        if self.max_proposals <= 0:
            return boxes

        out = []
        for item in boxes:
            if len(item) > self.max_proposals:
                if hasattr(item, "bbox"):
                    device = item.bbox.device
                elif torch.is_tensor(item):
                    device = item.device
                else:
                    device = torch.device("cpu")
                inds = torch.randperm(len(item), device=device)[: self.max_proposals]
                out.append(item[inds])
            else:
                out.append(item)
        return out

    def _extend_boxes(self, boxes):
        out = []
        for item in boxes:
            if hasattr(item, "extend"):
                out.append(item.extend(self.test_ext))
            else:
                out.append(item)
        return out

    def _align_dim(self, tensor, target_dim):
        cur_dim = tensor.size(-1)
        if cur_dim == target_dim:
            return tensor
        if cur_dim > target_dim:
            return tensor[..., :target_dim]
        return F.pad(tensor, (0, target_dim - cur_dim), mode="constant", value=0.0)

    def _extract_text_features(self, extras, device, dtype, target_dim):
        candidates = []
        if isinstance(extras, dict):
            _collect_numeric_tensors(extras.get("text_features"), candidates)
        elif isinstance(extras, (list, tuple)):
            for item in extras:
                if isinstance(item, dict):
                    _collect_numeric_tensors(item.get("text_features"), candidates)

        flat = []
        for tensor in candidates:
            tensor = tensor.to(device=device, dtype=dtype)
            if tensor.ndim == 0:
                continue
            if tensor.ndim == 1:
                flat.append(tensor.unsqueeze(0))
            else:
                flat.append(tensor.reshape(-1, tensor.shape[-1]))
        if not flat:
            return None

        aligned = [self._align_dim(tensor, target_dim) for tensor in flat]
        return torch.cat(aligned, dim=0)

    def _compute_logits(self, x, extras):
        logits = self.visual_classifier(x)

        text = self._extract_text_features(
            extras,
            device=x.device,
            dtype=x.dtype,
            target_dim=self.text_proj.out_features,
        )
        if text is None or text.numel() == 0:
            return logits

        vis = self.text_proj(x)
        vis = vis / (vis.norm(dim=-1, keepdim=True) + 1e-6)

        text = text / (text.norm(dim=-1, keepdim=True) + 1e-6)
        text_logits = vis @ text.t()
        if text_logits.size(1) != logits.size(1):
            return logits

        return 0.5 * logits + 0.5 * text_logits

    def _read_labels_from_proposal(self, proposal):
        if hasattr(proposal, "has_field") and proposal.has_field("labels"):
            return proposal.get_field("labels")
        if isinstance(proposal, dict) and "labels" in proposal:
            return proposal["labels"]
        if hasattr(proposal, "labels"):
            return proposal.labels
        return None

    def _build_targets(self, proposals, extras, device):
        fallback_labels = None
        if isinstance(extras, dict):
            fallback_labels = extras.get("labels")
        elif isinstance(extras, (list, tuple)):
            fallback_labels = [
                item.get("labels") if isinstance(item, dict) else None for item in extras
            ]

        target_chunks = []
        for idx, proposal in enumerate(proposals):
            labels = self._read_labels_from_proposal(proposal)
            if labels is None and isinstance(fallback_labels, (list, tuple)) and idx < len(fallback_labels):
                labels = fallback_labels[idx]
            if labels is None:
                continue

            labels = torch.as_tensor(labels, device=device)
            if labels.numel() == 0:
                continue

            if labels.ndim == 1:
                labels = labels.long().clamp(min=0, max=self.num_classes - 1)
                one_hot = torch.zeros((labels.size(0), self.num_classes), device=device, dtype=torch.float32)
                one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                target_chunks.append(one_hot)
            else:
                labels = labels.to(dtype=torch.float32).reshape(labels.shape[0], -1)
                labels = self._align_dim(labels, self.num_classes)
                target_chunks.append(labels)

        merged = _safe_cat(target_chunks, dim=0)
        if merged is None:
            return torch.zeros((0, self.num_classes), device=device, dtype=torch.float32)
        return merged

    def forward(
        self,
        primary_features,
        auxiliary_features=None,
        boxes=None,
        objects=None,
        extras=None,
        part_forward=-1,
    ):
        if extras is None:
            extras = {}
        if boxes is None:
            return [], {}, {}, {}

        proposals = self._subsample_proposals(boxes) if self.training else self._extend_boxes(boxes)

        x, pooled_person, _ = self.feature_extractor(
            primary_features,
            auxiliary_features,
            proposals,
            objects=objects,
            extras=extras,
            part_forward=part_forward,
        )

        if part_forward == 0:
            return [pooled_person, None], {}, {}, {}

        logits = self._compute_logits(x, extras)

        if not self.training:
            result = self.post_processor((logits,), boxes)
            return result, {}, {}, {}

        targets = self._build_targets(proposals, extras, logits.device)

        if targets.size(0) != logits.size(0) and targets.size(0) > 0 and logits.size(0) > 0:
            min_count = min(targets.size(0), logits.size(0))
            targets = targets[:min_count]
            logits_for_loss = logits[:min_count]
        else:
            logits_for_loss = logits

        if targets.numel() == 0 or logits_for_loss.numel() == 0:
            zero = logits.sum() * 0.0
            loss_action = zero
            acc = zero
        elif self.multi_label:
            targets = targets.to(dtype=logits_for_loss.dtype)
            loss_action = F.binary_cross_entropy_with_logits(logits_for_loss, targets)
            pred = (torch.sigmoid(logits_for_loss) > 0.5).to(dtype=targets.dtype)
            gt = (targets > 0.5).to(dtype=targets.dtype)
            acc = pred.eq(gt).to(dtype=torch.float32).mean()
        else:
            cls_target = targets.argmax(dim=1)
            loss_action = F.cross_entropy(logits_for_loss, cls_target)
            acc = (logits_for_loss.argmax(dim=1) == cls_target).to(dtype=torch.float32).mean()

        loss_dict = {"loss_action": loss_action}
        loss_weight = {"loss_action": 1.0}
        metric_dict = {"accuracy_action": acc}

        return [pooled_person, None], loss_dict, loss_weight, metric_dict

    def c2_weight_mapping(self):
        return {}


def build_roi_action_head(cfg, dim_in):
    return ROIActionHead(cfg, dim_in)
