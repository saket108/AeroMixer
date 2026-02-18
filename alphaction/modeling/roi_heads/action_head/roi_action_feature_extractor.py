import torch
from torch import nn
from torch.nn import functional as F

from alphaction.modeling import registry

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover
    roi_align = None

from alphaction.structures.bounding_box import BoxList


@registry.ROI_ACTION_FEATURE_EXTRACTORS.register("2MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    """2D ROI feature extractor for image+text multimodal heads."""

    def __init__(self, cfg, dim_in):
        super(MLPFeatureExtractor, self).__init__()
        self.cfg = cfg

        roi_head_cfg = getattr(cfg.MODEL, "ROI_ACTION_HEAD", None)
        self.resolution = int(getattr(roi_head_cfg, "POOLER_RESOLUTION", 7))
        self.sampling_ratio = int(getattr(roi_head_cfg, "POOLER_SAMPLING_RATIO", 2))
        representation_size = int(getattr(roi_head_cfg, "MLP_HEAD_DIM", 1024))

        self.dim_in = self._resolve_dim(dim_in)

        self.fc1 = nn.Linear(self.dim_in, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        nn.init.kaiming_uniform_(self.fc1.weight, a=1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, a=1)
        nn.init.constant_(self.fc2.bias, 0)

        self.dim_out = representation_size

    def _resolve_dim(self, dim_in):
        if isinstance(dim_in, int):
            return dim_in
        if isinstance(dim_in, (list, tuple)):
            for value in reversed(dim_in):
                if isinstance(value, int):
                    return value
        return 256

    def _pick_tensor(self, value):
        if torch.is_tensor(value):
            return value

        if isinstance(value, dict):
            values = list(value.values())
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            return None

        best = None
        for item in values:
            candidate = self._pick_tensor(item)
            if candidate is None:
                continue
            if best is None or candidate.ndim > best.ndim:
                best = candidate
        return best

    def _extract_feature_map(self, primary_features, auxiliary_features):
        feature_candidate = primary_features if primary_features is not None else auxiliary_features
        feature_map = self._pick_tensor(feature_candidate)
        if feature_map is None:
            raise ValueError("No tensor feature map found for ROI extraction.")

        if feature_map.ndim == 3:
            feature_map = feature_map.unsqueeze(0)
        if feature_map.ndim == 5:
            feature_map = feature_map.mean(dim=2)
        if feature_map.ndim != 4:
            raise ValueError(
                f"Expected a 4D feature map after reduction, got shape={tuple(feature_map.shape)}."
            )
        return feature_map

    def _extract_boxes_and_size(self, box_item, device):
        image_size = None
        if isinstance(box_item, BoxList):
            bbox = box_item.bbox
            image_size = box_item.size
        elif hasattr(box_item, "bbox"):
            bbox = box_item.bbox
            image_size = getattr(box_item, "size", None)
        elif isinstance(box_item, dict):
            bbox = box_item.get("boxes", [])
            image_size = box_item.get("size", None)
        else:
            bbox = box_item

        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.numel() == 0:
            bbox = bbox.reshape(0, 4)
        else:
            bbox = bbox.reshape(-1, 4)

        if image_size is not None:
            image_size = (float(image_size[0]), float(image_size[1]))

        return bbox, image_size

    def _project_boxes_to_feature(self, bbox, image_size, feat_h, feat_w):
        if bbox.numel() == 0:
            return bbox

        box = bbox.clone()
        box_max = float(box.abs().max().item())
        looks_normalized = box_max <= 2.0

        if image_size is not None:
            img_w, img_h = image_size
            if looks_normalized:
                box[:, 0::2] *= max(img_w - 1.0, 1.0)
                box[:, 1::2] *= max(img_h - 1.0, 1.0)

            scale_x = (feat_w - 1.0) / max(img_w - 1.0, 1.0)
            scale_y = (feat_h - 1.0) / max(img_h - 1.0, 1.0)
            box[:, 0::2] *= scale_x
            box[:, 1::2] *= scale_y
        elif looks_normalized:
            box[:, 0::2] *= max(feat_w - 1.0, 1.0)
            box[:, 1::2] *= max(feat_h - 1.0, 1.0)

        box[:, 0::2].clamp_(min=0.0, max=max(feat_w - 1.0, 0.0))
        box[:, 1::2].clamp_(min=0.0, max=max(feat_h - 1.0, 0.0))
        box[:, 2] = torch.maximum(box[:, 2], box[:, 0] + 1.0)
        box[:, 3] = torch.maximum(box[:, 3], box[:, 1] + 1.0)
        return box

    def _build_rois(self, proposals, feat_h, feat_w, device):
        rois = []
        for batch_idx, box_item in enumerate(proposals):
            bbox, image_size = self._extract_boxes_and_size(box_item, device=device)
            if bbox.numel() == 0:
                continue

            bbox = self._project_boxes_to_feature(bbox, image_size=image_size, feat_h=feat_h, feat_w=feat_w)
            batch_col = torch.full((bbox.size(0), 1), float(batch_idx), device=device)
            rois.append(torch.cat([batch_col, bbox], dim=1))

        if not rois:
            return torch.zeros((0, 5), dtype=torch.float32, device=device)
        return torch.cat(rois, dim=0)

    def _roi_pool(self, feature_map, proposals):
        if proposals is None:
            proposals = []
        _, channels, feat_h, feat_w = feature_map.shape
        rois = self._build_rois(proposals, feat_h=feat_h, feat_w=feat_w, device=feature_map.device)

        if rois.numel() == 0:
            return torch.zeros(
                (0, channels, self.resolution, self.resolution),
                dtype=feature_map.dtype,
                device=feature_map.device,
            )

        if roi_align is not None:
            try:
                return roi_align(
                    feature_map,
                    rois,
                    output_size=(self.resolution, self.resolution),
                    spatial_scale=1.0,
                    sampling_ratio=self.sampling_ratio,
                    aligned=True,
                )
            except TypeError:
                return roi_align(
                    feature_map,
                    rois,
                    output_size=(self.resolution, self.resolution),
                    spatial_scale=1.0,
                    sampling_ratio=self.sampling_ratio,
                )

        pooled = []
        for roi in rois:
            batch_idx = int(roi[0].item())
            x1, y1, x2, y2 = roi[1:].tolist()
            x1 = max(0, min(feat_w - 1, int(round(x1))))
            y1 = max(0, min(feat_h - 1, int(round(y1))))
            x2 = max(x1 + 1, min(feat_w, int(round(x2))))
            y2 = max(y1 + 1, min(feat_h, int(round(y2))))
            crop = feature_map[batch_idx:batch_idx + 1, :, y1:y2, x1:x2]
            pooled.append(F.adaptive_avg_pool2d(crop, (self.resolution, self.resolution)))
        return torch.cat(pooled, dim=0)

    def _align_dim(self, tensor, target_dim):
        cur_dim = tensor.size(-1)
        if cur_dim == target_dim:
            return tensor
        if cur_dim > target_dim:
            return tensor[..., :target_dim]
        return F.pad(tensor, (0, target_dim - cur_dim), mode="constant", value=0.0)

    def _collect_numeric_tensors(self, value, bucket):
        if value is None:
            return
        if torch.is_tensor(value):
            if value.numel() > 0:
                bucket.append(value)
            return
        if isinstance(value, dict):
            for child in value.values():
                self._collect_numeric_tensors(child, bucket)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                self._collect_numeric_tensors(child, bucket)
            return
        try:
            tensor = torch.as_tensor(value)
        except Exception:
            return
        if tensor.numel() > 0:
            bucket.append(tensor)

    def _extract_text_features(self, extras, device, dtype):
        if extras is None:
            return None

        candidates = []
        if isinstance(extras, dict):
            self._collect_numeric_tensors(extras.get("text_features"), candidates)
        elif isinstance(extras, (list, tuple)):
            for item in extras:
                if isinstance(item, dict):
                    self._collect_numeric_tensors(item.get("text_features"), candidates)

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

        base_dim = flat[0].size(-1)
        aligned = [self._align_dim(tensor, base_dim) for tensor in flat]
        return torch.cat(aligned, dim=0)

    def _apply_text_gate(self, pooled_vec, extras):
        text_features = self._extract_text_features(
            extras, device=pooled_vec.device, dtype=pooled_vec.dtype
        )
        if text_features is None or text_features.numel() == 0:
            return pooled_vec

        text_vec = text_features.mean(dim=0, keepdim=True)
        text_vec = self._align_dim(text_vec, pooled_vec.size(1))
        text_vec = text_vec / (text_vec.norm(dim=-1, keepdim=True) + 1e-6)

        vis = pooled_vec / (pooled_vec.norm(dim=-1, keepdim=True) + 1e-6)
        gate = torch.sigmoid((vis * text_vec).sum(dim=-1, keepdim=True))
        return pooled_vec * gate

    def forward(
        self,
        primary_features,
        auxiliary_features,
        proposals,
        objects=None,
        extras=None,
        part_forward=-1,
    ):
        if extras is None:
            extras = {}

        if part_forward == 1 and isinstance(extras, dict) and "current_feat_p" in extras:
            pooled_person = extras["current_feat_p"]
            pooled_person = torch.as_tensor(pooled_person)
            return pooled_person, pooled_person, None

        feature_map = self._extract_feature_map(primary_features, auxiliary_features)
        pooled_map = self._roi_pool(feature_map, proposals)
        pooled_person = F.adaptive_avg_pool2d(pooled_map, (1, 1)).flatten(1)

        if part_forward == 0:
            return None, pooled_person, None

        pooled_person = self._apply_text_gate(pooled_person, extras)
        x = F.relu(self.fc1(pooled_person))
        x = F.relu(self.fc2(x))
        return x, pooled_person, None


def make_roi_action_feature_extractor(cfg, dim_in):
    roi_cfg = getattr(cfg.MODEL, "ROI_ACTION_HEAD", None)
    name = "2MLPFeatureExtractor"
    if roi_cfg is not None:
        name = getattr(roi_cfg, "FEATURE_EXTRACTOR", name)
    if name not in registry.ROI_ACTION_FEATURE_EXTRACTORS:
        name = "2MLPFeatureExtractor"
    return registry.ROI_ACTION_FEATURE_EXTRACTORS[name](cfg, dim_in)
