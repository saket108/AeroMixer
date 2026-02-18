import torch
from torch import nn

from ..backbone import build_backbone
from ..roi_heads.roi_heads_3d import build_3d_roi_heads


class MultimodalActionDetector(nn.Module):
    """Detector wrapper with image+text-first semantics and legacy API support."""

    def __init__(self, cfg):
        super(MultimodalActionDetector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)

        dim_out = getattr(self.backbone, "dim_out", None)
        if dim_out is None:
            dim_out = getattr(self.backbone, "dim_embed", 256)
        if isinstance(dim_out, (list, tuple)):
            dim_out = dim_out[-1] if len(dim_out) > 0 else 256
        if not isinstance(dim_out, int):
            dim_out = 256

        self.roi_heads = build_3d_roi_heads(cfg, dim_out)

    def _run_backbone(self, primary_inputs, auxiliary_inputs=None):
        attempts = []
        if auxiliary_inputs is None:
            attempts.extend([
                lambda: self.backbone([primary_inputs]),
                lambda: self.backbone(primary_inputs),
            ])
        else:
            attempts.extend([
                lambda: self.backbone([primary_inputs, auxiliary_inputs]),
                lambda: self.backbone(primary_inputs, auxiliary_inputs),
                lambda: self.backbone([primary_inputs]),
                lambda: self.backbone(primary_inputs),
            ])

        last_error = None
        for run in attempts:
            try:
                return run()
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Backbone forward failed for all supported call patterns: {last_error}")

    def _split_features(self, backbone_out):
        if torch.is_tensor(backbone_out):
            return backbone_out, None

        if isinstance(backbone_out, (list, tuple)):
            if (
                len(backbone_out) == 2
                and torch.is_tensor(backbone_out[0])
                and torch.is_tensor(backbone_out[1])
                and backbone_out[0].ndim >= 4
                and backbone_out[1].ndim >= 4
            ):
                return backbone_out[0], backbone_out[1]
            return backbone_out, None

        return backbone_out, None

    def forward(
        self,
        primary_inputs,
        auxiliary_inputs=None,
        boxes=None,
        objects=None,
        extras=None,
        part_forward=-1,
        **kwargs,
    ):
        if auxiliary_inputs is None and "secondary_inputs" in kwargs:
            auxiliary_inputs = kwargs["secondary_inputs"]
        if extras is None:
            extras = {}

        if part_forward == 1:
            primary_features, aux_features = None, None
        else:
            backbone_out = self._run_backbone(primary_inputs, auxiliary_inputs)
            primary_features, aux_features = self._split_features(backbone_out)

        result, detector_losses, loss_weight, detector_metrics = self.roi_heads(
            primary_features=primary_features,
            auxiliary_features=aux_features,
            boxes=boxes,
            objects=objects,
            extras=extras,
            part_forward=part_forward,
        )

        if self.training:
            return detector_losses, loss_weight, detector_metrics, result
        return result

    def c2_weight_mapping(self):
        if not hasattr(self, "c2_mapping"):
            weight_map = {}
            for name, m_child in self.named_children():
                if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                    child_map = m_child.c2_weight_mapping()
                    for key, val in child_map.items():
                        weight_map[name + "." + key] = val
            self.c2_mapping = weight_map
        return self.c2_mapping


ActionDetector = MultimodalActionDetector


def build_detection_model(cfg):
    return MultimodalActionDetector(cfg)
