import torch

from .action_head.action_head import build_roi_action_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """Compatibility wrapper around the image+text ROI action head."""

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(
        self,
        primary_features=None,
        auxiliary_features=None,
        boxes=None,
        objects=None,
        extras=None,
        part_forward=-1,
        **kwargs,
    ):
        if primary_features is None and "slow_features" in kwargs:
            primary_features = kwargs["slow_features"]
        if auxiliary_features is None and "fast_features" in kwargs:
            auxiliary_features = kwargs["fast_features"]
        if extras is None:
            extras = {}

        result, loss_action, loss_weight, accuracy_action = self.action(
            primary_features,
            auxiliary_features,
            boxes,
            objects,
            extras,
            part_forward,
        )
        return result, loss_action, loss_weight, accuracy_action

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    weight_map[name + "." + key] = val
        return weight_map


Combined3dROIHeads = CombinedROIHeads


def build_3d_roi_heads(cfg, dim_in):
    heads = [("action", build_roi_action_head(cfg, dim_in))]
    return CombinedROIHeads(cfg, heads)
