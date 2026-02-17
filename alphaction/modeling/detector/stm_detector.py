from torch import nn
import torch
import torch.nn.functional as F

from ..backbone import build_backbone
from ..stm_decoder.stm_decoder import build_stm_decoder


# ------------------------------------------------------------
# UNIVERSAL DETECTOR (VIDEO + IMAGE)
# ------------------------------------------------------------
class STMDetector(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.stm_head = build_stm_decoder(cfg)

        # detect dataset mode
        self.is_image = cfg.DATA.INPUT_TYPE == "image"

        # lightweight projection for image mode
        if self.is_image:
            hidden_dim = cfg.MODEL.STM.HIDDEN_DIM
            self.img_proj = nn.Sequential(
                nn.Conv2d(self.backbone.dim_embed, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            )

        print(">>>> STMDetector running in",
              "IMAGE MODE" if self.is_image else "VIDEO MODE")


    # --------------------------------------------------------
    # IMAGE FORWARD (FAST)
    # --------------------------------------------------------
    def forward_image(self, primary_inputs, whwh, boxes=None, labels=None):

        # backbone returns (B,C,H,W) for images
        feats = self.backbone([primary_inputs])

        if isinstance(feats, tuple):
            feats = feats[0]

        # remove temporal dim if exists
        if feats.dim() == 5:
            feats = feats[:, :, 0]

        feats = self.img_proj(feats)

        # fake multi-scale list for decoder compatibility
        mapped_features = [feats.unsqueeze(2)] * 4

        return self.stm_head(
            mapped_features,
            whwh,
            gt_boxes=boxes,
            labels=labels,
            text_features=None,
            tau_inv=100,
            cls_feat=None,
            patch_feat=None,
            text_token_feats=None,
            extras={}
        )


    # --------------------------------------------------------
    # ORIGINAL VIDEO FORWARD (UNCHANGED)
    # --------------------------------------------------------
    def forward_video(self, primary_inputs, secondary_inputs, whwh, boxes, labels, extras):

        if self.backbone.num_pathways == 1:
            features = self.backbone([primary_inputs])
        else:
            features = self.backbone([primary_inputs, secondary_inputs])

        mapped_features = features

        return self.stm_head(
            mapped_features,
            whwh,
            gt_boxes=boxes,
            labels=labels,
            text_features=None,
            tau_inv=100,
            cls_feat=None,
            patch_feat=None,
            text_token_feats=None,
            extras=extras
        )


    # --------------------------------------------------------
    # UNIVERSAL FORWARD
    # --------------------------------------------------------
    def forward(self, primary_inputs, secondary_inputs, whwh,
                boxes=None, labels=None, extras={}, part_forward=-1):

        if self.is_image:
            return self.forward_image(primary_inputs, whwh, boxes, labels)
        else:
            return self.forward_video(primary_inputs, secondary_inputs, whwh, boxes, labels, extras)


# ------------------------------------------------------------
def build_detection_model(cfg):
    return STMDetector(cfg)
