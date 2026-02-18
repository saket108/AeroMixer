"""
Naive Baseline for Image Multimodal Models.
Converted to support image + text multimodal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone import build_backbone
from alphaction.modeling.stm_decoder.util.box_ops import clip_boxes_tensor
from torchvision.ops import roi_align
import numpy as np



class NaiveBaseline(nn.Module):
    """
    Naive Baseline for image multimodal detection.
    Supports both video and image modes.
    """
    
    def __init__(self, cfg):
        super(NaiveBaseline, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        
        # Check if we're in image mode
        self.is_image = cfg.DATA.INPUT_TYPE == "image" or getattr(cfg.DATA, 'IMAGE_MODE', False)
        
        assert self.backbone.visual_encoder.use_cls_feat
        assert cfg.DATA.OPEN_VOCABULARY

        self.use_roi_feat = cfg.MODEL.USE_ROI_FEAT
        self.multi_label_action = cfg.MODEL.MULTI_LABEL_ACTION
        
        print(">>>> NaiveBaseline running in",
              "IMAGE MODE" if self.is_image else "VIDEO MODE")
    
    
    def roi_align_pool(self, patch_feats, batch_boxes, raw_sizes, out_size=(7, 7), spatial_scale=1.0/16):
        """
        ROI Align Pooling for image features.
        
        Args:
            patch_feats: For image mode: (B, D, h, w) - no temporal dimension
                        For video mode: (B, D, T, h, w) - has temporal dimension
            boxes: list of boxes, not normalized
            raw_sizes: (B, 2) in (width, height)
        """
        # Handle both image and video modes
        if patch_feats.dim() == 4:
            # Image mode: (B, D, h, w)
            feat_maps = patch_feats  # No temporal dimension
        else:
            # Video mode: (B, D, T, h, w)
            feat_maps = patch_feats.mean(dim=2)  # temporally mean pooling
        
        B, D, h, w = feat_maps.size()
        device = feat_maps.device
        
        boxes_list = [np.hstack([np.ones((boxes.shape[0], 1)) * i, boxes]) for i, boxes in enumerate(batch_boxes)]
        boxes_tensor = torch.from_numpy(np.vstack(boxes_list)).type(patch_feats.dtype).to(device)
        roi_feat = roi_align(feat_maps, boxes_tensor, out_size, spatial_scale)  # (BN, D, 7, 7)
        
        # Handle both 2D and 3D ROI pooling results
        if roi_feat.dim() == 4:
            # Image mode: (BN, D, h, w) -> (BN, h*w, D)
            roi_feat = roi_feat.flatten(2).permute(0, 2, 1)  # (BN, h*w, D)
        else:
            # Fallback
            roi_feat = roi_feat.flatten(2).permute(0, 2, 1)
        
        # get meanpooled roi features
        roi_align_features = []
        batch_indices = boxes_tensor[:, 0].long()
        for i in range(B):
            rois = roi_feat[batch_indices == i].mean(dim=1)  # (n, D)
            roi_align_features.append(rois)
        
        return roi_align_features

    
    def forward(self, primary_inputs, secondary_inputs=None, whwh=None, boxes=None, labels=None, extras={}, part_forward=-1):
        """
        Forward pass for image multimodal.
        
        Args:
            primary_inputs: For image mode: single image tensor (B, C, H, W)
                           For video mode: slow video tensor
            secondary_inputs: For video mode: fast video tensor (ignored in image mode)
            whwh: Image sizes
            boxes: Bounding box proposals
            labels: Labels
            extras: Extra information
            part_forward: Which part to run
        """

        assert not self.training, "NaiveBaseline does not need training!"
        assert 'prior_boxes' in extras, "NaiveBaseline use loaded boxes for testing!"
        
        device = primary_inputs.device

        prior_boxes = extras['prior_boxes']
        box_list = []
        for i in range(len(prior_boxes)):
            box = torch.tensor(prior_boxes[i], dtype=torch.float32, device=device)
            cur_whwh = whwh[i]
            box = clip_boxes_tensor(box, cur_whwh[1], cur_whwh[0])
            box[:, 0::2] /= cur_whwh[0]
            box[:, 1::2] /= cur_whwh[1]
            box_list.append(box)

        # Handle both image and video modes
        if self.is_image:
            # Image mode: single input
            if self.backbone.num_pathways == 1:
                features = self.backbone([primary_inputs])
            else:
                # For image mode, duplicate the input for pathways that expect two inputs
                features = self.backbone([primary_inputs, primary_inputs])
        else:
            # Video mode: slow and fast inputs
            if self.backbone.num_pathways == 1:
                features = self.backbone([primary_inputs])
            else:
                features = self.backbone([primary_inputs, secondary_inputs])
        
        patch_feats, cls_feat_visual = features  # (B, 512)
        B = cls_feat_visual.size(0)

        if self.use_roi_feat:
            # feature projection & RoIAlign Pooling
            patch_feats = self.backbone.visual_encoder.project_patch_features(patch_feats[0])
            
            # For image mode, patch_feats is (B, D, H, W)
            # For video mode, patch_feats is (B, D, T, H, W)
            roi_features = self.roi_align_pool(patch_feats, prior_boxes, whwh[:, :2])

        # get the current text feature embeddings
        text_features = self.backbone.forward_text(device=primary_inputs.device)  # (K, 512)
        tau_inv = self.backbone.tau_inv

        if isinstance(text_features, list):
            text_features = torch.stack(text_features).mean(1)
        text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)  # (K, D)

        action_score_list = []
        if self.use_roi_feat:
            for roi_feat in roi_features:
                # action recognition
                vis_features_normed = roi_feat / roi_feat.norm(dim=-1, keepdim=True)  # (N, D)
                action_score = tau_inv * vis_features_normed @ text_features_normed.t()  # (N, K)
                action_score_list.append(action_score)
        else:
            vis_features_normed = cls_feat_visual / cls_feat_visual.norm(dim=-1, keepdim=True)  # (B, D)
            action_score = tau_inv * vis_features_normed @ text_features_normed.t()  # (B, K)
            for i in range(B):
                # with full frame input, we only have one score vector, which need to be repeated.
                scores = action_score[[i]].repeat(box_list[i].size(0), 1)  # (1, K)
                action_score_list.append(scores)
        
        return action_score_list, box_list


def build_naive_baseline(cfg):
    return NaiveBaseline(cfg)
