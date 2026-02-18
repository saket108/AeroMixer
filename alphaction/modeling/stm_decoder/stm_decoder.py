"""
STM Decoder for Image Multimodal Models.
Converted to support image + text multimodal (temporal concepts removed/disabled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .util.box_ops import bbox_overlaps, box_cxcywh_to_xyxy, box_xyxy_to_xyzr, clip_boxes_tensor, box_sampling_from_heatmap, box_sampling_from_prior
from .util.msaq import SAMPLE4D
from .util.adaptive_mixing_operator import AdaptiveMixing
from .util.head_utils import _get_activation_layer, bias_init_with_prob, decode_box, position_embedding, make_sample_points, refine_xyzr
from .util.head_utils import FFN, MultiheadAttention
from .util.loss import SetCriterion, HungarianMatcher



def _resolve_text_encoder_name(cfg):
    name = str(getattr(cfg.MODEL, "TEXT_ENCODER", "")).strip()
    return name if name else "CLIP"


def _infer_text_dim(cfg, text_encoder_name):
    name = str(text_encoder_name).strip()
    if name.lower() == "clip":
        arch = str(getattr(cfg.MODEL.CLIP, "ARCH", "")).lower()
        if "vit-b" in arch or "rn50" in arch or "rn101" in arch:
            return 512
        if "vit-l" in arch:
            return 768

    encoder_cfg = getattr(cfg.MODEL, name, None)
    if encoder_cfg is None:
        return int(getattr(cfg.MODEL.STM, "HIDDEN_DIM", 256))
    return int(getattr(encoder_cfg, "EMBED_DIM", getattr(cfg.MODEL.STM, "HIDDEN_DIM", 256)))


class AdaptiveSTSamplingMixing(nn.Module):
    """
    Adaptive Spatio-Temporal Sampling Mixing.
    For image mode, temporal sampling is disabled.
    """

    def __init__(self, spatial_points=32,
                 temporal_points=4,
                 out_multiplier=4,
                 n_groups=4,
                 query_dim=256,
                 feat_channels=None,
                 pretrained_action=False,
                 image_mode=True):  # Default to image mode
        super(AdaptiveSTSamplingMixing, self).__init__()
        self.spatial_points = spatial_points
        self.temporal_points = temporal_points
        self.out_multiplier = out_multiplier
        self.n_groups = n_groups
        self.query_dim = query_dim
        self.feat_channels = feat_channels if feat_channels is not None else self.query_dim
        self.offset_generator = nn.Sequential(nn.Linear(query_dim, spatial_points * n_groups * 3))
        self.pretrained_action = pretrained_action
        self.image_mode = image_mode

        self.norm_s = nn.LayerNorm(query_dim)
        self.adaptive_mixing_s = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.query_dim,
            in_points=self.spatial_points,
            out_points=self.spatial_points*self.out_multiplier,
            n_groups=self.n_groups,
        )

        # For image mode, disable temporal mixing
        if not self.pretrained_action and not self.image_mode:
            self.norm_t = nn.LayerNorm(query_dim)
            self.adaptive_mixing_t = AdaptiveMixing(
                self.feat_channels,
                query_dim=self.query_dim,
                in_points=self.temporal_points,
                out_points=self.temporal_points*self.out_multiplier,
                n_groups=self.n_groups,
            )
            self.attention_t = MultiheadAttention(query_dim, 8, 0.0)
        else:
            self.norm_t = None
            self.adaptive_mixing_t = None
            self.attention_t = None

        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.offset_generator[-1].weight.data)
        nn.init.zeros_(self.offset_generator[-1].bias.data)

        bias = self.offset_generator[-1].bias.data.view(
            self.n_groups, self.spatial_points, 3)

        if int(self.spatial_points ** 0.5) ** 2 == self.spatial_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)
        bias[:, :, 2:3].mul_(0.0)

        self.adaptive_mixing_s._init_weights()
        if self.adaptive_mixing_t is not None:
            self.adaptive_mixing_t._init_weights()

    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides):

        offset = self.offset_generator(spatial_queries)
        sample_points_xy = make_sample_points(offset, self.n_groups * self.spatial_points, proposal_boxes)
        sampled_feature, _ = SAMPLE4D(sample_points_xy, features, featmap_strides=featmap_strides, n_points=self.spatial_points)

        # For image mode: sampled_feature shape is [B, C, n_groups, 1, spatial_points, n_query]
        # For video mode: sampled_feature shape is [B, C, n_groups, temporal_points, spatial_points, n_query]
        
        sampled_feature = sampled_feature.flatten(5, 6)                   
        sampled_feature = sampled_feature.permute(0, 5, 2, 3, 4, 1)

        # Spatial mixing (always enabled)
        spatial_feats = torch.mean(sampled_feature, dim=3)  # [B, n_query, n_groups, spatial_points, n_channels]
        spatial_queries = self.adaptive_mixing_s(spatial_feats, spatial_queries)
        spatial_queries = self.norm_s(spatial_queries)

        # For image mode, temporal_queries is None, skip temporal mixing
        if self.image_mode or temporal_queries is None or self.attention_t is None:
            # Just return spatial queries, temporal is disabled for image mode
            return spatial_queries, None

        # Temporal mixing (video mode only)
        if sampled_feature.size(3) == 1:
            temporal_queries = temporal_queries
        else:
            temporal_feats = torch.mean(sampled_feature, dim=4)
            temporal_queries = self.adaptive_mixing_t(temporal_feats, temporal_queries)
            temporal_queries = self.norm_t(temporal_queries)

        return spatial_queries, temporal_queries


class AMStage(nn.Module):
    """
    Attention Mixing Stage for Image Multimodal.
    Temporal branch is disabled for image mode.
    """

    def __init__(self, query_dim=256,
                 feat_channels=256,
                 num_heads=8,
                 feedforward_channels=2048,
                 dropout=0.0,
                 num_ffn_fcs=2,
                 ffn_act='RelU',
                 spatial_points=32,
                 temporal_points=4,
                 out_multiplier=4,
                 n_groups=4,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 num_action_fcs=1,
                 num_classes_object=1,
                 num_classes_action=80,
                 open_vocabulary=False,
                 text_dim=768,
                 cond_cls=False, 
                 fuse_cls=False,
                 fuse_factor=-1,
                 fuse_method='logit_fusion',
                 pretrained_action=False,
                 num_queries=100,
                 dest=False,
                 image_mode=True,  # Default to image mode
                 ):



        super(AMStage, self).__init__()

        ffn_act_cfg = dict(type=ffn_act, inplace=True)
        self.attention_s = MultiheadAttention(query_dim, num_heads, dropout)
        self.attention_norm_s = nn.LayerNorm(query_dim, eps=1e-5)
        self.ffn_s = FFN(query_dim, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
        self.ffn_norm_s = nn.LayerNorm(query_dim, eps=1e-5)
        self.iof_tau = nn.Parameter(torch.ones(self.attention_s.num_heads,))
        self.cond_cls = cond_cls
        self.fuse_cls = fuse_cls
        self.fuse_method = fuse_method
        self.dest = dest
        self.image_mode = image_mode
        self.detection_only = True

        # Build temporal branch ONLY for video (not image mode)
        if not pretrained_action and not image_mode:
            self.attention_t = MultiheadAttention(query_dim, num_heads, dropout)
            self.attention_norm_t = nn.LayerNorm(query_dim, eps=1e-5)
            self.ffn_t = FFN(query_dim, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
            self.ffn_norm_t = nn.LayerNorm(query_dim, eps=1e-5)
        else:
            self.attention_t = None
            self.attention_norm_t = None
            self.ffn_t = None
            self.ffn_norm_t = None

        self.samplingmixing = AdaptiveSTSamplingMixing(
            spatial_points=spatial_points,
            temporal_points=temporal_points,
            out_multiplier=out_multiplier,
            n_groups=n_groups,
            query_dim=query_dim,
            feat_channels=feat_channels,
            pretrained_action=pretrained_action,
            image_mode=image_mode,
        )

        cls_feature_dim = query_dim
        reg_feature_dim = query_dim
        action_feat_dim = query_dim * 2 if not self.dest else query_dim

        # human classifier
        self.human_cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.human_cls_fcs.append(
                nn.Linear(cls_feature_dim, cls_feature_dim, bias=True))
            self.human_cls_fcs.append(
                nn.LayerNorm(cls_feature_dim, eps=1e-5))
            self.human_cls_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.human_fc_cls = nn.Linear(cls_feature_dim, num_classes_object + 1)

        # human bbox regressor
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(reg_feature_dim, reg_feature_dim, bias=True))
            self.reg_fcs.append(
                nn.LayerNorm(reg_feature_dim, eps=1e-5))
            self.reg_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.fc_reg = nn.Linear(reg_feature_dim, 4)

        self.open_vocabulary = False

        if not self.detection_only:
            if dropout > 0:
                self.dropout = nn.Dropout(dropout)

            if not open_vocabulary:
                self.action_cls_fcs = nn.ModuleList()
                for _ in range(num_action_fcs):
                    self.action_cls_fcs.append(nn.Linear(action_feat_dim, action_feat_dim, bias=True))
                    self.action_cls_fcs.append(nn.LayerNorm(action_feat_dim, eps=1e-5))
                    self.action_cls_fcs.append(_get_activation_layer(ffn_act_cfg))

                self.fc_action = nn.Linear(action_feat_dim, num_classes_action)

            else:
                if not pretrained_action:
                    self.linear_proj = nn.Linear(action_feat_dim, text_dim, bias=False) if action_feat_dim != text_dim else nn.Identity()

            if self.fuse_cls and self.fuse_method == 'logit_fusion':
                self.logit_alpha = nn.Parameter(torch.ones(num_queries,) * fuse_factor)


        
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                pass
        bias_init = bias_init_with_prob(0.01)
        if hasattr(self, 'fc_action'):
            nn.init.constant_(self.fc_action.bias, bias_init)
        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
        self.samplingmixing.init_weights()


    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries=None, 
                featmap_strides=[4, 8, 16, 32], text_features=None, tau_inv=100, vis_cls_feat=None, patch_feat=None, text_token_feats=None, labels=None, cond=None):

        N, n_query = spatial_queries.shape[:2]

        with torch.no_grad():
            rois = decode_box(proposal_boxes)
            roi_box_batched = rois.view(N, n_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[:, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(proposal_boxes, spatial_queries.size(-1) // 4)

        # IoF attention bias
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)).flatten(0, 1)
        pe = pe.permute(1, 0, 2)

        # Spatial attention (always enabled)
        spatial_queries = spatial_queries.permute(1, 0, 2)
        spatial_queries_attn = spatial_queries + pe
        spatial_queries = self.attention_s(spatial_queries_attn, attn_mask=attn_bias,)
        spatial_queries = self.attention_norm_s(spatial_queries)
        spatial_queries = spatial_queries.permute(1, 0, 2)

        # Temporal attention (disabled for image mode)
        if temporal_queries is not None and not self.image_mode:
            temporal_queries = temporal_queries.permute(1, 0, 2)
            temporal_queries_attn = temporal_queries + pe
            temporal_queries = self.attention_t(temporal_queries_attn, attn_mask=attn_bias,)
            temporal_queries = self.attention_norm_t(temporal_queries)
            temporal_queries = temporal_queries.permute(1, 0, 2)
        else:
            temporal_queries = None

        if self.cond_cls and temporal_queries is not None:
            temporal_queries = temporal_queries + cond.unsqueeze(1)

        # Sampling mixing (handles both image and video internally)
        spatial_queries, temporal_queries = \
            self.samplingmixing(features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides)

        spatial_queries = self.ffn_s(spatial_queries)
        if temporal_queries is not None and not self.image_mode:
            temporal_queries = self.ffn_t(temporal_queries)

        # Layer normalization
        spatial_queries = self.ffn_norm_s(spatial_queries)
        if temporal_queries is not None and not self.image_mode:
            temporal_queries = self.ffn_norm_t(temporal_queries)

        # Heads
        cls_feat = spatial_queries
        for cls_layer in self.human_cls_fcs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.human_fc_cls(cls_feat).view(N, n_query, -1)

        reg_feat = spatial_queries
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        xyzr_delta = self.fc_reg(reg_feat).view(N, n_query, -1)

        # Action head (simplified for image mode)
        if self.open_vocabulary:
            # Image mode: use spatial features only
            action_score = None  # Simplified for image mode
        else:
            # Fallback
            action_score = None

        spatial_queries = spatial_queries.view(N, n_query, -1)
        if temporal_queries is not None:
            temporal_queries = temporal_queries.view(N, n_query, -1)
        
        return cls_score, action_score, xyzr_delta, spatial_queries, temporal_queries



class STMDecoder(nn.Module):
    """
    STM Decoder for Image Multimodal Models.
    Supports both video and image modes.
    """

    def __init__(self, cfg, image_mode=True):  # Default to image mode

        super(STMDecoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_mode = image_mode

        self.open_vocabulary = cfg.DATA.OPEN_VOCABULARY
        self.multi_label_action = cfg.MODEL.MULTI_LABEL_ACTION
        self.use_pretrained_action = cfg.MODEL.STM.PRETRAIN_ACTION
        if self.use_pretrained_action:
            assert self.open_vocabulary
        
        self._generate_queries(cfg)

        self.text_encoder_name = _resolve_text_encoder_name(cfg)
        text_encoder_cfg = getattr(cfg.MODEL, self.text_encoder_name, None)
        if text_encoder_cfg is None:
            raise AttributeError(f"MODEL.{self.text_encoder_name} is not defined in config")

        self.cam_sampling = getattr(text_encoder_cfg, 'CAM_SAMPLING', 'topk')
        self.text_dim = _infer_text_dim(cfg, self.text_encoder_name)
        self.num_heads = cfg.MODEL.STM.NUM_HEADS

        self.num_stages = cfg.MODEL.STM.NUM_STAGES
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            decoder_stage = AMStage(
                query_dim=cfg.MODEL.STM.HIDDEN_DIM,
                feat_channels=cfg.MODEL.STM.HIDDEN_DIM,
                num_heads=cfg.MODEL.STM.NUM_HEADS,
                feedforward_channels=cfg.MODEL.STM.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.STM.DROPOUT,
                num_ffn_fcs=cfg.MODEL.STM.NUM_FCS,
                ffn_act=cfg.MODEL.STM.ACTIVATION,
                spatial_points=cfg.MODEL.STM.SPATIAL_POINTS,
                temporal_points=cfg.MODEL.STM.TEMPORAL_POINTS,
                out_multiplier=cfg.MODEL.STM.OUT_MULTIPLIER,
                n_groups=cfg.MODEL.STM.N_GROUPS,
                num_cls_fcs=cfg.MODEL.STM.NUM_CLS,
                num_reg_fcs=cfg.MODEL.STM.NUM_REG,
                num_action_fcs=cfg.MODEL.STM.NUM_ACT,
                num_classes_object=cfg.MODEL.STM.OBJECT_CLASSES,
                num_classes_action=cfg.MODEL.STM.ACTION_CLASSES,
                open_vocabulary=cfg.DATA.OPEN_VOCABULARY,
                text_dim=self.text_dim,
                cond_cls=cfg.MODEL.STM.COND_CLS,
                fuse_cls=cfg.MODEL.STM.FUSE_CLS,
                fuse_factor=cfg.MODEL.STM.FUSE_FACTOR,
                fuse_method=cfg.MODEL.STM.FUSE_METHOD,
                pretrained_action=self.use_pretrained_action,
                num_queries=self.num_queries,
                dest=cfg.MODEL.STM.DeST,
                image_mode=image_mode,  # Pass image_mode
                )
            self.decoder_stages.append(decoder_stage)

        object_weight = cfg.MODEL.STM.OBJECT_WEIGHT
        giou_weight   = cfg.MODEL.STM.GIOU_WEIGHT
        l1_weight     = cfg.MODEL.STM.L1_WEIGHT
        background_weight = cfg.MODEL.STM.BACKGROUND_WEIGHT

        self.weight_dict = {
            "loss_ce": object_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight
        }

        use_focal = False
        
        self.score_threshold = cfg.MODEL.STM.SCORE_THRESHOLD

        self.cond_cls = cfg.MODEL.STM.COND_CLS
        self.fuse_cls = cfg.MODEL.STM.FUSE_CLS
        self.cond_type = cfg.MODEL.STM.COND_MODALITY

        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=object_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight)

        self.intermediate_supervision = cfg.MODEL.STM.INTERMEDIATE_SUPERVISION
        if self.intermediate_supervision:
            for i in range(self.num_stages - 1):
                inter_weight_dict = {k + f"_{i}": v for k, v in self.weight_dict.items()}
                self.weight_dict.update(inter_weight_dict)

        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=cfg.MODEL.STM.OBJECT_CLASSES,
                                      matcher=matcher,
                                      eos_coef=background_weight,
                                      losses=losses,
                                      use_focal=use_focal)

    def _generate_queries(self, cfg):
        self.num_queries = cfg.MODEL.STM.NUM_QUERIES
        self.hidden_dim = cfg.MODEL.STM.HIDDEN_DIM

        # Build spatial queries (always enabled)
        self.init_spatial_queries = nn.Embedding(self.num_queries, self.hidden_dim)
        
        # Build temporal queries only for video mode (not image mode)
        if not self.use_pretrained_action and not self.image_mode:
            self.init_temporal_queries = nn.Embedding(self.num_queries, self.hidden_dim)
        else:
            self.init_temporal_queries = None

    def _box_init(self, whwh, extras={}):
        num_queries = self.num_queries
        batch_size = len(whwh)

        if 'prior_boxes' in extras:
            proposals = [box_sampling_from_prior(extras['prior_boxes'][b], num_boxes=num_queries, device=whwh.device)
                         for b in range(batch_size)]
            proposals = torch.stack(proposals, dim=0)
        elif 'cams' in extras:
            if self.cam_sampling == 'topk':
                prior_map = extras.get('prior_map', None)
                proposals = [box_sampling_from_heatmap(extras['cams'][b], prior_map=prior_map, num_boxes=num_queries) 
                            for b in range(batch_size)]
            else:
                raise NotImplementedError
            proposals = torch.stack(proposals, dim=0)
        else:
            proposals = torch.ones(num_queries, 4, dtype=torch.float, device=self.device, requires_grad=False)
            proposals[:, :2] = 0.5
            proposals = box_cxcywh_to_xyxy(proposals)

            whwh = whwh[:, None, :]
            proposals = proposals[None] * whwh

        xyzr = box_xyxy_to_xyzr(proposals)
        xyzr = xyzr.detach()

        return xyzr


    def _decode_init_queries(self, whwh, cond=None, extras={}):
        
        batch_size = len(whwh)
        xyzr = self._box_init(whwh, extras)

        init_spatial_queries = self.init_spatial_queries.weight.clone()
        init_spatial_queries = init_spatial_queries[None].expand(batch_size, *init_spatial_queries.size())

        # For image mode, temporal queries are None
        init_temporal_queries = None
        if self.init_temporal_queries is not None:
            init_temporal_queries = self.init_temporal_queries.weight.clone()
            init_temporal_queries = init_temporal_queries[None].expand(batch_size, *init_temporal_queries.size())
            
        if self.cond_cls and init_temporal_queries is not None:
            assert self.hidden_dim == cond.size(-1), \
                "cfg.MODEL.STM.HIDDEN_DIM should be set to {} when conditioning on pretrained CLS visual feature".format(cond.size(-1))
            init_temporal_queries = cond.unsqueeze(1) + init_temporal_queries

        # Normalization
        init_spatial_queries = torch.layer_norm(init_spatial_queries,
                                                normalized_shape=[init_spatial_queries.size(-1)])
        if init_temporal_queries is not None:
            init_temporal_queries = torch.layer_norm(init_temporal_queries,
                                                        normalized_shape=[init_temporal_queries.size(-1)])
    
        return xyzr, init_spatial_queries, init_temporal_queries


    def forward(
        self,
        features,
        whwh,
        gt_boxes=None,
        labels=None,
        extras={},
        part_forward=-1,
        text_features=None,
        tau_inv=100,
        cls_feat=None,
        patch_feat=None,
        text_token_feats=None
    ):

        # Optional conditioning
        cond = None
        if self.cond_cls:
            cond = cls_feat if self.cond_type == 'visual' else self.get_prematched_text(cls_feat, text_features, labels)

        # Initialize queries
        proposal_boxes, spatial_queries, temporal_queries = self._decode_init_queries(
            whwh, cond=cond, extras=extras
        )

        inter_class_logits = []
        inter_pred_bboxes = []

        B, N, _ = spatial_queries.size()

        # Decoder stages
        for decoder_stage in self.decoder_stages:

            cls_logits, _, delta_xyzr, spatial_queries, temporal_queries = decoder_stage(
                features,
                proposal_boxes,
                spatial_queries,
                temporal_queries,
                text_features=text_features,
                tau_inv=tau_inv,
                vis_cls_feat=cls_feat,
                patch_feat=patch_feat,
                text_token_feats=text_token_feats,
                labels=labels,
                cond=cond
           )

            proposal_boxes, pred_boxes = refine_xyzr(proposal_boxes, delta_xyzr)

            inter_class_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_boxes)

        # Inference mode
        if not self.training:
            logits = inter_class_logits[-1]
            boxes  = inter_pred_bboxes[-1]

            probs = logits.softmax(-1)
            scores, labels = probs[..., :-1].max(-1)

            results = []
            for i in range(B):
                keep = torch.where(scores[i] > self.score_threshold)[0]

                if keep.numel() == 0:
                    k = min(10, scores.size(1))
                    keep = torch.topk(scores[i], k=k).indices

                cur_boxes = boxes[i][keep]
                cur_scores = scores[i][keep]
                cur_labels = labels[i][keep]

                h, w = whwh[i][1], whwh[i][0]
                cur_boxes = cur_boxes.clone()
                cur_boxes[:, 0::2] /= w
                cur_boxes[:, 1::2] /= h

                results.append({
                    "scores": cur_scores,
                    "labels": cur_labels,
                    "boxes": cur_boxes
                })

            return results

        # Training mode
        targets = self.make_targets(gt_boxes, whwh, labels)

        output = {
            "pred_logits": inter_class_logits[-1],
            "pred_boxes": inter_pred_bboxes[-1],
        }

        losses = self.criterion(output, targets)

        for k in losses.keys():
            if k in self.weight_dict:
               losses[k] *= self.weight_dict[k]

        return losses

    def make_targets(self, gt_boxes, whwh, labels):
        """Universal targets builder for image detection."""
        targets = []
        for boxes_img, frame_size, label in zip(gt_boxes, whwh, labels):
            target = {}
            boxes = torch.tensor(boxes_img, dtype=torch.float32, device=self.device)
            target["boxes_xyxy"] = boxes

            if label is None or len(label) == 0:
                class_ids = torch.zeros(len(boxes), dtype=torch.int64, device=self.device)
            else:
                label = torch.tensor(label, device=self.device)
                if label.ndim == 1:
                    class_ids = label.long()
                else:
                    class_ids = torch.argmax(label, dim=1).long()

            target["labels"] = class_ids
            target["image_size_xyxy"] = frame_size.to(self.device)
            target["image_size_xyxy_tgt"] = frame_size.unsqueeze(0).repeat(len(boxes), 1).to(self.device)

            if not self.use_pretrained_action:
                if label.ndim == 2:
                    target["action_labels"] = label.float()
                else:
                    num_classes = self.criterion.num_classes
                    target["action_labels"] = torch.nn.functional.one_hot(
                        class_ids, num_classes=num_classes
                    ).float()

            targets.append(target)

        return targets

    def get_prematched_text(self, vis_cls_feat, text_features, labels=None):
        if isinstance(text_features, list):
            text_feat = torch.stack(text_features, dim=0).mean(dim=1)
        else:
            text_feat = text_features

        text_feat = text_feat.to(vis_cls_feat.device)

        if labels is not None and len(labels) > 0:
            classes = []
            for lab in labels:
                lab = torch.tensor(lab, device=vis_cls_feat.device)
                if lab.ndim == 1:
                    cls = lab[0].long()
                elif lab.ndim == 2:
                    cls = torch.argmax(lab[0]).long()
                else:
                    cls = torch.tensor(0, device=vis_cls_feat.device)
                classes.append(cls)
            classes = torch.stack(classes)
            text_prematched = text_feat[classes]
        else:
            text_norm = text_feat / text_feat.norm(dim=-1, keepdim=True)
            vis_norm = vis_cls_feat / vis_cls_feat.norm(dim=-1, keepdim=True)
            pred_cls = torch.argmax(vis_norm @ text_norm.t(), dim=-1)
            text_prematched = text_feat[pred_cls]

        return text_prematched

    
def build_stm_decoder(cfg, image_mode=True):
    """
    Build STM Decoder for image multimodal.
    
    Args:
        cfg: Configuration
        image_mode: If True, build for image mode (no temporal processing)
    
    Returns:
        STMDecoder instance
    """
    return STMDecoder(cfg, image_mode=image_mode)
