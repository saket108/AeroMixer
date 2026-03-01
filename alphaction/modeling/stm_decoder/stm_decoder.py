"""
STM Decoder for Image Multimodal Models.
Converted to support image + text multimodal (temporal concepts removed/disabled).
"""

import math
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
                 num_severity_fcs=1,
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
                 predict_severity=False,
                 iof_tau_mode="learned",
                 iof_tau_fixed=0.0,
                 iof_tau_clamp_min=0.0,
                 iof_tau_clamp_max=4.0,
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
        self.predict_severity = predict_severity
        self.iof_tau_mode = str(iof_tau_mode).lower()
        self.iof_tau_fixed = float(iof_tau_fixed)
        self.iof_tau_clamp_min = float(iof_tau_clamp_min)
        self.iof_tau_clamp_max = float(iof_tau_clamp_max)

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

        self.severity_fcs = nn.ModuleList()
        if self.predict_severity:
            for _ in range(num_severity_fcs):
                self.severity_fcs.append(
                    nn.Linear(reg_feature_dim, reg_feature_dim, bias=True))
                self.severity_fcs.append(
                    nn.LayerNorm(reg_feature_dim, eps=1e-5))
                self.severity_fcs.append(
                    _get_activation_layer(ffn_act_cfg))
            self.fc_severity = nn.Linear(reg_feature_dim, 1)
        else:
            self.fc_severity = None

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
        if self.fc_severity is not None:
            nn.init.zeros_(self.fc_severity.weight)
            nn.init.zeros_(self.fc_severity.bias)
        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
        self.samplingmixing.init_weights()

    @torch.no_grad()
    def _summarize_attention(self, attn_weights):
        if attn_weights is None:
            return {}

        weights = attn_weights.detach()
        if weights.ndim == 3:
            # Older torch may return [B, Q, K] when averaged across heads.
            weights = weights[:, None, :, :]

        eps = 1e-9
        entropy = -(weights * (weights + eps).log()).sum(dim=-1)
        entropy = entropy / math.log(max(weights.size(-1), 2))
        diag = weights.diagonal(dim1=-2, dim2=-1)
        top1 = weights.max(dim=-1).values

        return {
            "entropy": float(entropy.mean().item()),
            "diag": float(diag.mean().item()),
            "top1": float(top1.mean().item()),
        }

    def _resolve_iof_tau(self):
        mode = self.iof_tau_mode
        if mode == "learned":
            return self.iof_tau
        if mode == "zero":
            return torch.zeros_like(self.iof_tau)
        if mode == "fixed":
            return torch.full_like(self.iof_tau, self.iof_tau_fixed)
        if mode == "clamp":
            return self.iof_tau.clamp(min=self.iof_tau_clamp_min, max=self.iof_tau_clamp_max)
        raise ValueError(
            f"Unsupported IOF_TAU_MODE='{self.iof_tau_mode}'. "
            "Use 'learned', 'zero', 'fixed', or 'clamp'."
        )


    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries=None, 
                featmap_strides=[4, 8, 16, 32], text_features=None, tau_inv=100, vis_cls_feat=None, patch_feat=None, text_token_feats=None, labels=None, cond=None,
                collect_attn_stats=False, compare_nomask=False):

        N, n_query = spatial_queries.shape[:2]
        attn_stats = None

        with torch.no_grad():
            rois = decode_box(proposal_boxes)
            roi_box_batched = rois.view(N, n_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[:, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(proposal_boxes, spatial_queries.size(-1) // 4)

        # IoF attention bias
        effective_tau = self._resolve_iof_tau()
        attn_bias = (iof * effective_tau.view(1, -1, 1, 1)).flatten(0, 1)
        pe = pe.permute(1, 0, 2)

        # Spatial attention (always enabled)
        spatial_queries = spatial_queries.permute(1, 0, 2)
        spatial_queries_attn = spatial_queries + pe
        if collect_attn_stats:
            spatial_queries, attn_weights = self.attention_s(
                spatial_queries_attn,
                attn_mask=attn_bias,
                return_attn_weights=True,
                average_attn_weights=False,
            )
            attn_stats = self._summarize_attention(attn_weights)
            tau_vals = effective_tau.detach()
            bias_vals = attn_bias.detach()
            attn_stats.update(
                {
                    "tau_min": float(tau_vals.min().item()),
                    "tau_mean": float(tau_vals.mean().item()),
                    "tau_max": float(tau_vals.max().item()),
                    "bias_min": float(bias_vals.min().item()),
                    "bias_mean": float(bias_vals.mean().item()),
                    "bias_max": float(bias_vals.max().item()),
                }
            )

            if compare_nomask:
                with torch.no_grad():
                    _, attn_weights_nomask = self.attention_s(
                        spatial_queries_attn,
                        attn_mask=None,
                        return_attn_weights=True,
                        average_attn_weights=False,
                    )
                nomask_stats = self._summarize_attention(attn_weights_nomask)
                attn_stats.update(
                    {
                        "entropy_nomask": nomask_stats.get("entropy", 0.0),
                        "diag_nomask": nomask_stats.get("diag", 0.0),
                        "top1_nomask": nomask_stats.get("top1", 0.0),
                    }
                )
        else:
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

        severity_score = None
        if self.predict_severity and self.fc_severity is not None:
            severity_feat = reg_feat
            for sev_layer in self.severity_fcs:
                severity_feat = sev_layer(severity_feat)
            severity_score = self.fc_severity(severity_feat).view(N, n_query)

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
        
        return cls_score, action_score, severity_score, xyzr_delta, spatial_queries, temporal_queries, attn_stats



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
        self.predict_severity = cfg.MODEL.STM.PREDICT_SEVERITY
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
                num_severity_fcs=cfg.MODEL.STM.NUM_SEV,
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
                predict_severity=self.predict_severity,
                iof_tau_mode=cfg.MODEL.STM.IOF_TAU_MODE,
                iof_tau_fixed=cfg.MODEL.STM.IOF_TAU_FIXED,
                iof_tau_clamp_min=cfg.MODEL.STM.IOF_TAU_CLAMP_MIN,
                iof_tau_clamp_max=cfg.MODEL.STM.IOF_TAU_CLAMP_MAX,
                )
            self.decoder_stages.append(decoder_stage)

        object_weight = cfg.MODEL.STM.OBJECT_WEIGHT
        giou_weight   = cfg.MODEL.STM.GIOU_WEIGHT
        l1_weight     = cfg.MODEL.STM.L1_WEIGHT
        severity_weight = cfg.MODEL.STM.SEVERITY_WEIGHT
        background_weight = cfg.MODEL.STM.BACKGROUND_WEIGHT

        base_weight_dict = {
            "loss_ce": object_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight
        }
        if self.predict_severity:
            base_weight_dict["loss_severity"] = severity_weight
        self.weight_dict = dict(base_weight_dict)

        use_focal = False
        
        self.score_threshold = cfg.MODEL.STM.SCORE_THRESHOLD

        self.cond_cls = cfg.MODEL.STM.COND_CLS
        self.fuse_cls = cfg.MODEL.STM.FUSE_CLS
        self.cond_type = cfg.MODEL.STM.COND_MODALITY
        self.attn_telemetry = bool(getattr(cfg.MODEL.STM, "ATTN_TELEMETRY", False))
        self.attn_telemetry_stagewise = bool(getattr(cfg.MODEL.STM, "ATTN_TELEMETRY_STAGEWISE", True))
        self.attn_telemetry_compare_nomask = bool(
            getattr(cfg.MODEL.STM, "ATTN_TELEMETRY_COMPARE_NOMASK", False)
        )
        self.last_attn_metrics = {}

        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=object_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight)

        self.intermediate_supervision = cfg.MODEL.STM.INTERMEDIATE_SUPERVISION
        if self.intermediate_supervision:
            for i in range(self.num_stages - 1):
                inter_weight_dict = {f"{k}_{i}": v for k, v in base_weight_dict.items()}
                self.weight_dict.update(inter_weight_dict)

        losses = ["labels", "boxes"]
        if self.predict_severity:
            losses.append("severity")
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=cfg.MODEL.STM.OBJECT_CLASSES,
                                      matcher=matcher,
                                      eos_coef=background_weight,
                                      losses=losses,
                                      use_focal=use_focal)

    def _generate_queries(self, cfg):
        self.num_queries = cfg.MODEL.STM.NUM_QUERIES
        self.hidden_dim = cfg.MODEL.STM.HIDDEN_DIM
        self.query_init_mode = str(getattr(cfg.MODEL.STM, "QUERY_INIT_MODE", "learnable_anchors")).lower()
        self.query_init_base_scale = float(getattr(cfg.MODEL.STM, "QUERY_INIT_BASE_SCALE", 0.2))
        self.query_init_min_scale = float(getattr(cfg.MODEL.STM, "QUERY_INIT_MIN_SCALE", 0.02))
        self.query_init_max_scale = float(getattr(cfg.MODEL.STM, "QUERY_INIT_MAX_SCALE", 0.60))
        self.query_init_center_offset = float(getattr(cfg.MODEL.STM, "QUERY_INIT_CENTER_OFFSET", 0.25))
        self.query_init_log_wh_clamp = float(getattr(cfg.MODEL.STM, "QUERY_INIT_LOG_WH_CLAMP", 2.0))
        self.query_init_small_object_bias = bool(getattr(cfg.MODEL.STM, "QUERY_INIT_SMALL_OBJECT_BIAS", True))
        self.query_init_small_object_scale = float(getattr(cfg.MODEL.STM, "QUERY_INIT_SMALL_OBJECT_SCALE", 0.70))

        # Build spatial queries (always enabled)
        self.init_spatial_queries = nn.Embedding(self.num_queries, self.hidden_dim)

        # Learnable anchor-like priors used by default query initialization.
        base_xy, base_wh = self._build_anchor_priors(self.num_queries, self.query_init_base_scale)
        self.register_buffer("query_anchor_base_xy", base_xy)
        self.register_buffer("query_anchor_base_wh", base_wh)
        self.query_anchor_center_delta = nn.Parameter(torch.zeros(self.num_queries, 2))
        self.query_anchor_log_wh_delta = nn.Parameter(torch.zeros(self.num_queries, 2))
        
        # Build temporal queries only for video mode (not image mode)
        if not self.use_pretrained_action and not self.image_mode:
            self.init_temporal_queries = nn.Embedding(self.num_queries, self.hidden_dim)
        else:
            self.init_temporal_queries = None

    def _build_anchor_priors(self, num_queries, base_scale):
        rows = int(math.floor(math.sqrt(num_queries)))
        cols = int(math.ceil(float(num_queries) / max(rows, 1)))
        ys = (torch.arange(rows, dtype=torch.float32) + 0.5) / max(rows, 1)
        xs = (torch.arange(cols, dtype=torch.float32) + 0.5) / max(cols, 1)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)[:num_queries]
        base_wh = torch.full((num_queries, 2), float(base_scale), dtype=torch.float32)
        return base_xy, base_wh

    def _box_init(self, whwh, extras=None):
        if extras is None:
            extras = {}
        num_queries = self.num_queries
        batch_size = len(whwh)
        device = whwh.device

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
            if self.query_init_mode == "full_image":
                proposals = torch.ones(num_queries, 4, dtype=torch.float32, device=device, requires_grad=False)
                proposals[:, :2] = 0.5
                proposals = box_cxcywh_to_xyxy(proposals)
                proposals = proposals[None] * whwh[:, None, :]
            elif self.query_init_mode == "learnable_anchors":
                centers = self.query_anchor_base_xy.to(device=device) + torch.tanh(
                    self.query_anchor_center_delta.to(device=device)
                ) * self.query_init_center_offset
                centers = centers.clamp(0.01, 0.99)

                wh = self.query_anchor_base_wh.to(device=device) * torch.exp(
                    self.query_anchor_log_wh_delta.to(device=device).clamp(
                        min=-self.query_init_log_wh_clamp, max=self.query_init_log_wh_clamp
                    )
                )
                if self.query_init_small_object_bias:
                    wh = wh * self.query_init_small_object_scale
                wh = wh.clamp(min=self.query_init_min_scale, max=self.query_init_max_scale)

                proposals_norm = torch.cat([centers, wh], dim=-1)
                proposals_norm = box_cxcywh_to_xyxy(proposals_norm).clamp(0.0, 1.0)
                proposals = proposals_norm[None] * whwh[:, None, :]
            else:
                raise ValueError(
                    f"Unsupported QUERY_INIT_MODE='{self.query_init_mode}'. "
                    "Use 'learnable_anchors' or 'full_image'."
                )

        xyzr = box_xyxy_to_xyzr(proposals)
        xyzr = xyzr.detach()

        return xyzr


    def _decode_init_queries(self, whwh, cond=None, extras=None):
        if extras is None:
            extras = {}
        
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

    def _build_attention_metrics(self, stage_stats, device):
        if not stage_stats:
            return {}

        valid_stats = [stat for stat in stage_stats if isinstance(stat, dict) and len(stat) > 0]
        if not valid_stats:
            return {}

        metrics = {}
        metric_keys = sorted({key for stat in valid_stats for key in stat.keys()})
        for key in metric_keys:
            vals = [float(stat[key]) for stat in valid_stats if key in stat]
            if not vals:
                continue
            metrics[f"attn_{key}_avg"] = torch.tensor(
                sum(vals) / len(vals),
                device=device,
                dtype=torch.float32,
            )

        if self.attn_telemetry_stagewise:
            for stage_idx, stat in enumerate(stage_stats):
                if not isinstance(stat, dict):
                    continue
                for key, value in stat.items():
                    metrics[f"attn_s{stage_idx}_{key}"] = torch.tensor(
                        float(value),
                        device=device,
                        dtype=torch.float32,
                    )

        return metrics


    def forward(
        self,
        features,
        whwh,
        gt_boxes=None,
        labels=None,
        extras=None,
        part_forward=-1,
        text_features=None,
        tau_inv=100,
        cls_feat=None,
        patch_feat=None,
        text_token_feats=None
    ):
        if extras is None:
            extras = {}

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
        inter_pred_severity = []
        stage_attn_stats = []

        B, N, _ = spatial_queries.size()

        # Decoder stages
        for decoder_stage in self.decoder_stages:

            cls_logits, _, severity_pred, delta_xyzr, spatial_queries, temporal_queries, attn_stats = decoder_stage(
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
                cond=cond,
                collect_attn_stats=self.attn_telemetry,
                compare_nomask=self.attn_telemetry_compare_nomask,
           )

            proposal_boxes, pred_boxes = refine_xyzr(proposal_boxes, delta_xyzr)

            inter_class_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_boxes)
            inter_pred_severity.append(severity_pred)
            stage_attn_stats.append(attn_stats)

        attn_metrics = {}
        if self.attn_telemetry:
            attn_metrics = self._build_attention_metrics(stage_attn_stats, device=whwh.device)
            refine_l1_terms = []
            if len(inter_pred_bboxes) > 1:
                whwh_view = whwh[:, None, :]
                for stage_idx in range(1, len(inter_pred_bboxes)):
                    prev_boxes = inter_pred_bboxes[stage_idx - 1] / whwh_view
                    cur_boxes = inter_pred_bboxes[stage_idx] / whwh_view
                    delta_l1 = (cur_boxes - prev_boxes).abs().mean()
                    attn_metrics[f"refine_l1_s{stage_idx}"] = delta_l1.detach()
                    refine_l1_terms.append(delta_l1.detach())
                if refine_l1_terms:
                    attn_metrics["refine_l1_avg"] = torch.stack(refine_l1_terms).mean()
            self.last_attn_metrics = {k: float(v.detach().cpu().item()) for k, v in attn_metrics.items()}
        else:
            self.last_attn_metrics = {}

        # Inference mode
        if not self.training:
            logits = inter_class_logits[-1]
            boxes  = inter_pred_bboxes[-1]
            severity_scores = inter_pred_severity[-1]

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

                result_i = {
                    "scores": cur_scores,
                    "labels": cur_labels,
                    "boxes": cur_boxes
                }
                if severity_scores is not None:
                    cur_severity = torch.sigmoid(severity_scores[i][keep])
                    result_i["severity"] = cur_severity
                results.append(result_i)

            return results

        # Training mode
        targets = self.make_targets(gt_boxes, whwh, labels, extras=extras)

        output = {
            "pred_logits": inter_class_logits[-1],
            "pred_boxes": inter_pred_bboxes[-1],
        }
        if inter_pred_severity[-1] is not None:
            output["pred_severity"] = inter_pred_severity[-1]
        if self.intermediate_supervision and len(inter_class_logits) > 1:
            aux_outputs = []
            for aux_logits, aux_boxes, aux_severity in zip(
                inter_class_logits[:-1], inter_pred_bboxes[:-1], inter_pred_severity[:-1]
            ):
                aux_item = {
                    "pred_logits": aux_logits,
                    "pred_boxes": aux_boxes,
                }
                if aux_severity is not None:
                    aux_item["pred_severity"] = aux_severity
                aux_outputs.append(aux_item)
            output["aux_outputs"] = aux_outputs

        losses = self.criterion(output, targets)

        for k in losses.keys():
            if k in self.weight_dict:
               losses[k] *= self.weight_dict[k]

        if self.attn_telemetry and attn_metrics:
            losses.update({k: v.detach() for k, v in attn_metrics.items()})

        return losses

    def make_targets(self, gt_boxes, whwh, labels, extras=None):
        """Universal targets builder for image detection."""
        targets = []
        for idx, (boxes_img, frame_size, label) in enumerate(zip(gt_boxes, whwh, labels)):
            target = {}
            boxes = torch.as_tensor(boxes_img, dtype=torch.float32, device=self.device)
            target["boxes_xyxy"] = boxes

            if label is None or len(label) == 0:
                class_ids = torch.zeros(len(boxes), dtype=torch.int64, device=self.device)
            else:
                label = torch.as_tensor(label, device=self.device)
                if label.ndim == 1:
                    class_ids = label.long()
                else:
                    class_ids = torch.argmax(label, dim=1).long()

            target["labels"] = class_ids
            target["image_size_xyxy"] = frame_size.to(self.device)
            target["image_size_xyxy_tgt"] = frame_size.unsqueeze(0).repeat(len(boxes), 1).to(self.device)

            severity_values = None
            if isinstance(extras, list):
                if idx < len(extras) and isinstance(extras[idx], dict):
                    if "severity" in extras[idx]:
                        severity_values = extras[idx]["severity"]
            elif isinstance(extras, dict) and "severity" in extras:
                severity_raw = extras["severity"]
                if isinstance(severity_raw, (list, tuple)):
                    if idx < len(severity_raw):
                        severity_values = severity_raw[idx]
                else:
                    severity_values = severity_raw

            if severity_values is not None:
                severity_tensor = torch.as_tensor(severity_values, dtype=torch.float32, device=self.device).reshape(-1)
                if severity_tensor.numel() == 1 and len(boxes) > 1:
                    severity_tensor = severity_tensor.repeat(len(boxes))
                if severity_tensor.numel() == len(boxes):
                    target["severity"] = severity_tensor

            if not self.use_pretrained_action:
                if label is not None and label.ndim == 2:
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
