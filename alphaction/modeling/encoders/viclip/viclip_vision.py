#!/usr/bin/env python
"""
ViCLIP Vision Encoder for Image Multimodal Models.
Converted from 3D (video) to 2D (image) version.
"""
import os
import logging
from collections import OrderedDict

import torch
from torch import nn
from timm.layers import DropPath
from timm.models import register_model

import torch.utils.checkpoint as checkpoint

logger = logging.getLogger(__name__)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, dropout=0.):
        super().__init__()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("drop2", nn.Dropout(dropout)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.drop_path1(self.attention(self.ln_1(x)))
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0., checkpoint_num=0, dropout=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, drop_path=dpr[idx], dropout=dropout))
        self.checkpoint_num = checkpoint_num

    def forward(self, x):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for Image Multimodal.
    Converted from 3D (video) to 2D (image) version.
    """
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim=None, 
        kernel_size=1, num_frames=1, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=False,  # Default to False for image mode
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # Converted from Conv3d to Conv2d for image
        self.conv1 = nn.Conv2d(
            3, width, 
            (patch_size, patch_size), 
            (patch_size, patch_size), 
            (0, 0), bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.grid_size = [input_resolution // patch_size, input_resolution // patch_size]
        self.ln_pre = nn.LayerNorm(width)
        
        # For image mode, temporal embeddings are not used (set to None)
        self.temporal_positional_embedding = None
        self.num_frames = 1  # Force to 1 for image mode
        
        self.transformer = Transformer(
            width, layers, heads, drop_path=drop_path, checkpoint_num=checkpoint_num,
            dropout=dropout)

        self.ln_post = nn.LayerNorm(width)
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(width, output_dim))
        else:
            self.proj = None
        
        self.dropout = nn.Dropout(dropout)

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding'}
    
    def mask_tokens(self, inputs, masking_prob=0.0):
        B, L, _ = inputs.shape
        Lm = int(masking_prob * L)
        masked_indices = torch.zeros(B, L)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :Lm]
        batch_indices = (
            torch.arange(masked_indices.shape[0]).unsqueeze(-1).expand_as(indices)
        )
        masked_indices[batch_indices, indices] = 1
        masked_indices = masked_indices.bool()
        return inputs[~masked_indices].reshape(B, -1, inputs.shape[-1])

    def forward(self, x, interpolator=None, masking_prob=0.0, return_patch=False):
        """
        Forward pass for image input.
        Input x: (B, C, H, W) - 2D image
        """
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # For image: B, C, H, W
        B, C, H, W = x.shape
        
        # Reshape: (B, C, H, W) -> (B, H*W, C)
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        # Add class token
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device), 
            x
        ], dim=1)  # shape = [B, grid ** 2 + 1, width]
        
        # Add positional embedding
        pos_embed = self.positional_embedding.to(x.dtype)
        if (interpolator is not None) and (pos_embed.shape[0] != H * W + 1):
            grid_size = [H, W]
            pos_embed = interpolator(pos_embed.unsqueeze(0), self.grid_size, grid_size, 1)
        x = x + pos_embed

        # No temporal processing for image mode (unlike video)
        
        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  #BND -> NBD
        x = self.transformer(x)

        if return_patch:
            x_p = x[1:]

        x = self.ln_post(x)

        if self.proj is not None:
            x = self.dropout(x[0]) @ self.proj
        else:
            x = x.permute(1, 0, 2)  #NBD -> BND
        
        if return_patch:
            grid_size = [H, W]
            return x, x_p, grid_size

        return x


def load_state_dict(model, state_dict, input_resolution=224, patch_size=16, center=True):
    """
    Load state dict for image model (no temporal inflation needed).
    """
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                logger.info(f'Ignore: {k}')
                continue
            # For 2D model, we need to handle dimension mismatches
            logger.info(f'Skip: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')

    pos_embed_checkpoint = state_dict.get('positional_embedding', None)
    if pos_embed_checkpoint is not None:
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = (input_resolution // patch_size) ** 2
        orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            logger.info(f'Pos_emb from {orig_size} to {new_size}')
            extra_tokens = pos_embed_checkpoint[:1]
            pos_tokens = pos_embed_checkpoint[1:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            state_dict['positional_embedding'] = new_pos_embed
    
    message = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Load pretrained weights: {message}")


@register_model
def clip_joint_l14_image(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=1, drop_path=0., checkpoint_num=0,
    dropout=0.,
):
    """
    Image-only version of ViT-L/14 for multimodal image + text.
    """
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames,  # num_frames=1 for image
        drop_path=drop_path, checkpoint_num=checkpoint_num,
        dropout=dropout, temp_embed=False,  # No temporal embedding for image
    )
    
    if pretrained:
        logger.info('Image pretrained weights not available yet')
    
    return model.eval()


@register_model
def clip_joint_b16_image(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=1, drop_path=0., dropout=0.,
):
    """
    Image-only version of ViT-B/16 for multimodal image + text.
    """
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=16,
        width=768, layers=12, heads=12, output_dim=512,
        kernel_size=kernel_size, num_frames=num_frames,
        drop_path=drop_path, dropout=dropout, temp_embed=False,
    )
    
    if pretrained:
        logger.info('Image pretrained weights not available yet')
    
    return model.eval()


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Test image model (no temporal dimension)
    model = clip_joint_l14_image(pretrained=False)
    logger.info(model)

    # Input: (B, C, H, W) for image instead of (B, C, T, H, W) for video
    flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224))
    s = time.time()
    logger.info(flop_count_table(flops, max_depth=1))
    logger.info(time.time()-s)
    logger.info(model(torch.rand(1, 3, 224, 224)).shape)
