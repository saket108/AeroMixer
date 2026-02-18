# Copyright (c) Facebook, Inc. and its affiliates.

"""
Common building blocks for image + text multimodal models.
Contains ONLY image/multimodal components - no video concepts.
"""

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ImageTextFusion(nn.Module):
    """
    Image + Text Multimodal Fusion Module.
    Fuses visual features with text features for image + text multimodal models.
    
    Supports multiple fusion strategies:
    - "concat" : Concatenate image and text features
    - "add" : Element-wise addition (requires same dimension)
    - "bilinear" : Bilinear fusion
    - "mlp" : MLP-based fusion
    - "cross_attention" : Cross-attention between image and text
    """
    
    def __init__(
        self,
        image_dim,
        text_dim,
        fusion_dim=None,
        mode="concat",
        hidden_dim=None,
        drop_rate=0.0,
        num_heads=8,
    ):
        """
        Args:
            image_dim (int): Dimension of image features
            text_dim (int): Dimension of text features
            fusion_dim (int): Output dimension after fusion (if None, uses image_dim)
            mode (str): Fusion mode - "concat", "add", "bilinear", "mlp", "cross_attention"
            hidden_dim (int): Hidden dimension for MLP fusion
            drop_rate (float): Dropout rate
            num_heads (int): Number of attention heads for cross-attention
        """
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim or image_dim
        self.mode = mode
        
        if mode == "concat":
            # Concatenate and project
            self.fuse_fn = nn.Sequential(
                nn.Linear(image_dim + text_dim, self.fusion_dim),
                nn.GELU(),
                nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity(),
            )
        elif mode == "add":
            # Project text to image dimension and add
            if image_dim != text_dim:
                self.text_proj = nn.Linear(text_dim, image_dim)
            else:
                self.text_proj = nn.Identity()
            # Optional projection after addition
            if self.fusion_dim != image_dim:
                self.output_proj = nn.Linear(image_dim, self.fusion_dim)
            else:
                self.output_proj = nn.Identity()
        elif mode == "bilinear":
            # Bilinear fusion
            self.bilinear = nn.Bilinear(image_dim, text_dim, self.fusion_dim)
        elif mode == "mlp":
            # MLP fusion
            hidden_dim = hidden_dim or (image_dim + text_dim) // 2
            self.fuse_fn = nn.Sequential(
                nn.Linear(image_dim + text_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity(),
                nn.Linear(hidden_dim, self.fusion_dim),
            )
        elif mode == "cross_attention":
            # Cross-attention fusion
            self.image_proj = nn.Linear(image_dim, image_dim)
            self.text_proj = nn.Linear(text_dim, image_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=image_dim, 
                num_heads=num_heads, 
                dropout=drop_rate,
                batch_first=True
            )
            self.output_proj = nn.Linear(image_dim, self.fusion_dim)
        else:
            raise NotImplementedError(f"Fusion mode '{mode}' not supported")
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: Tensor of shape (B, C_image) or (B, N, C_image)
            text_features: Tensor of shape (B, C_text) or (B, M, C_text)
            
        Returns:
            fused_features: Tensor of shape (B, fusion_dim) or (B, N, fusion_dim)
        """
        if self.mode == "concat":
            # Handle both (B, C) and (B, N, C) cases
            if image_features.dim() == 2:
                return self.fuse_fn(torch.cat([image_features, text_features], dim=-1))
            else:
                # For sequence features: (B, N, C_image) and (B, M, C_text)
                # Expand text to match image sequence length
                B, N, C_img = image_features.shape
                _, M, C_txt = text_features.shape
                # Take first text token or average
                text_repr = text_features[:, 0, :] if text_features.dim() == 3 else text_features
                text_repr = text_repr.unsqueeze(1).expand(-1, N, -1)
                return self.fuse_fn(torch.cat([image_features, text_repr], dim=-1))
        
        elif self.mode == "add":
            text_proj = self.text_proj(text_features)
            if image_features.dim() == 2:
                fused = image_features + text_proj
            else:
                # For sequence features, add text repr to each position
                text_repr = text_features[:, 0, :] if text_features.dim() == 3 else text_features
                fused = image_features + text_repr.unsqueeze(1)
            return self.output_proj(fused)
        
        elif self.mode == "bilinear":
            if image_features.dim() == 3:
                # For sequence features, apply bilinear to each position
                text_repr = text_features[:, 0, :] if text_features.dim() == 3 else text_features
                text_repr = text_repr.unsqueeze(1).expand(-1, image_features.size(1), -1)
                return self.bilinear(image_features, text_repr)
            return self.bilinear(image_features, text_features)
        
        elif self.mode == "mlp":
            if image_features.dim() == 2:
                return self.fuse_fn(torch.cat([image_features, text_features], dim=-1))
            else:
                text_repr = text_features[:, 0, :] if text_features.dim() == 3 else text_features
                text_repr = text_repr.unsqueeze(1).expand(-1, image_features.size(1), -1)
                return self.fuse_fn(torch.cat([image_features, text_repr], dim=-1))
        
        elif self.mode == "cross_attention":
            # Cross-attention between image and text
            B, N, C_img = image_features.shape
            # Project features
            image_q = self.image_proj(image_features)
            text_k = self.text_proj(text_features)
            text_v = self.text_proj(text_features)
            
            # For text, use first token or pooled
            if text_features.dim() == 3:
                text_k = text_k[:, 0:1, :]
                text_v = text_v[:, 0:1, :]
            
            # Cross attention: image queries attend to text keys/values
            attn_output, _ = self.cross_attn(image_q, text_k, text_v)
            fused = image_q + attn_output  # Residual connection
            return self.output_proj(fused)
        
        else:
            raise NotImplementedError


class ImageStem2D(nn.Module):
    """
    Image Stem Module for 2D image processing.
    Converts input images to feature maps using 2D convolutions.
    """
    
    def __init__(
        self,
        dim_in=3,
        dim_out=64,
        kernel=7,
        stride=2,
        padding=3,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim_in (int): Input channel dimension (default: 3 for RGB)
            dim_out (int): Output channel dimension
            kernel (int): Convolution kernel size
            stride (int): Convolution stride
            padding (int): Convolution padding
            norm_module (nn.Module): Normalization module
        """
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = norm_module(num_features=dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, dim_out, H', W')
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
