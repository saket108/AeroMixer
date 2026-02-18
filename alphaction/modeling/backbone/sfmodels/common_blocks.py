# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Common building blocks for image + text multimodal models.
Image-only version - no video concepts.
"""

import torch.nn as nn


class Conv2dBN(nn.Module):
    """
    2D Convolution with BatchNorm for image processing.
    """
    def __init__(self, cfg, dim_in, dim_out, kernels, stride, padding, dilation=1, init_weight=None):
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernels, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        
        if hasattr(cfg.MODEL.BACKBONE, 'FROZEN_BN') and cfg.MODEL.BACKBONE.FROZEN_BN:
            from alphaction.layers import FrozenBatchNorm2d
            self.bn = FrozenBatchNorm2d(dim_out, eps=cfg.MODEL.BACKBONE.BN_EPSILON)
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)
        else:
            self.bn = nn.BatchNorm2d(dim_out, eps=cfg.MODEL.BACKBONE.BN_EPSILON, momentum=cfg.MODEL.BACKBONE.BN_MOMENTUM)
            if init_weight is not None:
                nn.init.constant_(self.bn.weight, init_weight)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for 2D image processing.
    """
    def __init__(self, cfg, dim_in, dim_out, dim_inner, stride, dilation=1):
        super(Bottleneck, self).__init__()
        # 1x1 convolution
        self.conv1 = Conv2dBN(cfg, dim_in, dim_inner, 1, 1, 0)
        # 3x3 convolution
        self.conv2 = Conv2dBN(cfg, dim_inner, dim_inner, 3, stride, dilation, dilation=dilation)
        # 1x1 convolution with BN_INIT_GAMMA
        self.conv3 = Conv2dBN(cfg, dim_inner, dim_out, 1, 1, 0, 
                               init_weight=getattr(cfg.MODEL.BACKBONE, 'BN_INIT_GAMMA', 1.0))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out


class ResBlock(nn.Module):
    """
    Residual block for 2D image processing.
    """
    def __init__(self, cfg, dim_in, dim_out, dim_inner, stride, dilation=1, need_shortcut=False):
        super(ResBlock, self).__init__()

        self.bottleneck = Bottleneck(cfg, dim_in, dim_out, dim_inner, stride, dilation)
        
        if not need_shortcut:
            self.shortcut = None
        else:
            self.shortcut = Conv2dBN(cfg, dim_in, dim_out, 1, stride, 0)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.bottleneck(x)
        
        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        
        out = residual + shortcut
        return self.relu(out)


class ResNLBlock(nn.Module):
    """
    Residual block with Non-local for 2D image processing.
    """
    def __init__(self, cfg, dim_in, dim_out, stride, num_blocks, dim_inner, 
                 nonlocal_inds=[], nonlocal_group=1, nonlocal_pool=[], dilation=1):
        super(ResNLBlock, self).__init__()
        
        self.blocks = []
        
        for idx in range(num_blocks):
            block_name = "res_{}".format(idx)
            block_stride = stride if idx == 0 else 1
            block_dilation = dilation if idx > 0 else 1
            
            need_shortcut = (dim_in != dim_out) or (block_stride != 1)
            
            res_module = ResBlock(cfg, dim_in, dim_out, dim_inner, block_stride, 
                                 block_dilation, need_shortcut)
            self.add_module(block_name, res_module)
            self.blocks.append(block_name)
            dim_in = dim_out
            
            # Add non-local block if specified
            if idx in nonlocal_inds:
                from .nonlocal_helper import Nonlocal
                nl_block_name = "nonlocal_{}".format(idx)
                nl_module = Nonlocal(dim_out, dim_out // 2, nonlocal_pool if nonlocal_pool else None)
                self.add_module(nl_block_name, nl_module)
                self.blocks.append(nl_block_name)

    def forward(self, x):
        for layer_name in self.blocks:
            x = getattr(self, layer_name)(x)
        return x


# Backward compatibility - keep old names as aliases
Conv3dBN = Conv2dBN  # Alias for backward compatibility
