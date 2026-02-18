#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Stem helper for image + text multimodal models.
Contains 2D (image) Stem components ONLY - no video concepts.
"""

import torch.nn as nn


def get_stem_func(name):
    """
    Retrieves the stem module by name.
    """
    trans_funcs = {"image_stem": ImageStem, "resnet_stem": ResNetBasicStem}
    assert (
        name in trans_funcs.keys()
    ), "Stem function '{}' not supported".format(name)
    return trans_funcs[name]


class ImageStem(nn.Module):
    """
    Image 2D stem module for image-only processing.
    Performs spatial Convolution, BN, and Relu following by a spatial pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (int): the kernel size of the convolution in the stem layer.
            stride (int): the stride size of the convolution in the stem layer.
            padding (int): the padding size of the convolution in the stem layer.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm2d.
        """
        super(ImageStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x


class ResNetBasicStem(nn.Module):
    """
    ResNet 2D stem module for image processing.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x


class ImageModelStem(nn.Module):
    """
    Image 2D stem module for single image processing.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm2d,
        stem_func_name="image_stem",
    ):
        super(ImageModelStem, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(norm_module, stem_func_name)

    def _construct_stem(self, norm_module, stem_func_name):
        trans_func = get_stem_func(stem_func_name)
        stem = trans_func(
            self.dim_in,
            self.dim_out,
            self.kernel,
            self.stride,
            self.padding,
            self.inplace_relu,
            self.eps,
            self.bn_mmt,
            norm_module,
        )
        self.stem = stem

    def forward(self, x):
        return self.stem(x)


class PatchEmbed(nn.Module):
    """
    Patch Embedding for image tokens.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=16,
        stride=4,
        padding=7,
        conv_2d=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, keep_spatial=False):
        x = self.proj(x)
        if keep_spatial:
            return x, x.shape
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape
