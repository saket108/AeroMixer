#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
ResNet helper for image + text multimodal models.
Contains ONLY 2D (image) ResNet components - no video concepts.
"""

import torch.nn as nn

from .common import drop_path
from .nonlocal_helper import Nonlocal


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        "basic_transform": BasicTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class BasicTransform(nn.Module):
    """
    2D Basic transformation: 3x3 conv (spatial only).
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner=None,
        num_groups=1,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm2d,
        block_idx=0,
    ):
        super(BasicTransform, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride, dilation, norm_module)

    def _construct(self, dim_in, dim_out, stride, dilation, norm_module):
        # 3x3, BN, ReLU.
        self.a = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        # 1x3x3, BN.
        self.b = nn.Conv2d(
            dim_out,
            dim_out,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )

        self.b.final_conv = True

        self.b_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )

        self.b_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        return x


class BottleneckTransform(nn.Module):
    """
    2D Bottleneck transformation: 1x1, 3x3, 1x1 (spatial only).
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm2d,
        block_idx=0,
    ):
        super(BottleneckTransform, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # 1x1, BN, ReLU.
        self.a = nn.Conv2d(
            dim_in,
            dim_inner,
            kernel_size=1,
            stride=str1x1,
            padding=0,
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 3x3, BN, ReLU.
        self.b = nn.Conv2d(
            dim_inner,
            dim_inner,
            kernel_size=3,
            stride=str3x3,
            padding=dilation,
            groups=num_groups,
            bias=False,
            dilation=dilation,
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1, BN.
        self.c = nn.Conv2d(
            dim_inner,
            dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.c.final_conv = True

        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Module):
    """
    2D Residual block for image processing.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm2d,
        block_idx=0,
        drop_connect_rate=0.0,
    ):
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
        block_idx,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        self.relu = nn.ReLU(inplace=self._inplace_relu)

    def forward(self, x):
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = drop_path(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    """
    Stage of 2D ResNet for image processing.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        num_blocks,
        dim_inner,
        num_groups,
        nonlocal_inds,
        nonlocal_group,
        nonlocal_pool,
        dilation,
        instantiation="softmax",
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
        norm_module=nn.BatchNorm2d,
        drop_connect_rate=0.0,
    ):
        super(ResStage, self).__init__()
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self._drop_connect_rate = drop_connect_rate
        self.num_pathways = 1  # Single pathway for 2D
        
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        nonlocal_inds,
        nonlocal_pool,
        instantiation,
        dilation,
        norm_module,
    ):
        # For 2D, dim_in and dim_out are single values, not lists
        if not isinstance(dim_in, list):
            dim_in = [dim_in]
            dim_out = [dim_out]
            stride = [stride]
            dim_inner = [dim_inner]
            num_groups = [num_groups]
            nonlocal_inds = [nonlocal_inds]
            nonlocal_group = [self.nonlocal_group]
            nonlocal_pool = [nonlocal_pool]
            dilation = [dilation]

        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks):
                # Retrieve the transformation function.
                trans_func = get_trans_func(trans_func_name)
                # Construct the block.
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    stride[pathway] if i == 0 else 1,
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                    block_idx=i,
                    drop_connect_rate=self._drop_connect_rate,
                )
                self.add_module("res{}".format(i), res_block)
                if i in nonlocal_inds[pathway]:
                    nln = Nonlocal(
                        dim_out[pathway],
                        dim_out[pathway] // 2,
                        nonlocal_pool[pathway],
                        instantiation=instantiation,
                        norm_module=norm_module,
                    )
                    self.add_module("nonlocal{}".format(i), nln)

    def forward(self, x):
        # For 2D, input is a single tensor, not a list
        if not isinstance(x, list):
            x = [x]
        
        output = []
        for pathway in range(self.num_pathways):
            x_path = x[pathway]
            for i in range(self.num_blocks):
                m = getattr(self, "res{}".format(i))
                x_path = m(x_path)
                if hasattr(self, "nonlocal{}".format(i)):
                    nln = getattr(self, "nonlocal{}".format(i))
                    x_path = nln(x_path)
            output.append(x_path)

        return output[0] if self.num_pathways == 1 else output
