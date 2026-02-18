#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Non-local helper for image + text multimodal models.
Contains ONLY 2D (image) Nonlocal blocks - no video concepts.
"""

import torch
import torch.nn as nn


class Nonlocal(nn.Module):
    """
    2D Non-local Neural Networks for image processing.
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions.
    
    This is the 2D version adapted for image + text multimodal models.
    """

    def __init__(
        self,
        dim,
        dim_inner,
        pool_size=None,
        instantiation="softmax",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            dim_inner (int): number of dimension inside of the Non-local block.
            pool_size (list): the kernel size of spatial pooling.
                By default pool_size is None, then there would be no pooling used.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            zero_init_final_conv (bool): If true, zero initializing the final
                convolution of the Non-local block.
            zero_init_final_norm (bool):
                If true, zero initializing the final batch norm of the Non-local
                block.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm2d.
        """
        super(Nonlocal, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = (
            False
            if pool_size is None
            else any((size > 1 for size in pool_size))
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(
            zero_init_final_conv, zero_init_final_norm, norm_module
        )

    def _construct_nonlocal(self, zero_init_final_conv, zero_init_final_norm, norm_module):
        # Three convolution heads: theta, phi, and g (2D convolutions)
        self.conv_theta = nn.Conv2d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = nn.Conv2d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = nn.Conv2d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        # Final convolution output (2D)
        self.conv_out = nn.Conv2d(
            self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        )
        # Zero initializing the final convolution output.
        self.conv_out.zero_init = zero_init_final_conv

        # 2D BatchNorm
        self.bn = norm_module(
            num_features=self.dim,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
        )
        # Zero initializing the final bn.
        self.bn.transform_final_bn = zero_init_final_norm

        # Optional spatial pooling (2D)
        if self.use_pool:
            self.pool = nn.MaxPool2d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
            )

    def forward(self, x):
        x_identity = x
        N, C, H, W = x.size()

        theta = self.conv_theta(x)

        # Perform spatial pooling to reduce the computation.
        if self.use_pool:
            x = self.pool(x)

        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # (N, C, HxW) * (N, C, HxW) => (N, HxW, HxW)
        theta_phi = torch.einsum("nct,ncp->ntp", (theta, phi))
        
        # Normalize the affinity tensor
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self.dim_inner**-0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

        # (N, HxW, HxW) * (N, C, HxW) => (N, C, HxW)
        theta_phi_g = torch.einsum("ntg,ncg->nct", (theta_phi, g))

        # (N, C, HxW) => (N, C, H, W)
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, H, W)

        p = self.conv_out(theta_phi_g)
        p = self.bn(p)
        return x_identity + p
