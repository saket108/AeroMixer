"""Image ResNet backbones for image+text multimodal detection."""

import torch
import torch.nn as nn
from torchvision import models


def _infer_depth(cfg):
    conv_body = str(getattr(cfg.MODEL.BACKBONE, "CONV_BODY", "")).lower()
    if "152" in conv_body:
        return 152
    if "101" in conv_body:
        return 101
    if "50" in conv_body:
        return 50
    return int(getattr(cfg.RESNET, "DEPTH", 50))


def _build_torchvision_resnet(depth):
    if depth >= 152:
        return models.resnet152(weights=None)
    if depth >= 101:
        return models.resnet101(weights=None)
    return models.resnet50(weights=None)


class ImageResNet(nn.Module):
    """Torchvision ResNet wrapper returning patch + pooled features."""

    def __init__(self, cfg):
        super(ImageResNet, self).__init__()

        depth = _infer_depth(cfg)
        net = _build_torchvision_resnet(depth)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool

        self.num_pathways = 1
        self.dim_out = int(net.fc.in_features)
        self.dim_embed = self.dim_out

    def _prepare_input(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("ImageResNet received an empty input list.")
            x = x[0]

        if x.ndim == 5:
            # Input comes as B, C, T, H, W in legacy pathway code. Collapse T for image mode.
            x = x.mean(dim=2)

        if x.ndim != 4:
            raise ValueError(f"ImageResNet expects 4D input after preprocessing, got shape={tuple(x.shape)}")
        return x

    def forward(self, x):
        x = self._prepare_input(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        patch_feat = self.layer4(x)

        cls_feat = self.avgpool(patch_feat)
        cls_feat = torch.flatten(cls_feat, 1)

        return patch_feat, cls_feat


class ImageResNetLite(nn.Module):
    """Lightweight 2D CNN backbone returning patch + pooled features."""

    def __init__(self, cfg):
        super(ImageResNetLite, self).__init__()

        self.num_pathways = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.block2 = self._make_block(64, 128, stride=2)
        self.block3 = self._make_block(128, 256, stride=2)
        self.block4 = self._make_block(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dim_out = 512
        self.dim_embed = 512

    def _make_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _prepare_input(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("ImageResNetLite received an empty input list.")
            x = x[0]

        if x.ndim == 5:
            x = x.mean(dim=2)

        if x.ndim != 4:
            raise ValueError(f"ImageResNetLite expects 4D input after preprocessing, got shape={tuple(x.shape)}")
        return x

    def forward(self, x):
        x = self._prepare_input(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block2(x)
        x = self.block3(x)
        patch_feat = self.block4(x)

        cls_feat = self.avgpool(patch_feat)
        cls_feat = torch.flatten(cls_feat, 1)
        return patch_feat, cls_feat
