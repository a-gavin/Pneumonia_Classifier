#! /usr/bin/env python3

import torch.nn as nn

from .resnet import ResNetResidualBlock, conv3x3, activation_func, conv_bn


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class TestBlock(ResNetResidualBlock):
    """
    Change ResNetBasicBlock to only have one conv3x3 and add a Dropout2d layer
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.blocks = nn.Sequential(
            conv3x3(in_channels, out_channels, bias=False, stride=self.downsampling),
            nn.Dropout2d(0.2),
            activation_func(self.activation),
        )
