#! /usr/bin/env python3
"""
Base implementation of ResNet can be found at:
    https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
"""
from functools import partial

import torch.nn as nn


class Conv2dAuto(nn.Conv2d):
    """
    Allow 2D convolutional layers to have 'auto' padding
    """

    def __init__(self, *args, **kwargs):
        """
        Creates a pytorch Conv2d layer allows for dynamic padding based on the kernel size
        :param args: same as torch.nn.Conv2d
        :param kwargs: same as torch.nn.Conv2d
        """
        super().__init__(*args, **kwargs)

        # dynamic add padding based on the kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


def activation_func(activation):
    """
    Return a torch.nn activation function given the its name as a string
    :param activation: the name of the activation as a string
    :return: the torch.nn Module of the activation function
    """
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    """
    Creates a block of convolutional layers to reduce in_channels to out_channels and sum up the original input
    """

    def __init__(self, in_channels, out_channels, activation='relu'):
        """
        Initializes the ResidualBlock with the number of in_channels, out_channels, and the activation function
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param activation: the activation function to be applied in the forward method
        """
        super().__init__()

        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)

        x = self.blocks(x)
        x += residual
        x = self.activate(x)

        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResNetResidualBlock(ResidualBlock):
    """
    Defines a ResidualBlock for ResNet which allows for an increase in out_channels through an expansion parameter
    """

    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        """

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param expansion: value that out_channels should be scaled by
        :param downsampling: the size of the stride used in the Conv2d layer
        :param conv: the torch.nn Module for the type of convolution
        :param args: same as ResidualBlock
        :param kwargs: same as ResidualBlock
        """
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv

        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        ) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    """
    Convenient for the stacking of a convolutional layer and BatchNorm2d layer
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param conv: the torch.nn Module for the type of convolutional layer to be used
    :param args: same as the convolutional layer Module being passed
    :param kwargs: same as the convolutional layer Module being passed
    :return: a Sequential Module containing the convolutional layer and BatchNorm2d layer
    """
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv, batchnorm, activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """
        Initializes the basic block used in ResNet
        :param in_channels: number of input channels
        :param out_channels: number of ouput channels
        :param args: same as ResNetResidualBlock
        :param kwargs: same as ResNetResidualBlock
        """
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    """
    Increases the network depth while keeping the parameters size as low as possible
    the authors defined a BottleNeck block as:
        “The three layers are 1x1, 3x3, and 1x1 convolutions, where the 1×1 layers are responsible
        for reducing and then increasing (restoring) dimensions, leaving the 3×3 layer a bottleneck with smaller
        input/output dimensions.”
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """
        Initializes the bottleneck used in ResNet
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param args: same as ResNetResidualBlock
        :param kwargs: same as ResNetResidualBlock
        """
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by 'n' blocks stacked one after the other
    """

    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        """
        Initializes the layer of stacked blocks
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param block: the type of block to be used
        :param n: the total number of blocks to be added in the layer
        :param args: same as block
        :param kwargs: same as block
        """
        super().__init__()

        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels, downsampling=1,
                    *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)

        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        """
        Initializes the ResNetEncoder
        :param in_channels: number of input channels
        :param blocks_sizes: the various block sizes used to create all of the blocks
        :param deepths: number of blocks in the first ResNetLayer
        :param activation: the activation function to be applied in each ResNetLayer
        :param block: the type of block to be used
        :param args: same as ResNetLayer
        :param kwargs: same as ResNetLayer
        """
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)

        for block in self.blocks:
            x = block(x)

        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the correct class by
    using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        """
        Initializes the ResNetDecoder
        :param in_features: number of input features
        :param n_classes: number of classes
        """
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x


class ResNet(nn.Module):
    """
    The final model for ResNet
    """

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        """
        Initializes ResNet
        :param in_channels: number of input channels
        :param n_classes: number of classes
        :param args: same as ResNetEncoder
        :param kwargs: same as ResNetEncoder
        """
        super().__init__()

        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def resnet18(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)


def resnet34(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)


def resnet50(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)


def resnet101(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 23, 3], *args, **kwargs)


def resnet152(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)
