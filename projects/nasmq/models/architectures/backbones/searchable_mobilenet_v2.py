# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models import OneShotMutableChannel, OneShotMutableValue, mutate_conv_module
from mmrazor.registry import MODELS

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
    from mmcls.models.utils import make_divisible
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')
    make_divisible = get_placeholder('mmcls')


def parse_values(candidate_lists: List):
    """Parse a list with format `(min_range, max_range, step)`.

    NOTE: this method is required when customizing search space in configs.
    """

    def _range_to_list(input_range: List) -> List:
        # assert len(input_range) == 3, (
        #     'The format should be `(min_range, max_range, step)` with dim=3, '
        #     f'but got dim={len(input_range)}.')
        if input_range[-1] == 'categorical':
            return list(input_range[:-1])
        start, end, step = input_range
        return list(np.arange(start, end, step))

    return [_range_to_list(i) for i in candidate_lists]


class InvertedResidual(BaseModule):
    """InvertedResidual block for QNASMobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


def mutate_block(block_layer: InvertedResidual,
                 mutable_in_channels,
                 mutable_out_channels,
                 expand_ratio,
                 mid_channels_mult,
                 prefix=''):
    """Mutate DepthWise layers."""
    convs = block_layer.conv
    if len(convs) == 2:
        mutate_conv_module(convs[0], mutable_in_channels, mutable_in_channels)
        mutate_conv_module(convs[1], mutable_in_channels, mutable_out_channels)
    elif len(convs) ==3:
        in_channels = max(mutable_in_channels.choices)
        hidden_channels = list(set(make_divisible(m * in_channels * expand_ratio, 8) for m in mid_channels_mult))
        mutable_mid_channels =  OneShotMutableChannel(alias=prefix + 'mid.out_channels',
            candidate_choices=hidden_channels, num_channels=max(hidden_channels))        
        mutate_conv_module(convs[0], mutable_in_channels, mutable_mid_channels)
        mutate_conv_module(convs[1], mutable_mid_channels, mutable_mid_channels)
        mutate_conv_module(convs[2], mutable_mid_channels, mutable_out_channels)
    else:
        raise RuntimeError(f'Unexpect block_layer length: {len(convs)}')


@MODELS.register_module()
class QNASMobileNetV2(BaseBackbone):
    """QNASMobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                     [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                     [6, 320, 1, 1]]

    def __init__(self,
                 arch_setting=None,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 conv_cfg=dict(type='BigNasConv2d'),
                 norm_cfg=dict(type='DynamicBatchNorm2d', requires_grad=True),                 
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(QNASMobileNetV2, self).__init__(init_cfg)
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.arch_setting = arch_setting
        # adapt mutable settings
        self.channels_mult = parse_values([self.arch_setting['channels_mult']])[0]
        self.out_channels_list = []

        self.in_channels = make_divisible(32 * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            self.out_channels_list.append(list(set(
                make_divisible(m * out_channels, 8) for m in self.channels_mult)))
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(1280 * widen_factor)
        else:
            self.out_channel = 1280

        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

        self.register_mutables()

    def register_mutables(self):
        """Mutate the BigNAS-style MobileNetV2."""
        conv1_channels = list(set(make_divisible(m * self.conv1.out_channels, 8) for m in self.channels_mult))
        conv1_mutable_channels = OneShotMutableChannel(
            alias='backbone.conv1_channels',
            num_channels=max(conv1_channels),
            candidate_choices=conv1_channels)

        mutate_conv_module(self.conv1, mutable_out_channels=conv1_mutable_channels)

        mid_mutable = conv1_mutable_channels
        # mutate the built MobileNetV2 layers
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            layer_name = f'layer{i + 1}'
            layer = getattr(self, layer_name)

            out_channels = self.out_channels_list[i]
            prefix = f'backbone.{layer_name}.'

            mutable_out_channels = OneShotMutableChannel(
                alias=prefix + 'out_channels',
                candidate_choices=out_channels,
                num_channels=max(out_channels))

            for k in range(num_blocks):
                block_prefix = prefix + str(k) + '.'
                mutate_block(layer[k], mid_mutable, mutable_out_channels, expand_ratio, self.channels_mult, block_prefix)
                mid_mutable = mutable_out_channels

        conv2_channels = list(set(make_divisible(m * self.out_channel, 8) for m in self.channels_mult))
        self.last_mutable_channels = OneShotMutableChannel(
            alias='backbone.conv2_channels',
            num_channels=max(conv2_channels),
            candidate_choices=conv2_channels)
        mutate_conv_module(self.conv2, mid_mutable, self.last_mutable_channels)

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for QNASMobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride,
                    expand_ratio=expand_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(QNASMobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
