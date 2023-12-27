# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.registry import MODELS
from mmrazor.models.architectures.dynamic_ops.bricks import DynamicSequential
from mmrazor.models.architectures.utils.mutable_register import (
    mutate_conv_module)
from mmrazor.models.architectures.utils import set_dropout
from mmrazor.models.mutables import (OneShotMutableChannel,
                                     OneShotMutableValue)
from mmrazor.models.utils import make_divisible

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')


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
    """InvertedResidual block for MobileNetV2.

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
        self.expand_ratio = expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        # hidden_dim = int(round(in_channels * expand_ratio))
        hidden_dim = make_divisible(int(in_channels * expand_ratio), 8)

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


@MODELS.register_module()
class BigNASMobileNetV2(BaseBackbone):
    """MobileNetV2 backbone.

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
    ref_arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                         [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                         [6, 320, 1, 1]]

    def __init__(self,
                 arch_setting: Dict[str, List],
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 conv_cfg: Dict = dict(type='BigNasConv2d'),
                 norm_cfg: Dict = dict(type='DynamicBatchNorm2d'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 fine_grained_mode=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(BigNASMobileNetV2, self).__init__(init_cfg)
        self.arch_setting = arch_setting
        for key, value in arch_setting.items():
            assert key in ['num_blocks_add', 'expand_ratio_add', 'out_channels_mult']
            assert len(self.ref_arch_settings) == len(value)
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
        self.fine_grained_mode = fine_grained_mode

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

        # adapt mutable settings
        self.num_blocks_list = parse_values(self.arch_setting['num_blocks_add'])
        self.expand_ratio_list = parse_values(self.arch_setting['expand_ratio_add'])
        self.num_channels_list = parse_values(self.arch_setting['out_channels_mult'])

        for i, layer_cfg in enumerate(self.ref_arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = channel * widen_factor

            self.num_blocks_list[i] = [
                int(a + num_blocks) for a in self.num_blocks_list[i]]
            self.expand_ratio_list[i] = [
                int(a + expand_ratio) for a in self.expand_ratio_list[i]]
            self.num_channels_list[i] = [
                make_divisible(m * out_channels, 8) for m in self.num_channels_list[i]]

            inverted_res_layer = self.make_layer(
                out_channels=max(self.num_channels_list[i]),
                num_blocks=max(self.num_blocks_list[i]),
                stride=stride,
                expand_ratio=max(self.expand_ratio_list[i]))
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(1280 * widen_factor)
        else:
            self.out_channel = 1280
        # import pdb; pdb.set_trace()
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
        """Mutate the BigNAS-style ResNet."""

        mid_mutable = OneShotMutableChannel(
            alias='backbone.conv1_channels',
            num_channels=self.conv1.out_channels,
            candidate_choices=[self.conv1.out_channels])

        # mutate the built mobilenet layers
        for i, layer_name in enumerate(self.layers[:-1]):
            layer = getattr(self, layer_name)
            num_blocks = self.num_blocks_list[i]
            expand_ratios = self.expand_ratio_list[i]
            out_channels = self.num_channels_list[i]

            prefix = 'backbone.layers.' + str(i + 1) + '.'

            mutable_out_channels = OneShotMutableChannel(
                alias=prefix + 'out_channels',
                candidate_choices=out_channels,
                num_channels=max(out_channels))

            if not self.fine_grained_mode:
                mutable_expand_ratio = OneShotMutableValue(
                    alias=prefix + 'expand_ratio', value_list=expand_ratios)

            mutable_depth = OneShotMutableValue(
                alias=prefix + 'depth', value_list=num_blocks)
            layer.register_mutable_attr('depth', mutable_depth)

            for k in range(max(num_blocks)):

                if self.fine_grained_mode:
                    mutable_expand_ratio = OneShotMutableValue(
                        alias=prefix + str(k) + '.expand_ratio',
                        value_list=expand_ratios)
                mutate_block(layer[k], mid_mutable,
                             mutable_out_channels,
                             mutable_expand_ratio)
                mid_mutable = mutable_out_channels

        self.last_mutable_channels = OneShotMutableChannel(
            alias='backbone.conv2_channels',
            num_channels=self.conv2.out_channels,
            candidate_choices=[self.conv2.out_channels])
        mutate_conv_module(self.conv2, mid_mutable, self.last_mutable_channels)

    def set_dropout(self, dropout_stages, drop_path_rate):
        pass
        # layers = [getattr(self, layer_name) for i, layer_name in enumerate(self.res_layers)]
        # set_dropout(layers=layers, module=self.block, 
        #             dropout_stages=dropout_stages,
        #             drop_path_rate=drop_path_rate)

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

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

        return DynamicSequential(*layers)

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
        super(BigNASMobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def mutate_block(block_layer: InvertedResidual,
                 mutable_in_channels,
                 mutable_out_channels,
                 mutable_expand_ratio):
    """Mutate InvertedResidual layers."""

    # TODO(shiguang): align with hidden_dim = int(round(in_channels * expand_ratio))
    derived_mid_channels = \
        (mutable_in_channels * mutable_expand_ratio).derive_divide_mutable(1, 8)

    mutate_conv_module(block_layer.conv[-2], derived_mid_channels, derived_mid_channels)
    mutate_conv_module(block_layer.conv[-1], derived_mid_channels, mutable_out_channels)
    # conv1
    if block_layer.expand_ratio != 1:
        mutate_conv_module(block_layer.conv[0], mutable_in_channels, derived_mid_channels)
