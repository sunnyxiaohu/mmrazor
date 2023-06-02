# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmengine.logging import MMLogger
from mmengine.model import Sequential, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_ops.bricks import DynamicSequential
from mmrazor.models.architectures.utils.mutable_register import (
    mutate_conv_module)
from mmrazor.models.architectures.utils import set_dropout    
from mmrazor.models.mutables import (MutableChannelContainer,
                                     OneShotMutableChannel,
                                     OneShotMutableChannelUnit,
                                     OneShotMutableValue)
from mmrazor.models.utils import make_divisible                                     
from mmrazor.registry import MODELS
from ..ops.resnet_series import BasicBlock, Bottleneck

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')

logger = MMLogger.get_current_instance()


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


def mutate_basicblock(block_layer: BasicBlock,
                        mutable_in_channels,
                        mutable_out_channels,
                        mutable_expand_ratio):
    """Mutate basicblock layers."""

    derived_mid_channels = \
        (mutable_expand_ratio * mutable_out_channels).derive_divide_mutable(1, 8)
    # conv1
    block_layer.conv1.register_mutable_attr('in_channels', mutable_in_channels)
    block_layer.conv1.register_mutable_attr('out_channels', derived_mid_channels)
    # norm1
    block_layer.norm1.register_mutable_attr('num_features', derived_mid_channels)

    # conv2
    block_layer.conv2.register_mutable_attr('in_channels', derived_mid_channels)
    block_layer.conv2.register_mutable_attr('out_channels', mutable_out_channels)
    # norm2
    block_layer.norm2.register_mutable_attr('num_features', mutable_out_channels)

    if block_layer.downsample:
        block_layer.downsample[-2].register_mutable_attr('in_channels', mutable_in_channels)
        block_layer.downsample[-2].register_mutable_attr('out_channels', mutable_out_channels)
        block_layer.downsample[-1].register_mutable_attr('num_features', mutable_out_channels)


def mutate_bottleneck(block_layer: Bottleneck,
                        mutable_in_channels,
                        mutable_out_channels,
                        mutable_expand_ratio):
    """Mutate bottleneck layers."""

    derived_mid_channels = \
        (mutable_expand_ratio * mutable_out_channels).derive_divide_mutable(1, 8)
    # conv1
    block_layer.conv1.register_mutable_attr('in_channels', mutable_in_channels)
    block_layer.conv1.register_mutable_attr('out_channels', derived_mid_channels)
    # norm1
    block_layer.norm1.register_mutable_attr('num_features', derived_mid_channels)

    # conv2
    block_layer.conv2.register_mutable_attr('in_channels', derived_mid_channels)
    block_layer.conv2.register_mutable_attr('out_channels', derived_mid_channels)
    # norm2
    block_layer.norm2.register_mutable_attr('num_features', derived_mid_channels)

    # conv3
    block_layer.conv3.register_mutable_attr('in_channels', derived_mid_channels)
    block_layer.conv3.register_mutable_attr('out_channels', mutable_out_channels)
    # norm3
    block_layer.norm3.register_mutable_attr('num_features', mutable_out_channels)

    if block_layer.downsample:
        block_layer.downsample[-2].register_mutable_attr('in_channels', mutable_in_channels)
        block_layer.downsample[-2].register_mutable_attr('out_channels', mutable_out_channels)
        block_layer.downsample[-1].register_mutable_attr('num_features', mutable_out_channels)


@MODELS.register_module()
class BigNASResNet(BaseBackbone):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        fine_grained_mode (bool): Whether to use fine-grained mode (search
            kernel size & expand ratio for each MB block in each layers).
            Defaults to False.
    Example:
        >>> from mmcls.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    ref_arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2), (64, 128, 256, 512)),
        50: (Bottleneck, (3, 4, 6, 3), (256, 512, 1024, 2048))
    }

    def __init__(self,
                 arch_setting: Dict[str, List],
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg: Dict = dict(type='BigNasConv2d'),
                 norm_cfg: Dict = dict(type='DynamicBatchNorm2d'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 fine_grained_mode=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(BigNASResNet, self).__init__(init_cfg)

        if depth not in self.ref_arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.arch_setting = arch_setting
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.fine_grained_mode = fine_grained_mode
        self.block, stage_blocks, out_channels = self.ref_arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.expansion = self.block.expansion

        # adapt mutable settings
        self.num_blocks_list = parse_values(self.arch_setting['num_blocks_add'])
        self.num_blocks_list = [[
            int(a + stage_blocks[idx]) for a in a_factor
        ] for idx, a_factor in enumerate(self.num_blocks_list)]
        self.expand_ratio_list = \
            parse_values(self.arch_setting['expand_ratio_mult'])
        self.expand_ratio_list = [[
            m * self.expansion for m in m_factor
        ] for idx, m_factor in enumerate(self.expand_ratio_list)]
        self.num_channels_list = \
            parse_values(self.arch_setting['out_channels_mult'])

        self.num_channels_list = [[
            make_divisible(m * out_channels[idx], 8) for m in m_factor
        ] for idx, m_factor in enumerate(self.num_channels_list)]

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels

        # stochastic depth decay rule
        total_depth = sum(max(nb) for nb in self.num_blocks_list)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        # net_num_blocks = sum(stage_blocks)
        # dpr = np.linspace(0, drop_path_rate, net_num_blocks)
        # block_id = 0
        for i, num_blocks in enumerate(self.num_blocks_list):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=max(num_blocks),
                in_channels=_in_channels,
                out_channels=max(self.num_channels_list[i]),
                expansion=max(self.expand_ratio_list[i]),
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:max(num_blocks)])
            _in_channels = max(self.num_channels_list[i])
            dpr = dpr[max(num_blocks):]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # import pdb; pdb.set_trace()
        self.register_mutables()

        self._freeze_stages()

        self.feat_dim = res_layer[res_layer.pure_module_nums - 1].out_channels

    def make_res_layer(self, block, num_blocks, in_channels, out_channels,
                       expansion, stride=1, avg_down=False, conv_cfg=None,
                       norm_cfg=dict(type='BN'), drop_path_rate=0.0, **kwargs):

        if isinstance(drop_path_rate, float):
            drop_path_rate = [drop_path_rate] * num_blocks

        assert len(drop_path_rate
                   ) == num_blocks, 'Please check the length of drop_path_rate'

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=drop_path_rate[0],
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=drop_path_rate[i],
                    **kwargs))
        dynamic_seq = DynamicSequential(*layers)
        return dynamic_seq

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(BigNASResNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        assert not self.deep_stem
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(BigNASResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def register_mutables(self):
        """Mutate the BigNAS-style ResNet."""
        # OneShotMutableChannelUnit._register_channel_container(
        #     self, MutableChannelContainer)

        mid_mutable = OneShotMutableChannel(
            alias='backbone.stem_channels',
            num_channels=self.stem_channels,
            candidate_choices=[self.stem_channels])

        # mutate the built mobilenet layers
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
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
            res_layer.register_mutable_attr('depth', mutable_depth)

            for k in range(max(num_blocks)):

                if self.fine_grained_mode:
                    mutable_expand_ratio = OneShotMutableValue(
                        alias=prefix + str(k) + '.expand_ratio',
                        value_list=expand_ratios)
                if self.block is BasicBlock:
                    mutate_basicblock(res_layer[k], mid_mutable,
                                        mutable_out_channels,
                                        mutable_expand_ratio)
                elif self.block is Bottleneck:
                    mutate_bottleneck(res_layer[k], mid_mutable,
                                        mutable_out_channels,
                                        mutable_expand_ratio)
                else:
                    raise ValueError(f'Unsupposed block type: {self.block}')
                mid_mutable = mutable_out_channels
        self.last_mutable_channels = mid_mutable

    def set_dropout(self, dropout_stages, drop_path_rate):
        layers = [getattr(self, layer_name) for i, layer_name in enumerate(self.res_layers)]
        set_dropout(layers=layers, module=self.block, 
                    dropout_stages=dropout_stages,
                    drop_path_rate=drop_path_rate)


@MODELS.register_module()
class BigNASResNetD(BigNASResNet):
    def __init__(self, *args, deep_stem=True, **kwargs):
        super().__init__(*args, deep_stem=deep_stem, **kwargs)

    def forward(self, x):
        assert self.deep_stem
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
