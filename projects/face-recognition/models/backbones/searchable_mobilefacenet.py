# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn import (AdaptiveAvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d,
                      Linear, ReLU, Sequential)
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.registry import MODELS

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')
from mmrazor.models.architectures.dynamic_ops import (BigNasConv2d,
                                                      DynamicBatchNorm2d,
                                                      DynamicSequential)
from mmrazor.models.architectures.utils.mutable_register import (
    mutate_conv_module)                                                      
from mmrazor.models.mutables import (OneShotMutableChannel,
                                     OneShotMutableValue)
from mmrazor.models.utils import make_divisible


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


class Flatten(BaseModule):

    def forward(self, x):
        # return x.view(x.size(0), -1)
        return torch.flatten(x, 1, -1)


class PoolBlock(BaseModule):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1)):
        super(PoolBlock, self).__init__()
        self.layers = nn.Sequential(
            AdaptiveAvgPool2d((1, 1)), DynamicBatchNorm2d(num_features=out_c))

    def forward(self, x):
        return self.layers(x)


# class ConvBlock(BaseModule):

#     def __init__(self,
#                  in_c,
#                  out_c,
#                  kernel=(1, 1),
#                  stride=(1, 1),
#                  padding=(0, 0),
#                  groups=1):
#         super(ConvBlock, self).__init__()
#         self.layers = nn.Sequential(
#             BigNasConv2d(
#                 in_c,
#                 out_c,
#                 kernel,
#                 groups=groups,
#                 stride=stride,
#                 padding=padding,
#                 bias=False), DynamicBatchNorm2d(num_features=out_c), ReLU(out_c))

    def forward(self, x):
        return self.layers(x)


# class LinearBlock(BaseModule):

#     def __init__(self,
#                  in_c,
#                  out_c,
#                  kernel=(1, 1),
#                  stride=(1, 1),
#                  padding=(0, 0),
#                  groups=1):
#         super(LinearBlock, self).__init__()
#         self.layers = nn.Sequential(
#             BigNasConv2d(
#                 in_c,
#                 out_c,
#                 kernel,
#                 stride,
#                 padding,
#                 groups=groups,
#                 bias=False), DynamicBatchNorm2d(num_features=out_c))

#     def forward(self, x):
#         return self.layers(x)


class DepthWise(BaseModule):

    def __init__(self,
                 in_c,
                 out_c,
                 residual=False,
                 kernel=(3, 3),
                 stride=(2, 2),
                 padding=(1, 1),
                 groups=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvModule(
                in_channels=in_c,
                out_channels=groups,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=groups,
                out_channels=groups,
                groups=groups,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=groups,
                out_channels=out_c,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(BaseModule):

    def __init__(self,
                 c,
                 num_block,
                 groups,
                 kernel=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                DepthWise(c, c, True, kernel, stride, padding, groups,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.layers = DynamicSequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(BaseModule):

    def __init__(self, embedding_size, mid_features=512):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            PoolBlock(None, mid_features, kernel=(7, 7), stride=(1, 1)), Flatten(),
            Linear(mid_features, embedding_size, bias=True),
            BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)

def mutate_block(block_layer: DepthWise,
                 mutable_in_channels,
                 mutable_out_channels,
                 mutable_mid_channels):
    """Mutate DepthWise layers."""

    mutate_conv_module(block_layer.layers[0], mutable_in_channels, mutable_mid_channels)
    mutate_conv_module(block_layer.layers[1], mutable_mid_channels, mutable_mid_channels)
    mutate_conv_module(block_layer.layers[2], mutable_mid_channels, mutable_out_channels)


@MODELS.register_module()
class SearchableMobileFaceNet(BaseBackbone):

    ref_arch_settings = [
        # out_channel, mid_channel, num_blocks, downsample_channel
        [64, 128, 2, 128],
        [128, 128, 4, 256],
        [128, 256, 6, 512],
        [0, 256, 2, 0],
    ]

    def __init__(self,
                 num_features=512,
                 fp16=True,
                 arch_setting=None,
                 conv_cfg: Dict = dict(type='BigNasConv2d'),
                 norm_cfg: Dict = dict(type='DynamicBatchNorm2d'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(SearchableMobileFaceNet, self).__init__(init_cfg)
        self.fp16 = fp16
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg        
        self.frozen_stages = -1
        self.norm_eval = False
        self.arch_setting = arch_setting
        # adapt mutable settings
        self.num_blocks_list = parse_values(self.arch_setting['num_blocks_add'])
        self.out_channels_list = parse_values(self.arch_setting['out_channels_mult'])
        self.mid_channels_list = parse_values(self.arch_setting['mid_channels_mult'])
        self.ds_channels_list = parse_values(self.arch_setting['ds_channels_mult'])

        in_channel = 64
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvModule(
                in_channels=3,
                out_channels=in_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i, (out_channel, mid_channel, num_blocks, ds_channel) in enumerate(self.ref_arch_settings):
            if i == 0:
                assert num_blocks > 1, f'Only support num_blocks > 1 for layer1'
            self.num_blocks_list[i] = [
                int(a + num_blocks) for a in self.num_blocks_list[i]]
            self.out_channels_list[i] = [
                make_divisible(m * out_channel, 32) if out_channel > 0 else 0 for m in self.out_channels_list[i]]
            self.mid_channels_list[i] = [
                make_divisible(m * mid_channel, 32) for m in self.mid_channels_list[i]]
            self.ds_channels_list[i] = [
                make_divisible(m * ds_channel, 32) if ds_channel > 0 else 0 for m in self.ds_channels_list[i]]                
            out_channel = max(self.out_channels_list[i])
            mid_channel = max(self.mid_channels_list[i])
            num_blocks = max(self.num_blocks_list[i])
            ds_channel = max(self.ds_channels_list[i])
            layer, out_channel = self.make_layer(
                in_channel, out_channel, mid_channel, num_blocks, ds_channel, conv_block=(i==0 and num_blocks==1))
            in_channel = out_channel
            self.layers.append(layer)
        mid_features = 512            
        self.conv_sep = ConvModule(
                in_channels=in_channel,
                out_channels=mid_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        self.features = GDC(num_features, mid_features=mid_features)
        self.register_mutables()

    def register_mutables(self):
        """Mutate the BigNAS-style ResNet."""

        mid_mutable = OneShotMutableChannel(
            alias='backbone.layers.0.out_channels',
            num_channels=self.layers[0].out_channels,
            candidate_choices=[self.layers[0].out_channels])

        # mutate the built mobilenet layers
        for i, layer in enumerate(self.layers[1:]):
            num_blocks = self.num_blocks_list[i]
            mid_channel = self.mid_channels_list[i]
            out_channel = self.out_channels_list[i]
            ds_channel = self.ds_channels_list[i]

            prefix = 'backbone.layers.' + str(i) + '.'

            mutable_out_channel = OneShotMutableChannel(
                alias=prefix + 'out_channel',
                candidate_choices=out_channel,
                num_channels=max(out_channel))
            mutable_mid_channel = OneShotMutableChannel(
                alias=prefix + 'mid_channel',
                candidate_choices=mid_channel,
                num_channels=max(mid_channel))
            mutable_ds_channel = OneShotMutableChannel(
                alias=prefix + 'ds_channel',
                candidate_choices=ds_channel,
                num_channels=max(ds_channel))                
            # if not self.fine_grained_mode:
            #     mutable_expand_ratio = OneShotMutableValue(
            #         alias=prefix + 'expand_ratio', value_list=expand_ratios)

            mutable_depth = OneShotMutableValue(
                alias=prefix + 'depth', value_list=num_blocks)
            layer[0].layers.register_mutable_attr('depth', mutable_depth)

            for k in range(max(num_blocks)):

                # if self.fine_grained_mode:
                #     mutable_expand_ratio = OneShotMutableValue(
                #         alias=prefix + str(k) + '.expand_ratio',
                #         value_list=expand_ratios)
                mutate_block(layer[0].layers[k], mid_mutable, mid_mutable, mutable_mid_channel)
            if len(layer) > 1:
                assert len(layer) == 2
                mutate_block(layer[1], mid_mutable, mutable_out_channel, mutable_ds_channel)
                mid_mutable = mutable_out_channel
        self.last_mutable_channels = OneShotMutableChannel(
            alias='backbone.conv_sep.out_channel',
            num_channels=self.conv_sep.out_channels,
            candidate_choices=[self.conv_sep.out_channels])
        mutate_conv_module(self.conv_sep, mid_mutable, self.last_mutable_channels)

    def make_layer(self, in_channel, out_channel, mid_channel, num_blocks, ds_channel, conv_block=False):
        layer = []
        if conv_block:
            block = ConvModule(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=mid_channel,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)                
        else:
            block = Residual(
                in_channel,
                num_block=num_blocks,
                groups=mid_channel,
                kernel=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        layer.append(block)
        if ds_channel > 0:
            layer.append(DepthWise(
                in_channel,
                out_channel,
                kernel=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                groups=ds_channel,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        else:
            out_channel = in_channel                
        return nn.Sequential(*layer), out_channel

    def init_weights(self):
        super().init_weights()
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, BigNasConv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, DynamicBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for func in self.layers:
            x = func(x)
        with torch.cuda.amp.autocast(enabled=False):
            if self.fp16:
                x = x.float()
            x = self.conv_sep(x)
            x = self.features(x)
        return (x, )

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
        super(SearchableMobileFaceNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
