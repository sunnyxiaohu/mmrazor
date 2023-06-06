# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Swish
from mmengine.model import BaseModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmrazor.registry import MODELS
from ..backbones.csp_layer import CSPLayer

@MODELS.register_module()
class YOLOXPAFPNQAT(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOXPAFPNQAT, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        
        # backbone output connect for qat
        self.actconnnect = build_activation_layer(act_cfg)
        
        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))   
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
    
        x0 = inputs[0]   # low output
        x1 = inputs[1]   # middle output
        
        x0 = self.actconnnect(x0)
        x1 = self.actconnnect(x1)
        
        x2 = inputs[2]   # high output
        
        # top-down path
        x2_for_upsample = self.reduce_layers[0](x2)
        x2_for_concat = self.reduce_layers[1](x2)
        
        x2_for_upsample = self.upsample(x2_for_upsample)
        x1 = self.top_down_blocks[0](torch.cat([x2_for_upsample, x1], 1))
        
        x1_for_upsample = self.reduce_layers[2](x1)
        x1_for_concat = self.reduce_layers[3](x1)
        
        x1_for_upsample = self.upsample(x1_for_upsample)
        x0_out = self.top_down_blocks[1](torch.cat([x1_for_upsample, x0], 1))
        
        # bottom-up paths
        x0_out_downsample = self.downsamples[0](x0_out)
        x1_out = self.bottom_up_blocks[0](torch.cat([x0_out_downsample, x1_for_concat], 1))
        
        x1_out_downsample = self.downsamples[1](x1_out)
        x2_out = self.bottom_up_blocks[1](torch.cat([x1_out_downsample, x2_for_concat], 1))
        
        # out convs
        outs = []
        outs.append(self.out_convs[0](x0_out))
        outs.append(self.out_convs[1](x1_out))
        outs.append(self.out_convs[2](x2_out))
        
        return tuple(outs)
        
