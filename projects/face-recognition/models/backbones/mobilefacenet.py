# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmengine.model import BaseModule

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Sequential,AdaptiveAvgPool2d
from mmrazor.registry import MODELS
from torch.nn.modules.batchnorm import _BatchNorm
import torch
try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
    from mmcls.models.utils import make_divisible
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')
    make_divisible = get_placeholder('mmcls')

class Flatten(BaseModule):
    def forward(self, x):
        # return x.view(x.size(0), -1)
        return torch.flatten(x, 1, -1)


class PoolBlock(BaseModule):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1)):
        super(PoolBlock, self).__init__()
        self.layers = nn.Sequential(
            AdaptiveAvgPool2d((1,1)),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)

class ConvBlock(BaseModule):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            ReLU(out_c)
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(BaseModule):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class DepthWise(BaseModule):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        )

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
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, True, kernel, stride, padding, groups))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(BaseModule):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            PoolBlock(512, 512, kernel=(7, 7), stride=(1, 1)),
            Flatten(),
            Linear(512, embedding_size, bias=True),
            BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)

@MODELS.register_module()
class MobileFaceNet(BaseBackbone):
    def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=2, init_cfg=None):
        super(MobileFaceNet, self).__init__(init_cfg)
        self.frozen_stages=-1
        self.norm_eval=False
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            )
        
        self.layers.extend(
        [
            DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),
            Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
            Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512),
            Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
        ])

        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(num_features)

    def init_weights(self):
        super().init_weights()
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for func in self.layers:
            x = func(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = self.conv_sep(x.float())
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
        super(MobileFaceNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
