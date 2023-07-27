from typing import Dict

import torch.nn as nn
from typing import Callable
import torch.nn.functional as F

try:
    import torch.nn.qat as nnqat
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    nnqat = get_package_placeholder('torch>=1.13')

from mmrazor.models import BaseMutable
from mmrazor.models.architectures.dynamic_ops import (BigNasConv2d,
                                                      DynamicConvMixin)


class DynamicQConv2d(nnqat.Conv2d, DynamicConvMixin):

    _FLOAT_MODULE = BigNasConv2d
    _FLOAT_CONV_MODULE = BigNasConv2d

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, input):
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        weight, bias, padding = self.get_dynamic_params()
        return self.conv_func(input, self.weight_fake_quant(weight), bias,
                              self.stride, padding, self.dilation, groups)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

            Args:
               `mod`: a float module, either produced by torch.ao.quantization utilities
               or directly from user
        """
        qat_conv = super(DynamicQConv2d, cls).from_float(mod)

        for attr, value in mod.mutable_attrs.items():
            qat_conv.register_mutable_attr(attr, value)

        return qat_conv

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        return nn.Conv2d

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(module)
