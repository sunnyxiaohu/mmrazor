import torch.nn as nn
from typing import Callable
import torch.nn.functional as F

try:
    import torch.nn.qat as nnqat
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    nnqat = get_package_placeholder('torch>=1.13')

from mmrazor.models.architectures.dynamic_ops import (BigNasConv2d,
                                                      DynamicConvMixin)

from mmengine.utils import import_modules_from_strings
custom_imports = 'projects.nas-mqbench.models.architectures.dynamic_qops.dynamic_qlinear'
dynamic_qlinear = import_modules_from_strings(custom_imports)


class DynamicQConv2d(nnqat.Conv2d, DynamicConvMixin):

    _FLOAT_MODULE = BigNasConv2d
    _FLOAT_CONV_MODULE = BigNasConv2d
    accepted_mutable_attrs = {'in_channels', 'out_channels', 'quant_bits'}

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, input):
        weight_fake_quant = self.weight_fake_quant
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        weight, bias, padding = self.get_dynamic_params()
        if 'quant_bits' in self.mutable_attrs:
            bit = self.mutable_attrs['quant_bits'].current_choice
            dynamic_qlinear.update_qdype_qmin_qmax(self, bit)
            if bit == 32:
                weight_fake_quant = dynamic_qlinear.bypass             
        return self.conv_func(input, weight_fake_quant(weight), bias, 
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
            assert attr in qat_conv.accepted_mutable_attrs, (
                f'Get invalid attrs, `{attr}` can not be accepted, '
                f'excepted in {qat_conv.accepted_mutable_attrs.keys()}')
            qat_conv.register_mutable_attr(attr, value)

        assert hasattr(mod, "qbconfig"), "Input float module must have qbconfig defined"
        for attr, value in mod.qbconfig.items():
            assert attr in ('quant_bits', ), (
                f'Get invalid attrs, `{attr}` can not be accepted, '
                f'excepted in {("quant_bits", )}')
            qat_conv.mutable_attrs[attr] = value

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
        return cls.from_float(mod)
