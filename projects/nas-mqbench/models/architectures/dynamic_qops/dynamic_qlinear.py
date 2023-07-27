from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # import torch.nn.intrinsic.qat as nniqat
    import torch.nn.qat as nnqat
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    # nniqat = get_package_placeholder('torch>=1.13')
    nnqat = get_package_placeholder('torch>=1.13')

from mmrazor.models import BaseMutable
from mmrazor.models.architectures.dynamic_ops import (DynamicLinear,
                                                      DynamicLinearMixin)


class DynamicQLinear(nnqat.Linear, DynamicLinearMixin):

    _FLOAT_MODULE = DynamicLinear

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, input):
        weight, bias = self.get_dynamic_params()
        return F.linear(input, self.weight_fake_quant(weight), bias)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict
            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        qat_linear = super(DynamicQLinear, cls).from_float(mod)

        for attr, value in mod.mutable_attrs.items():
            qat_linear.register_mutable_attr(attr, value)

        return qat_linear

    @property
    def static_op_factory(self):
        return nn.Linear

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(module)

