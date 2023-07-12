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

from mmrazor.models.architectures.dynamic_ops import (DynamicLinear,
                                                      DynamicLinearMixin)

def update_qdype_qmin_qmax(self, bit):
    # TODO: calc qdype according quant_min, quant_max (rely on backend support)
    # reduce_range is False by default.
    qdtype = self.weight_fake_quant.dtype
    quant_min = self.weight_fake_quant.quant_min
    quant_max = self.weight_fake_quant.quant_max

    is_symmetric_range = False
    if abs(quant_min) == abs(quant_max):
        is_symmetric_range = True
    if qdtype == torch.quint8:
        quant_min = 0
        quant_max = 2**bit - 1
    elif qdtype == torch.qint8:
        quant_max = 2**(bit - 1) - 1
        if is_symmetric_range:
            quant_min = -2**(bit - 1) + 1
        else:
            quant_min = -2**(bit - 1)
    else:
        raise ValueError(f'Only support qint8 and quint8, got {dtype}')
    self.weight_fake_quant.quant_max = \
        self.weight_fake_quant.activation_post_process.quant_max = quant_max
    self.weight_fake_quant.quant_min = \
        self.weight_fake_quant.activation_post_process.quant_min = quant_min

def bypass(x):
    return x


class DynamicQLinear(nnqat.Linear, DynamicLinearMixin):

    _FLOAT_MODULE = DynamicLinear
    accepted_mutable_attrs = {'in_features', 'out_features', 'quant_bits'}

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, input):
        weight_fake_quant = self.weight_fake_quant
        weight, bias = self.get_dynamic_params()
        if 'quant_bits' in self.mutable_attrs:
            bit = self.mutable_attrs['quant_bits'].current_choice
            update_qdype_qmin_qmax(self, bit)
            if bit == 32:
                weight_fake_quant = bypass
        return F.linear(input, self.weight_fake_quant(weight), bias)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict
            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        qat_linear = super(DynamicQLinear, cls).from_float(mod)

        for attr, value in mod.mutable_attrs.items():
            assert attr in qat_linear.accepted_mutable_attrs, (
                f'Get invalid attrs, `{attr}` can not be accepted, '
                f'excepted in {qat_linear.accepted_mutable_attrs.keys()}')
            qat_linear.register_mutable_attr(attr, value)

        assert hasattr(mod, "qbconfig"), "Input float module must have qbconfig defined"
        for attr, value in mod.qbconfig.items():
            assert attr in ('quant_bits', ), (
                f'Get invalid attrs, `{attr}` can not be accepted, '
                f'excepted in {("quant_bits", )}')
            qat_linear.mutable_attrs[attr] = value

        return qat_linear

    @property
    def static_op_factory(self):
        return nn.Linear

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(mod)
