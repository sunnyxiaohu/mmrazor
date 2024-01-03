from typing import Any, Dict, Optional, Set, Tuple

import copy
import torch
from torch import Tensor, nn

import torch.nn.functional as F

try:
    # import torch.nn.intrinsic.qat as nniqat
    import torch.nn.qat as nnqat
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    # nniqat = get_package_placeholder('torch>=1.13')
    nnqat = get_package_placeholder('torch>=1.13')

from mmrazor.models import BaseMutable
from ..dynamic_ops import DynamicLinear, DynamicMixin, DynamicLinearMixin

def traverse_children(module: nn.Module) -> None:
    for name, mutable in module.items():
        if isinstance(mutable, DynamicMixin):
            module[name] = mutable.to_static_op()
        if hasattr(mutable, '_modules'):
            traverse_children(mutable._modules)


class DynamicQLinear(nnqat.Linear, DynamicLinearMixin):

    _FLOAT_MODULE = DynamicLinear

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, input):
        weight, bias, out_mask = self.get_dynamic_params(self.weight_fake_quant(self.weight), self.bias)
        return F.linear(input, weight, bias)

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
        return nnqat.Linear

    @classmethod
    def convert_from(cls, module):
        return cls.from_float(module)

    def get_dynamic_params(self: nn.Linear, orig_weight=None, orig_bias=None) -> Tuple[Tensor, Optional[Tensor]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Sliced weight and bias.
        """
        if orig_weight is None and orig_bias is None:
            orig_weight, orig_bias = self.weight, self.bias

        if 'in_features' in self.mutable_attrs:
            in_mask = self.mutable_attrs['in_features'].current_mask.to(
                orig_weight.device)
        else:
            in_mask = torch.ones(orig_weight.size(1)).bool().to(
                orig_weight.device)
        if 'out_features' in self.mutable_attrs:

            out_mask = self.mutable_attrs['out_features'].current_mask.to(
                orig_weight.device)
        else:
            out_mask = torch.ones(orig_weight.size(0)).bool().to(
                orig_weight.device)

        weight = orig_weight[out_mask][:, in_mask]
        bias = orig_bias[out_mask] if orig_bias is not None else None

        return weight, bias, out_mask

    def to_static_op(self):
        weight, bias, out_mask = self.get_dynamic_params(self.weight, self.bias)
        out_channels = weight.size(0)
        in_channels = weight.size(1)

        cls = self.static_op_factory
        mod = cls._FLOAT_MODULE(  # type: ignore[attr-defined]
            in_channels,
            out_channels,
            bias=self.bias is not None)
        mod.weight = torch.nn.Parameter(weight)
        if bias is not None:
            mod.bias = torch.nn.Parameter(bias)

        fake_quant = self.weight_fake_quant.to_static_op()
        if len(fake_quant.scale) > 1 and len(fake_quant.scale) != out_channels:
          fake_quant.scale.data = fake_quant.scale.data[out_mask]
          fake_quant.zero_point.data = fake_quant.zero_point.data[out_mask]

        mod.qconfig = self.qconfig
        mod.train(self.training)
        mod = cls.from_float(mod)
        mod.weight_fake_quant = fake_quant
        return mod