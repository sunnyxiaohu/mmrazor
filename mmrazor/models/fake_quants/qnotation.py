# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS

try:
    from torch.ao.quantization import FakeQuantizeBase
except ImportError:
    from mmrazor.utils import get_placeholder
    FakeQuantizeBase = get_placeholder('torch>=1.13')


class QFakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, maxinum, fl, quant_min, quant_max):
        """ 量化函数，使用直通估计器（STE）保持梯度 """

        float_to_q = torch.floor(X * 2 ** fl + 0.5)  # 避免 round() 梯度丢失
        float_to_q.clamp_(min=quant_min, max=quant_max)
        # assert float_to_q < 2 ** (bitwidth-1) -1
        q_to_float = float_to_q.clone()  # 避免 in-place 操作影响梯度

        X_q = q_to_float / (2 ** fl)

        return X_q

    @staticmethod
    def backward(ctx, grad_output):
        """ 直通估计器（STE），只对 X 计算梯度 """
        return grad_output, None, None, None, None


@MODELS.register_module()
class QNotationFakeQuantize(FakeQuantizeBase):
    """This is an extension of the FakeQuantize module in fake_quantize.py,
    which supports learning of the scale and zero point parameters through
    backpropagation.

    In addition to the attributes in the original FakeQuantize module, the
    QNotationFakeQuantize module also includes the following attributes to
    support quantization parameter learning.

    * :attr:`fake_quant_enabled` defines the flag for enabling fake
      quantization on the output.

    Args:
        observer (module): Module for observing statistics on input tensors and
            calculating scale and zero-point.
        quant_min (int): Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max (int): Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.
        observer_kwargs (dict | optional): Arguments for the observer module.
    """

    def __init__(self,
                 observer,
                 quant_min=0,
                 quant_max=255,
                 **observer_kwargs):
        super(QNotationFakeQuantize, self).__init__()
        assert quant_min < quant_max, \
            'quant_min must be strictly less than quant_max.'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # also pass quant_min and quant_max to observer
        observer_kwargs['quant_min'] = quant_min
        observer_kwargs['quant_max'] = quant_max

        self.activation_post_process = observer(**observer_kwargs)
        assert \
            torch.iinfo(self.activation_post_process.dtype).min <= quant_min, \
            'quant_min out of bound'
        assert \
            quant_max <= torch.iinfo(self.activation_post_process.dtype).max, \
            'quant_max out of bound'
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1

        bitrange = torch.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        self.register_buffer('scale', torch.tensor([self.bitwidth + 0.]))  # maxinum
        self.register_buffer('zero_point', torch.tensor([0.0]))  # fl

    @torch.jit.export
    def calculate_qparams(self):
        """Calculate the quantization parameters."""
        return self.scale, self.zero_point

    def forward(self, X):
        """Forward computation.

        Forward path returns fake quantized X.
        """

        if X.numel() == 0:
            return X

        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            max_val_pos = torch.max(-self.activation_post_process.min_val, self.activation_post_process.max_val)
            maxinum = torch.ceil(torch.log2(max_val_pos) + 1)
            fl = (self.bitwidth - maxinum)
            self.scale.data.copy_(maxinum)
            self.zero_point.data.copy_(fl)
        # import pdb; pdb.set_trace()
        if self.fake_quant_enabled[0] == 1:
            X = QFakeQuantizeFunction.apply(X, self.scale, self.zero_point, self.quant_min, self.quant_max)

        return X

    @torch.jit.export
    def extra_repr(self):
        """The printable representational string."""
        repr_str = f'fake_quant_enabled={self.fake_quant_enabled}, '
        repr_str += f'quant_min={self.activation_post_process.quant_min}, '
        repr_str += f'quant_max={self.activation_post_process.quant_max}, '
        repr_str += f'maxinum={self.scale}, '
        repr_str += f'fl={self.zero_point}, '
        repr_str += f'dtype={self.dtype}, '
        repr_str += f'qscheme={self.qscheme}.'
        return repr_str
