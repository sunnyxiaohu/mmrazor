# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from mmrazor.registry import MODELS
from mmrazor.models import LearnableFakeQuantize
from mmrazor.models.architectures.dynamic_ops import DynamicMixin

try:
    from torch.ao.quantization import FakeQuantizeBase
except ImportError:
    from mmrazor.utils import get_placeholder
    FakeQuantizeBase = get_placeholder('torch>=1.13')


def update_qdype_qmin_qmax(fake_quant, bit):
    # TODO: calc qdype according quant_min, quant_max (rely on backend support)
    # reduce_range is False by default.
    qdtype = fake_quant.dtype
    quant_min = fake_quant.quant_min
    quant_max = fake_quant.quant_max

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
        raise ValueError(f'Only support qint8 and quint8, got {qdtype}')
    fake_quant.quant_max = \
        fake_quant.activation_post_process.quant_max = quant_max
    fake_quant.quant_min = \
        fake_quant.activation_post_process.quant_min = quant_min


@MODELS.register_module()
class DynamicLearnableFakeQuantize(LearnableFakeQuantize, DynamicMixin):
    """This is an extension of the FakeQuantize module in fake_quantize.py,
    which supports learning of the scale and zero point parameters through
    backpropagation.

    In addition to the attributes in the original FakeQuantize module, the
    DynamicLearnableFakeQuantize module also includes the following attributes to
    support quantization parameter learning.
    """
    accepted_mutable_attrs = {'quant_bits'}
    FLOAT_BITS = 32

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @property
    def static_op_factory(self):
        return LearnableFakeQuantize

    def convert_from(cls, module):
        """Convert a nn.Linear module to a DynamicLinear.

        Args:
            module (:obj:`torch.nn.Linear`): The original Linear module.
        """
        raise NotImplementedError()

    def to_static_op(self) -> nn.Module:
        # quant_bits, index = self.get_dynamic_params()
        # if quant_bits == self.FLOAT_BITS:
        #     return 
        raise NotImplementedError()

    def register_mutable_attr(self, attr, mutable):
        assert hasattr(self, 'mutable_attrs')
        if attr == 'quant_bits':
            device = self.scale.device
            self.scale.data = torch.ones(1, len(mutable.choices)).to(device)
            self.zero_point.data = torch.zeros(1, len(mutable.choices)).to(device)
            self.mutable_attrs['quant_bits'] = mutable
        else:
            raise NotImplementedError

    def get_dynamic_params(self):
        if 'quant_bits' in self.mutable_attrs:
            quant_bits = self.mutable_attrs['quant_bits'].current_choice
            update_qdype_qmin_qmax(self, quant_bits)
            index = self.mutable_attrs['quant_bits'].choices.index(quant_bits)
        else:
            # TODO: handle 
            if abs(self.quant_min) == abs(self.quant_max):
                quant_bits = int(math.log(self.quant_max - self.quant_min + 2, 2))
            else:
                quant_bits = int(math.log(self.quant_max - self.quant_min + 1, 2))
            index = None                
        return quant_bits, index

    @torch.jit.export
    def calculate_qparams(self):
        """Calculate the quantization parameters."""
        raise NotImplementedError()


    def forward(self, X):
        """Forward computation.

        Forward path returns fake quantized X.
        """
        quant_bits, index = self.get_dynamic_params()
        if quant_bits == self.FLOAT_BITS:
            return X

        org_static_enabled = self.static_enabled[0]
        if index is not None:
            local_scale = self.scale.data[:, index]
        else:
            local_scale = self.scale.data
        # Check whether is initialized or not. We choose scale as indicator
        # since zero_point may also be zero after initialized.
        if torch.equal(local_scale, torch.ones_like(local_scale)):
            self.static_enabled[0] = 1
        if self.static_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = \
                self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.qscheme in (torch.per_channel_symmetric,
                                torch.per_channel_affine):
                self.scale.data.repeat(1, len(_scale))
                self.zero_point.data.repeat(1, len(_zero_point))

            if index is not None:
                self.scale.data[:, index] = _scale
                self.zero_point.data[:, index] = _zero_point
            else:
                self.scale.data.copy_(_scale)
                self.zero_point.data.copy_(_zero_point)

            self.activation_post_process.reset_min_max_vals()                
        else:
            self.scale.data.clamp_(min=self.eps.item())

        self.static_enabled[0] = org_static_enabled
        # import pdb; pdb.set_trace()
        if self.fake_quant_enabled[0] == 1:
            scale = self.scale[:, index] if index is not None else self.scale
            zero_point = self.zero_point[:, index] if index is not None else self.zero_point
            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max)**0.5
            else:
                grad_factor = 1.0
            if self.qscheme in (torch.per_channel_symmetric,
                                torch.per_channel_affine):
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, scale, zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if not (self.quant_min <= zero_point <= self.quant_max):
                    print(self.quant_min, zero_point, self.quant_max)
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, self.quant_min,
                    self.quant_max, grad_factor)

        return X

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Removing this function throws an error that the the size of the
        loaded tensor does not match the original size i.e., These buffers
        start out with numel 0 and become numel 1 once they have their first
        forward pass.

        Modified from https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fake_quantize.py  # noqa:E501
        """
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                local_val = getattr(self, name)
                # import pdb; pdb.set_trace()
                if local_val.shape == val.shape:
                    continue

                if name == 'scale':
                    self.scale.data.repeat(val.size(0), 1)
                else:
                    assert name == 'zero_point'
                    self.zero_point.data.repeat(val.size(0), 1)
                state_dict[key] = val.repeat(1, local_val.size(1))
                # For torchscript module we need to update the attributes here
                # since we do not call the `_load_from_state_dict` function
                # defined module.py
                # TODO: handle torch.jit.is_scripting
                # if torch.jit.is_scripting():
                #     if name == 'scale':
                #         self.scale.copy_(val)
                #     else:
                #         assert name == 'zero_point'
                #         self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super(LearnableFakeQuantize,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)


@MODELS.register_module()
class DynamicBatchLearnableFakeQuantize(DynamicLearnableFakeQuantize):
    """This is implementation of BatchQuantizer.
    More details can be found in: https://openreview.net/pdf?id=qQAtFdyDr-
    Note that the differences between LSQ and BatchLSQ:
        1. during training, instead of learning the scale and zero_point,
        we use extreme value estimators to caputre the range variation in the
        current batch, and learn a multiplicative residual delta_scale and
        an additive residual delta_zero_point.
        2. during test, we have to recalibrate the minmax statics by
        enabling the observer updatable and forwarding a few batches fo data.
    """

    @torch.jit.export
    def enable_val(self):
        """Disables static observer accumulating data from input and doesn't
        update the quantization parameters.

        Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=True)
        self.activation_post_process.reset_min_max_vals()

    @torch.jit.export
    def observe_quant_params(self):
        """Shows the quantization parameters."""
        # print('LearnableFakeQuantize Scale: {}'.format(self.delta_scale.detach()))
        # print('LearnableFakeQuantize Zero Point: {}'.format(
        #     self.delta_zero_point.detach()))
        raise NotImplementedError()

    def forward(self, X):
        """Forward computation.

        Forward path returns fake quantized X.
        """
        quant_bits, index = self.get_dynamic_params()
        if quant_bits == self.FLOAT_BITS:
            return X
        # import pdb; pdb.set_trace()
        self.activation_post_process(X.detach())
        _scale, _zero_point = \
            self.activation_post_process.calculate_qparams()

        # TODO: Support per-channel according the shape of inputs.
        if self.fake_quant_enabled[0] == 1:
            delta_scale = self.scale[:, index] if index is not None else self.scale
            delta_zero_point = self.zero_point[:, index] if index is not None else self.zero_point
            scale = _scale * delta_scale
            zero_point = _zero_point + delta_zero_point
            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max)**0.5
            else:
                grad_factor = 1.0
            if self.qscheme in (torch.per_channel_symmetric,
                                torch.per_channel_affine):
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, scale, zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if not (self.quant_min <= zero_point <= self.quant_max):
                    print(self.quant_min, zero_point, self.quant_max)
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, self.quant_min,
                    self.quant_max, grad_factor)

        return X
