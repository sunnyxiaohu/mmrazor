# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from mmrazor.registry import MODELS
from mmrazor.models import LearnableFakeQuantize
from mmrazor.models.architectures.dynamic_ops import DynamicMixin


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
    BASE_BITS = 4

    def __init__(self, *args, param_share_mode = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # mode 0: Unshared
        # mode 1: Full shared with all the bits sharing the same scale
        # mode 2: Reparameter and partially shared
        # mode 3: Full shared with `S_{k+1} = 0.5*(1 - 1/(2^{k+1} - 1))*S_k`
        # mode 4: mode 2 with adelta constrained
        assert param_share_mode in [0, 1, 2, 3, 4, 5], f'Unexpected param_share_mode: {param_share_mode}'
        self.param_share_mode = param_share_mode
        if self.param_share_mode in [2, 4]:
            self.scale_adelta = Parameter(torch.tensor([0.]))
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
            if self.param_share_mode == 0:
                self.scale.data = torch.ones(1, len(mutable.choices)).to(device)
                self.zero_point.data = torch.zeros(1, len(mutable.choices)).to(device)
            if self.param_share_mode in [2, 4]:
                self.scale_adelta.data = torch.zeros(1, len(mutable.choices)).to(device)
                self.zero_point.data = torch.zeros(1, len(mutable.choices)).to(device)
            self.mutable_attrs['quant_bits'] = mutable
        else:
            raise NotImplementedError
        self.BASE_BITS = self.mutable_attrs['quant_bits'].choices[0]

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
        if self.param_share_mode in [1, 3]:
            index = None
        return quant_bits, index

    @torch.jit.export
    def calculate_qparams(self):
        """Calculate the quantization parameters."""
        raise NotImplementedError()


    def forward(self, X):
        """Forward computation.

        Forward path returns fake quantized X.
        static_enabled transitate states, when param_share_mode==2:
        1: init base scale -> 2: init adelta scale -> 0: by learning
        """
        quant_bits, index = self.get_dynamic_params()
        if quant_bits == self.FLOAT_BITS:
            return X

        mult = 1.0 / 2 ** (quant_bits - self.BASE_BITS)
        org_static_enabled = self.static_enabled[0]
        if index is not None and self.param_share_mode == 0:
            local_scale = self.scale.data[:, index]
        else:
            local_scale = self.scale.data
        # Check whether is initialized or not. We choose scale as indicator
        # since zero_point may also be zero after initialized.
        scale_initialized = not torch.equal(local_scale, torch.ones_like(local_scale))
        scale_initialized = scale_initialized or not self.training
        if not scale_initialized:
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
                if self.param_share_mode in [2, 4]:
                    self.scale_adelta.data.repeat(1, len(_scale))

            if index is not None and self.param_share_mode == 0:
                self.scale.data[:, index] = _scale
                self.zero_point.data[:, index] = _zero_point
            elif index is not None and self.param_share_mode in [2, 4]:
                self.scale.data.copy_(_scale)
                self.zero_point.data[:, index] = _zero_point
            else:
                self.scale.data.copy_(_scale)
                self.zero_point.data.copy_(_zero_point)

            self.activation_post_process.reset_min_max_vals()
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        # transitate to initialize scale_delta.
        if scale_initialized and self.param_share_mode in [2, 4]:
            local_scale = self.scale_adelta.data[:, index]
            if torch.equal(local_scale, torch.zeros_like(local_scale)):
                self.activation_post_process(X.detach())
                _scale, _zero_point = \
                    self.activation_post_process.calculate_qparams()
                _scale = _scale.to(self.scale.device)
                _zero_point = _zero_point.to(self.zero_point.device)
                self.scale_adelta.data[:, index] = _scale - mult * self.scale.data
                self.activation_post_process.reset_min_max_vals()
            else:
                self.scale_adelta.data.clamp_(min=self.eps.item())

        self.static_enabled[0] = org_static_enabled
        # import pdb; pdb.set_trace()
        if self.fake_quant_enabled[0] == 1:
            if index is None:
                if 'quant_bits' in self.mutable_attrs and len(self.mutable_attrs['quant_bits'].choices) > 1:
                    assert not self.zero_point_trainable
                    self.activation_post_process(X.detach())
                    _, _zero_point = self.activation_post_process.calculate_qparams()
                    _zero_point = _zero_point.to(self.zero_point.device)
                    self.zero_point.data.copy_(_zero_point)
                    self.activation_post_process.reset_min_max_vals()
                zero_point = self.zero_point
                scale = self.scale
                if self.param_share_mode == 3:
                    scale = mult * scale
            else:
                zero_point = self.zero_point[:, index]
                if self.param_share_mode in [2, 4]:
                    scale = self.scale
                    idx = 0
                    pre_bits = self.BASE_BITS
                    # import pdb; pdb.set_trace()
                    while(idx <= index):
                        quant_bits = self.mutable_attrs['quant_bits'].choices[idx]
                        mult = 1.0 / 2 ** (quant_bits - pre_bits)
                        if self.param_share_mode == 4:
                            M = 0.3  # 0.7  # 0.05
                            clip_m = mult * scale * M
                            self.scale_adelta.data[:, idx] = torch.clamp(self.scale_adelta.data[:, idx], -clip_m, clip_m)
                        scale_adelta = self.scale_adelta[:, idx]
                        scale = mult * scale + scale_adelta
                        idx += 1
                        pre_bits = quant_bits
                else:
                    scale = self.scale[:, index]
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
    def __init__(self, *args, residual_mode = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_mode = residual_mode
        assert residual_mode in [0, 1], f'Unexpected residual_mode: {residual_mode}'
        assert self.param_share_mode in [0, 1, 3, 4, 5], f'Unexpected param_share_mode: {self.param_share_mode}'
        if self.param_share_mode == 4:
            self.scale_adelta.data = torch.tensor([1.])

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
        quant_bits, index = self.get_dynamic_params()
        _scale, _zero_point = \
            self.activation_post_process.calculate_qparams()
        self.scale.data.abs_()
        self.scale.data.clamp_(min=self.eps.item())
        delta_add = self.zero_point[:, index] if index is not None else self.zero_point
        if self.param_share_mode in [3, 5]:
            assert self.residual_mode == 0, f'Unsuported residual_mode: {self.residual_mode}'
            M = 0.2
            delta_mult = self.scale * (1 + M) ** (quant_bits - self.BASE_BITS)
            scale = _scale * delta_mult
            zero_point = _zero_point + delta_add
        elif self.param_share_mode == 4:
            # import pdb; pdb.set_trace()
            assert self.residual_mode == 0, f'Unsuported residual_mode: {self.residual_mode}'
            idx = 0
            while(idx < len(self.mutable_attrs['quant_bits'].choices)):
                idx_quant_bits = self.mutable_attrs['quant_bits'].choices[idx]
                M = 0.3  # 0.7  # 0.05
                idx_delta_mult = self.scale_adelta[:, idx]
                clip_m = idx_delta_mult * (1 + M) ** (quant_bits - idx_quant_bits)
                if idx == index:
                    idx += 1
                    continue
                elif idx > index:
                    self.scale_adelta.data[:, index] = torch.clamp(self.scale_adelta.data[:, index], min=clip_m)
                else:
                    self.scale_adelta.data[:, index] = torch.clamp(self.scale_adelta.data[:, index], max=clip_m)
                idx += 1
            self.scale_adelta.data.abs_()
            self.scale_adelta.data.clamp_(min=self.eps.item())
            delta_mult = self.scale_adelta[:, index]
            scale = _scale * delta_mult
            zero_point = _zero_point + delta_add
        else:
            delta_mult = self.scale[:, index] if index is not None else self.scale
            if self.residual_mode == 0:
                scale = _scale * delta_mult
                zero_point = _zero_point + delta_add
            elif self.residual_mode == 1:
                scale = _scale * delta_mult + delta_add
                zero_point = _zero_point.float()
        return scale, zero_point

    def register_mutable_attr(self, attr, mutable):
        super().register_mutable_attr(attr, mutable)
        device = self.scale.device
        if self.param_share_mode == 4:
            self.scale_adelta.data = torch.ones_like(self.scale_adelta).to(device)
        if self.param_share_mode == 5:
            self.zero_point.data = torch.zeros(1, len(mutable.choices)).to(device)

    def forward(self, X):
        """Forward computation.

        Forward path returns fake quantized X.
        """
        quant_bits, index = self.get_dynamic_params()
        if quant_bits == self.FLOAT_BITS:
            return X
        # import pdb; pdb.set_trace()
        self.activation_post_process(X.detach())

        # TODO: Support per-channel according the shape of inputs.
        if self.fake_quant_enabled[0] == 1:
            scale, zero_point = self.observe_quant_params()

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
                # if not (self.quant_min <= zero_point <= self.quant_max):
                #     print(self.quant_min, zero_point, self.quant_max)
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, self.quant_min,
                    self.quant_max, grad_factor)

        return X
