# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.parameter import Parameter

from mmrazor.registry import MODELS
from mmrazor.models import LearnableFakeQuantize

try:
    from torch.ao.quantization import FakeQuantizeBase
    from torch.ao.quantization.observer import (MinMaxObserver,
                                                PerChannelMinMaxObserver)    
except ImportError:
    from mmrazor.utils import get_placeholder
    FakeQuantizeBase = get_placeholder('torch>=1.13')
    MinMaxObserver = get_placeholder('torch>=1.13')
    PerChannelMinMaxObserver = get_placeholder('torch>=1.13')    


def enable_param_learning(mod):
    """Enables learning of quantization parameters, if applicable. Example
    usage::

    # model is any PyTorch model model.apply(enable_param_learning)
    """
    if isinstance(mod, LearnableFakeQuantize):
        mod.enable_param_learning()


def enable_static_estimate(mod):
    """Enables static observer estimates, if applicable. Example usage::

    # model is any PyTorch model model.apply(enable_static_estimate)
    """
    if isinstance(mod, LearnableFakeQuantize):
        mod.enable_static_estimate()


def enable_val(mod):
    """Enable validation, if applicable. Example usage::

    # model is any PyTorch model model.apply(enable_val)
    """
    if isinstance(mod, LearnableFakeQuantize):
        mod.enable_val()


@MODELS.register_module()
class BatchLearnableFakeQuantize(LearnableFakeQuantize):
    """This is implementation of BatchQuantizer.
    More details can be found in: https://openreview.net/pdf?id=qQAtFdyDr-
    Note that the differences between LSQ and BatchLSQ:
        1. during training, instead of learning the scale and zero_point,
        we use extreme value estimators to caputre the range variation in the
        current batch, and learn a multiplicative residual delta_scale and
        an additive residual delta_zero_point.
        2. during test, we have to recalibrate the minmax statics by
        enabling the observer updatable and forwarding a few batches fo data.

    In addition to the attributes in the original FakeQuantize module, the
    LearnableFakeQuantize module also includes the following attributes to
    support quantization parameter learning.

    * :attr:`fake_quant_enabled` defines the flag for enabling fake
      quantization on the output.

    * :attr:`static_enabled` defines the flag for using observer's static
      estimation for scale and zero point.

    * :attr:`learning_enabled` defines the flag for enabling backpropagation
      for scale and zero point.

    Args:
        observer (module): Module for observing statistics on input tensors and
            calculating scale and zero-point.
        quant_min (int): Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max (int): Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.
        delta_scale (float): The initial value of the floating-point scale factor.
            Defaults to 1.
        delta_zero_point (float): The initial value of the floating-point zero-point.
            Defaults to 0.
        zero_point_trainable (bool): Whether the zero_point is trainable.
            Defaults to False.
        observer_kwargs (dict | optional): Arguments for the observer module.
    """

    def __init__(self,
                 observer,
                 quant_min=0,
                 quant_max=255,
                 delta_scale=1.,
                 delta_zero_point=0.,
                 zero_point_trainable=False,
                 extreme_estimator=0,
                 **observer_kwargs):
        super(LearnableFakeQuantize, self).__init__()
        assert quant_min < quant_max, \
            'quant_min must be strictly less than quant_max.'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # also pass quant_min and quant_max to observer
        observer_kwargs['quant_min'] = quant_min
        observer_kwargs['quant_max'] = quant_max

        self.delta_scale = Parameter(torch.tensor([delta_scale]))
        self.delta_zero_point_trainable = zero_point_trainable
        if zero_point_trainable:
            self.delta_zero_point = Parameter(torch.tensor([delta_zero_point]))
        else:
            self.register_buffer('delta_zero_point', torch.tensor([delta_zero_point]))

        self.activation_post_process = observer(**observer_kwargs)
        assert \
            torch.iinfo(self.activation_post_process.dtype).min <= quant_min, \
            'quant_min out of bound'
        assert \
            quant_max <= torch.iinfo(self.activation_post_process.dtype).max, \
            'quant_max out of bound'
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        if self.qscheme in (torch.per_channel_symmetric,
                            torch.per_channel_affine):
            raise ValueError(f'Only support per_tensor now.')
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        self.register_buffer('fake_quant_enabled',
                             torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('static_enabled',
                             torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('learning_enabled',
                             torch.tensor([0], dtype=torch.uint8))

        bitrange = torch.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        self.register_buffer('eps',
                             torch.tensor([torch.finfo(torch.float32).eps]))
        self.extreme_estimator = extreme_estimator                             

    @torch.jit.export
    def enable_param_learning(self):
        """Enables learning of quantization parameters and disables static
        observer estimates.

        Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=True) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=False)
        return self

    @torch.jit.export
    def enable_static_estimate(self):
        """Enables static observer estimates and disables learning of
        quantization parameters.

        Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=True)

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
    def enable_static_observation(self):
        """Enables static observer accumulating data from input but doesn't
        update the quantization parameters.

        Forward path returns the original X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=False) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def toggle_observer_update(self, enabled=True):
        """Toggles whether static observer accumulates data from input."""
        self.static_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def enable_observer(self, enabled=True):
        """Enables static observer accumulating data from input."""
        self.toggle_observer_update(enabled)

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        """Toggles whether the quantization parameters are learnable."""
        self.learning_enabled[0] = int(enabled)
        self.delta_scale.requires_grad = enabled
        if self.delta_zero_point_trainable:
            self.delta_zero_point.requires_grad = enabled
        return self

    @torch.jit.export
    def toggle_fake_quant(self, enabled=True):
        """Toggles whether the fake quantization is enabled."""
        self.fake_quant_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def observe_quant_params(self):
        """Shows the quantization parameters."""
        # print('LearnableFakeQuantize Scale: {}'.format(self.delta_scale.detach()))
        # print('LearnableFakeQuantize Zero Point: {}'.format(
        #     self.delta_zero_point.detach()))
        raise NotImplementedError()

    @torch.jit.export
    def calculate_qparams(self):
        """Calculate the quantization parameters."""
        # self.delta_scale.data.clamp_(min=self.eps.item())
        # scale = self.delta_scale.detach()
        # zero_point = self.delta_zero_point.detach().round().clamp(
        #     self.quant_min, self.quant_max).long()
        # return scale, zero_point
        raise NotImplementedError()

    def forward(self, X):
        """Forward computation.

        Forward path returns fake quantized X.
        """
        if self.static_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = \
                self.activation_post_process.calculate_qparams()
        else:
            if self.extreme_estimator == 0:
                min_val_cur, max_val_cur = torch.aminmax(X.detach())
            elif self.extreme_estimator == 1:
                x_dim = X.size()
                new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
                new_axis_list[1] = 0
                new_axis_list[0] = 1
                y = X.permute(new_axis_list)
                # Need to match dtype of min/max because the updates to buffers
                # are done in place and types need to match for comparisons
                # y = y.to(self.min_val.dtype)??
                y = torch.flatten(y, start_dim=1)
                min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
                min_val_cur, max_val_cur = min_val_cur.mean(), max_val_cur.mean()
            elif self.extreme_estimator == 2:
                mean, var = X.detach().mean(), X.detach().var()
                min_val_cur, max_val_cur = mean - 3 * var, mean + 3 * var
            else:
                raise ValueError(f'Unsupported extreme estimator type: {self.extreme_estimator}')
            _scale, _zero_point = MinMaxObserver._calculate_qparams(
                self.activation_post_process, min_val_cur, max_val_cur)

        if self.fake_quant_enabled[0] == 1:
            scale = _scale * self.delta_scale
            zero_point = _zero_point + self.delta_zero_point
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
        local_state = ['delta_scale', 'delta_zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'delta_scale':
                    self.delta_scale.data = self.delta_scale.data.resize_(val.shape)
                else:
                    assert name == 'delta_zero_point'
                    self.delta_zero_point.data = self.delta_zero_point.data.resize_(
                        val.shape)
                # For torchscript module we need to update the attributes here
                # since we do not call the `_load_from_state_dict` function
                # defined module.py
                if torch.jit.is_scripting():
                    if name == 'delta_scale':
                        self.delta_scale.copy_(val)
                    else:
                        assert name == 'delta_zero_point'
                        self.delta_zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super(LearnableFakeQuantize,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    @torch.jit.export
    def extra_repr(self):
        """The printable representational string."""
        repr_str = f'static_enabled={self.static_enabled}, '
        repr_str += f'fake_quant_enabled={self.fake_quant_enabled}, '
        repr_str += f'quant_min={self.activation_post_process.quant_min}, '
        repr_str += f'quant_max={self.activation_post_process.quant_max}, '
        repr_str += f'dtype={self.dtype}, '
        repr_str += f'qscheme={self.qscheme}, '
        repr_str += f'delta_scale={self.delta_scale}, '
        repr_str += f'delta_zero_point={self.delta_zero_point}, '
        repr_str += f'zero_point_trainable={self.delta_zero_point_trainable}'
        return repr_str
