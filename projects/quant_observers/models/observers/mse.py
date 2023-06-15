# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.distributed as dist

from mmrazor.models.observers.base import BaseObserver
from mmrazor.registry import MODELS
from torch.ao.quantization.utils import (is_per_tensor,is_per_channel)


def sync_tensor(tensor):
    """Synchronize the target tensor during distributed training."""
    if torch.distributed.is_initialized() and tensor.is_cuda:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


@MODELS.register_module()
class MSEObserver(BaseObserver):
    """mse observer.

    Calculate mseobserver of whole calibration dataset.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 factory_kwargs=None,
                 p_value=2.0,
                 ch_axis=-1,
                 iter=95,
                 step=0.01,
                 mse_plus=False) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "MSEObserver's qscheme only support \
                    torch.per_tensor_symmetric, torch.per_tensor_affine"
            )
        super(MSEObserver, self).__init__(
            dtype,
            qscheme,
            reduce_range,
            quant_min,
            quant_max,
            factory_kwargs=factory_kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        if (self.qscheme == torch.per_tensor_symmetric and self.reduce_range
                and self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric \
                                       quantization for quint8')
        self.p_value = p_value
        self.ch_axis = ch_axis
        self.iter = iter
        self.step = step
        self.mse_plus =mse_plus

    def lp_loss(self, pred, tgt, dim=None):
        """loss function measured in L_p Norm."""
        return (pred - tgt).abs().pow(self.p_value).mean(dim) if dim else (
            pred - tgt).abs().pow(self.p_value).mean()

    def mse(self,
            x: torch.Tensor,
            x_min: torch.Tensor,
            x_max: torch.Tensor,
            iter=80):
        best_score = 1e+10
        best_min, best_max = torch.tensor(
            [1.0], dtype=torch.float), torch.tensor([1.0], dtype=torch.float)
        best_min.copy_(x_min)
        best_max.copy_(x_max)
        if not self.mse_plus:
            for i in range(iter):
                new_min = x_min * (1.0 - (i * self.step))
                new_max = x_max * (1.0 - (i * self.step))
                scale, zero_point = self._calculate_qparams(new_min, new_max)
                x_q = torch.fake_quantize_per_tensor_affine(
                    x, scale.item(), int(zero_point.item()), self.quant_min,
                    self.quant_max)
                score = self.lp_loss(x_q, x)
                if score < best_score:
                    best_score = score
                    best_min, best_max = new_min, new_max
        else:
            for i in range(iter):
                new_min = x_min * (1.0 - (i * self.step))
                for i in range(iter):
                    new_max = x_max * (1.0 - (i * self.step))
                    scale, zero_point = self._calculate_qparams(new_min, new_max)
                    x_q = torch.fake_quantize_per_tensor_affine(
                        x, scale.item(), int(zero_point.item()), self.quant_min,
                        self.quant_max)
                    score = self.lp_loss(x_q, x)
                    if score < best_score:
                        best_score = score
                        best_min, best_max = new_min, new_max
        return best_min, best_max

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        min_val_cur, max_val_cur = self.mse(
            x, min_val_cur, max_val_cur, iter=self.iter)

        min_val = torch.min(self.min_val, min_val_cur)
        max_val = torch.max(self.max_val, max_val_cur)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point


@MODELS.register_module()
class PerChannelMSEObserver(MSEObserver):
    """MSE per-channel observer.

    Calculate mseperchannelobserver of whole calibration dataset.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 factory_kwargs=None,
                 p_value=2.0,
                 ch_axis=0,
                 iter=80,
                 step=0.01,
                 mse_plus=False) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "PerChannelMSEObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine"
            )
        super(PerChannelMSEObserver, self).__init__(
            dtype,
            qscheme,
            reduce_range,
            quant_min,
            quant_max,
            factory_kwargs=factory_kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        self.p_value = p_value
        self.ch_axis = ch_axis
        self.iter = iter
        self.step = step
        self.mse_plus = mse_plus

    def mse_perchannel(self,
                       x: torch.Tensor,
                       x_min: torch.Tensor,
                       x_max: torch.Tensor,
                       iter=80,
                       ch_axis=0):
        assert x_min.shape == x_max.shape
        assert ch_axis >= 0, f'{ch_axis}'
        best_score = 1e+10 * torch.ones_like(x_min)
        best_min, best_max = x_min.clone(), x_max.clone()
        reduce_dim = tuple([i for i in range(len(x.shape)) if i != ch_axis])
        if not self.mse_plus:
            for i in range(iter):
                new_min = x_min * (1.0 - (i * self.step))
                new_max = x_max * (1.0 - (i * self.step))
                scale, zero_point = self._calculate_qparams(new_min, new_max)
                x_q = torch.fake_quantize_per_channel_affine(
                    x, scale, zero_point.int(), ch_axis, self.quant_min,
                    self.quant_max)
                score = self.lp_loss(x_q, x, reduce_dim)
                update_idx = (score < best_score)
                best_score[update_idx] = score[update_idx]
                best_min[update_idx] = new_min[update_idx]
                best_max[update_idx] = new_max[update_idx]
        else:
            for i in range(iter):
                new_min = x_min * (1.0 - (i * self.step))
                for i in range(iter):
                    new_max = x_max * (1.0 - (i * self.step))
                    scale, zero_point = self._calculate_qparams(new_min, new_max)
                    x_q = torch.fake_quantize_per_channel_affine(
                        x, scale, zero_point.int(), ch_axis, self.quant_min,
                        self.quant_max)
                    score = self.lp_loss(x_q, x, reduce_dim)
                    update_idx = (score < best_score)
                    best_score[update_idx] = score[update_idx]
                    best_min[update_idx] = new_min[update_idx]
                    best_max[update_idx] = new_max[update_idx]
        return best_min, best_max

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        x_channel = x.permute(new_axis_list)
        x_channel = x_channel.to(self.min_val.dtype)
        y = torch.flatten(x_channel, start_dim=1)
        min_val_cur, max_val_cur = torch._aminmax(y, 1)
        min_val_cur, max_val_cur = self.mse_perchannel(
            x, min_val_cur, max_val_cur, iter=self.iter, ch_axis=self.ch_axis)
        min_val = torch.min(min_val, min_val_cur)
        max_val = torch.max(max_val, max_val_cur)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point


@MODELS.register_module()
class EMAMSEObserver(MSEObserver):
    """ema mse observer.

    Calculate EMAMSEObserver of whole calibration dataset.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 factory_kwargs=None,
                 p_value=2.0,
                 averaging_constant =0.1,
                 ch_axis=-1,
                 iter=95,
                 step=0.01,
                 mse_plus=False) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "EMAMSEObserver's qscheme only support \
                    torch.per_tensor_symmetric, torch.per_tensor_affine"
            )
        super(EMAMSEObserver, self).__init__(
            dtype,
            qscheme,
            reduce_range,
            quant_min,
            quant_max,
            factory_kwargs=factory_kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        if (self.qscheme == torch.per_tensor_symmetric and self.reduce_range
                and self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric \
                                       quantization for quint8')
        self.p_value = p_value
        self.averaging_constant = averaging_constant
        self.ch_axis = ch_axis
        self.iter = iter
        self.step = step
        self.mse_plus =mse_plus

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        min_val_cur, max_val_cur = self.mse(
            x, min_val_cur, max_val_cur, iter=self.iter)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = self.min_val * (1.0 - self.averaging_constant) + min_val_cur * self.averaging_constant
            max_val = self.max_val * (1.0 - self.averaging_constant) + max_val_cur * self.averaging_constant
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point


@MODELS.register_module()
class PerChannelEMAMSEObserver(PerChannelMSEObserver):
    """MSE per-channel observer.

    Calculate mseperchannelobserver of whole calibration dataset.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 factory_kwargs=None,
                 p_value=2.0,
                 ch_axis=0,
                 iter=80,
                 step=0.01,
                 averaging_constant =0.1,
                 mse_plus=False) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "PerChannelEMAMSEObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine"
            )
        super(PerChannelEMAMSEObserver, self).__init__(
            dtype,
            qscheme,
            reduce_range,
            quant_min,
            quant_max,
            factory_kwargs=factory_kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        self.p_value = p_value
        self.ch_axis = ch_axis
        self.iter = iter
        self.step = step
        self.averaging_constant = averaging_constant
        self.mse_plus =mse_plus


    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        x_channel = x.permute(new_axis_list)
        x_channel = x_channel.to(self.min_val.dtype)
        y = torch.flatten(x_channel, start_dim=1)
        min_val_cur, max_val_cur = torch._aminmax(y, 1)
        min_val_cur, max_val_cur = self.mse_perchannel(
            x, min_val_cur, max_val_cur, iter=self.iter, ch_axis=self.ch_axis)
        
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = self.min_val * (1.0 - self.averaging_constant) + min_val_cur * self.averaging_constant
            max_val = self.max_val * (1.0 - self.averaging_constant) + max_val_cur * self.averaging_constant
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point
