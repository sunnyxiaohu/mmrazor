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
class EMAQuantileObserver(BaseObserver):
    """EMAQuantile observer.

    Calculate EMAQuantileObserver of whole calibration dataset.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 ch_axis=-1,
                 averaging_constant=0.1,
                 _percentile=0.9999,
                 bins=2048,
                 factory_kwargs=None) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "EMAQuantileObserver's qscheme only support \
                    torch.per_tensor_symmetric, torch.per_tensor_affine"
            )
        super(EMAQuantileObserver, self).__init__(
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
        if (self.qscheme == torch.per_tensor_affine):
            raise NotImplementedError('NotImplemente for asymmetric')
        self.ch_axis = ch_axis
        assert self.ch_axis == -1, 'Quantile observer only support in per-tensor scheme.'
        self.averaging_constant = averaging_constant
        self._percentile = _percentile
        self.bins = bins

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        max_hist_range = torch.max(-min_val_cur, max_val_cur)
        hist = torch.histc(
            torch.abs(x), bins=self.bins, min=0., max=max_hist_range)
        cur_total = 0
        clip_value = max_hist_range
        for i, cnt in enumerate(hist):
            if cur_total + cnt >= self._percentile * x.numel():
                clip_value = (i + 0.5) * (max_hist_range / self.bins)
                break
            cur_total += cnt

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = max(min_val_cur, -clip_value)
            self.max_val = min(max_val_cur, clip_value)
        else:
            self.min_val = self.min_val * (1.0 - self.averaging_constant) + max(
                min_val_cur, -clip_value) * self.averaging_constant
            self.max_val = self.max_val * (1.0 - self.averaging_constant) + min(
                max_val_cur, clip_value) * self.averaging_constant
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point



@MODELS.register_module()
class EMAPPQQuantileObserver(BaseObserver):
    """EMAPPQQuantileObserver observer.

    Calculate EMAPPQQuantileObserverObserver of whole calibration dataset.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 ch_axis=-1,
                 averaging_constant=0.1,
                 _percentile=0.9999,
                 factory_kwargs=None) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "EMAPPQQuantileObserver's qscheme only support \
                    torch.per_tensor_symmetric, torch.per_tensor_affine"
            )
        super(EMAPPQQuantileObserver, self).__init__(
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
        self.ch_axis = ch_axis
        assert self.ch_axis == -1, 'Quantile observer only support in per-tensor scheme.'
        self.averaging_constant = averaging_constant
        self._percentile = _percentile

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        numel = x.numel()
        min_idx, max_idx = int(numel * (1 - self._percentile)), int(numel * (self._percentile))
        # torch.kthvalue needs index from 1 to numel ...
        min_idx = max(0, min_idx) + 1
        max_idx = min(max_idx, numel - 1) + 1
        min_val_cur = torch.kthvalue(x.flatten(), k = min_idx, dim=0)[0]
        max_val_cur = torch.kthvalue(x.flatten(), k = max_idx, dim=0)[0]
        if min_val == float("inf") and max_val == float("-inf"):
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
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
class PerChannelEMAPPQQuantileObserver(BaseObserver):
    """EMAPPQQuantileObserver observer.

    Calculate EMAPPQQuantileObserverObserver of whole calibration dataset.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_channel_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 ch_axis=0,
                 averaging_constant=0.1,
                 _percentile=0.9999,
                 factory_kwargs=None) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "PerChannelEMAPPQQuantileObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine"
            )
        super(PerChannelEMAPPQQuantileObserver, self).__init__(
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
        self.ch_axis = ch_axis
        assert self.ch_axis == 0, 'Quantile observer only support in per-channel scheme.'
        self.averaging_constant = averaging_constant
        self._percentile = _percentile

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        x_channel = x.permute(new_axis_list)
        x_channel = x_channel.to(self.min_val.dtype)
        y = torch.flatten(x_channel, start_dim=1)
        numel = y.shape[1]
        min_idx, max_idx = int(numel * (1 - self._percentile)), int(numel * (self._percentile))
        min_idx = max(0, min_idx) + 1
        max_idx = min(max_idx, numel - 1) + 1
        min_val_cur = torch.kthvalue(y, k = min_idx, dim=1)[0]
        max_val_cur = torch.kthvalue(y, k = max_idx, dim=1)[0]
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * (1.0 - self.averaging_constant) + min_val_cur * self.averaging_constant
            self.max_val = self.max_val * (1.0 - self.averaging_constant) + max_val_cur * self.averaging_constant
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point