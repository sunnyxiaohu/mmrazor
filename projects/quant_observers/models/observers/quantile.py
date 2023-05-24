# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.distributed as dist

from mmrazor.models.observers.base import BaseObserver
from mmrazor.registry import MODELS


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
                 ema_ratio=0.9,
                 threshold=0.99999,
                 bins=2048,
                 factory_kwargs=None) -> None:
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
        self.ch_axis = ch_axis
        assert self.ch_axis == -1, 'Quantile observer only support in per-tensor scheme.'
        self.ema_ratio = ema_ratio
        self.threshold = threshold
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
            if cur_total + cnt >= self.threshold * x.numel():
                clip_value = (i + 0.5) * (max_hist_range / self.bins)
                break
            cur_total += cnt

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = max(min_val_cur, -clip_value)
            self.max_val = min(max_val_cur, clip_value)
        else:
            self.min_val = self.min_val * self.ema_ratio + max(
                min_val_cur, -clip_value) * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + min(
                max_val_cur, clip_value) * (1.0 - self.ema_ratio)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point
