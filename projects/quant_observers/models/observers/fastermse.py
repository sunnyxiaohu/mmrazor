# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
import torch
import torch.distributed as dist

from mmrazor.registry import MODELS
from .kl import KLObserver

try:
    from torch.ao.quantization.observer import MinMaxObserver
except ImportError:
    from mmrazor.utils import get_placeholder
    MinMaxObserver = get_placeholder('torch>=1.13')


def sync_tensor(tensor):
    """Synchronize the target tensor during distributed training."""
    if torch.distributed.is_initialized() and tensor.is_cuda:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


@MODELS.register_module()
class FasterMSEObserver(KLObserver):
    """KLObserver collects histogram of given tensor.

    It is designed for per-tensor quantization or activation quantization.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_symmetric,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 factory_kwargs=None,
                 hist_bins=4096,
                 ch_axis=-1) -> None:
        super(FasterMSEObserver, self).__init__(
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
        self._hist_bins = hist_bins
        self.ch_axis = ch_axis

    def compute_mse_loss(self, histogram: list, start: int, step: int,
                         end: int):
        num_of_elements = sum(histogram)
        loss = 0
        for idx, bin in enumerate(histogram):
            if idx < start:
                # 如果所选的 bin 已经超出了起点，那从 bin 的中心到起点的距离即
                # ((idx 到 起点的距离) + 0.5)
                # 注意 hist 统计时是直接取 floor 的，因此会在这里额外 - 1
                error = ((start - idx - 1) + 0.5)
            elif idx > end:
                # 注意 hist 统计时是直接取 floor 的
                error = ((idx - end) + 0.5)
            else:
                # 分别计算左右两侧的 err
                l_idx = (idx - start) % step
                r_idx = step - l_idx - 1
                if l_idx == r_idx:
                    error = (l_idx + 0.25)
                else:
                    l_err = (l_idx + 0.5)
                    r_err = (r_idx + 0.5)
                    error = min(l_err, r_err)
            loss += (bin * error * error) / num_of_elements
        return loss

    def convert_any_to_numpy(self,
                             x: Union[torch.Tensor, np.ndarray, int, float,
                                      list, tuple],
                             accept_none: bool = True) -> np.ndarray:
        if x is None and accept_none: return None
        if x is None and not accept_none:
            raise ValueError('Trying to convert an empty value.')
        if isinstance(x, np.ndarray): return x
        elif isinstance(x, int) or isinstance(x, float): return np.array([
                x,
        ])
        elif isinstance(x, torch.Tensor):
            if x.numel() == 0 and accept_none: return None
            if x.numel() == 0 and not accept_none:
                raise ValueError('Trying to convert an empty value.')
            if x.numel() >= 1: return x.detach().cpu().numpy()
        elif isinstance(x, list) or isinstance(x, tuple):
            return np.array(x)
        else:
            raise TypeError(
                f'input value {x}({type(x)}) can not be converted as numpy type.'
            )

    @torch.jit.export
    def _calculate_qparams(self):
        histogram = self.convert_any_to_numpy(self._hist).tolist()
        num_of_quant_levels = (self.quant_max - self.quant_min) + 1

        losses = []
        if self.qscheme == torch.per_tensor_affine:

            # at least we can have a min-max result
            step = self._hist_bins // num_of_quant_levels + 1
            loss = self.compute_mse_loss(
                histogram=histogram,
                start=0,
                step=step,
                end=num_of_quant_levels * step)
            losses.append({
                'mse': loss,
                'start': 0,
                'end': num_of_quant_levels * step
            })

            for start in range(0, self._hist_bins, 8):
                if (start * self._hist_scale) + self.min_val > 0:
                    break  # start can not > 0

                for step in range(1,
                                  self._hist_bins // num_of_quant_levels + 1):
                    end = start + num_of_quant_levels * step
                    if end > (self._hist_bins + num_of_quant_levels): break
                    loss = self.compute_mse_loss(
                        histogram=histogram, start=start, step=step, end=end)
                    losses.append({'mse': loss, 'start': start, 'end': end})

            best_policy = sorted(losses, key=lambda x: x['mse'])[0]
            best_start = best_policy['start']
            best_end = best_policy['end']

            # translate start & end to scale & zero_point.
            range_min, range_max = (
                best_start * self._hist_scale) + self.min_val, (
                    best_end * self._hist_scale) + self.min_val
            scale, zero_point = MinMaxObserver._calculate_qparams(
                self, min_val=range_min, max_val=range_max)
            return scale, zero_point

        elif self.ch_axis == 0:
            raise NotImplementedError(
                'faster Mse observer do not support PER_CHANNEL policy now, please wait.'
            )

        elif self.qscheme == torch.per_tensor_symmetric:
            # at least we can have a min-max result
            step = self._hist_bins // num_of_quant_levels + 1
            loss = self.compute_mse_loss(
                histogram=histogram,
                start=0,
                step=step,
                end=num_of_quant_levels * step)
            losses.append({'mse': loss, 'end': num_of_quant_levels * step})

            for step in range(1, self._hist_bins // num_of_quant_levels + 1):
                end = num_of_quant_levels * step
                if end > (self._hist_bins + num_of_quant_levels): break
                loss = self.compute_mse_loss(
                    histogram=histogram, start=0, step=step, end=end)
                losses.append({'mse': loss, 'end': end})

            best_policy = sorted(losses, key=lambda x: x['mse'])[0]
            best_end = best_policy['end']

            # translate start & end to scale & zero_point.
            range_min, range_max = -(best_end * self._hist_scale), (
                best_end * self._hist_scale)
            scale, zero_point = MinMaxObserver._calculate_qparams(
                self,
                min_val=torch.tensor(range_min),
                max_val=torch.tensor(range_max))
            return scale, zero_point

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        if self._phase == 'Detecting Minmax':
            if self.qscheme == torch.per_tensor_symmetric:
                hist_range = float(max(abs(self.max_val), abs(self.min_val)))
            else:
                hist_range = self.max_val - self.min_val
            self._hist_scale = hist_range / self._hist_bins
            device = self.min_val.device
            scale = torch.ones(
                self.min_val.size(), dtype=torch.float32, device=device)
            zero_point = torch.zeros_like(self.max_val)
        elif self._phase == 'Collating Hist':
            scale, zero_point = self._calculate_qparams()
            scale.data = sync_tensor(scale).data
            zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point
