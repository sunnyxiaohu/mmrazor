# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.distributed as dist

from mmrazor.models.observers.base import BaseObserver
from mmrazor.registry import MODELS

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
class KLObserver(MinMaxObserver):
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
        super(KLObserver, self).__init__(
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
        self._phase = 'Detecting Minmax'
        self._hist = None
        self._hist_scale = None

    def forward(self, x_orig):
        if self._phase == 'Detecting Minmax':
            return super().forward(x_orig)
        elif self._phase == 'Collating Hist':
            x = x_orig.clone().detach().to(self.min_val.dtype)
            if self._hist is None:
                self._hist = torch.zeros(
                    size=(self._hist_bins, ),
                    dtype=torch.int32,
                    device=x.device)

            if self.qscheme == torch.per_tensor_affine:
                # ASYMMETRICAL Hist
                hist = torch.histc(
                    x, self._hist_bins, min=self.min_val, max=self.max_val)
                self._hist += hist.int()

            elif self.qscheme == torch.per_tensor_symmetric:
                # SYMMETRICAL Hist
                hist = torch.histc(
                    torch.abs(x),
                    self._hist_bins,
                    min=0,
                    max=self._hist_scale * self._hist_bins)
                self._hist += hist.int()

            else:
                raise TypeError(
                    'Quantization Property is invalid, '
                    'expect either ASYMMETRICAL or SYMMETRICAL config here.')
        return x_orig

    def torch_KL_divergence(self,
                            hist: torch.Tensor,
                            ref_hist: torch.Tensor,
                            eps=1e-30) -> float:
        if hist.ndim != 1 or ref_hist.ndim != 1:            raise ValueError(
                'Only 1 dimension tensor can compute KL divergence with another tensor. '\
f'While your input hist has dimension {hist.ndim} and ref_hist has dimension {ref_hist.ndim}')
        if len(hist) != len(ref_hist):
            raise ValueError(
                'Can not compute KL divergence, len(hist) != len(ref_hist')

        # here we compute KL divergence at float64 precision, make sure your hist and ref_hist are stored at cpu.
        # otherwise it might be very slow.
        return torch.dot(
            hist.double(),
            torch.log10(hist.double() + eps) -
            torch.log10(ref_hist.double() + eps)).item()

    @torch.jit.export
    def _calculate_qparams(self):
        histogram = self._hist.to('cpu').float()
        losses, quant_bins = [], self.quant_max + 1

        # following code is curcial, do not remove
        histogram[:int(self._hist_bins * .002)] = 0
        histogram[int(self._hist_bins * .002)] = 1

        hist_sum = torch.sum(histogram)
        for bin_range in range(quant_bins, self._hist_bins + quant_bins - 1,
                               quant_bins):
            p_hist = torch.zeros(
                size=(bin_range, ), dtype=torch.float, device='cpu')
            p_hist[:bin_range].copy_(histogram[:bin_range])
            p_hist[bin_range - 1] += torch.sum(histogram[bin_range:])
            p_hist = p_hist / hist_sum

            expand_ratio = int(bin_range / quant_bins)
            q_hist = histogram[:bin_range].clone()
            q_hist = q_hist.reshape((quant_bins, expand_ratio))
            positive_map = q_hist > 0
            positive_cnt = positive_map.sum(axis=1, keepdim=True)
            positive_cnt[positive_cnt == 0] = 1
            q_hist = torch.div(q_hist.sum(axis=1, keepdim=True), positive_cnt)
            q_hist = q_hist.repeat([1, expand_ratio])
            q_hist = q_hist * positive_map
            q_hist = q_hist / torch.sum(q_hist)
            q_hist = q_hist.flatten()

            losses.append({
                'kl': self.torch_KL_divergence(p_hist, q_hist),
                'bin_range': bin_range
            })

        best_bin_range = sorted(losses, key=lambda x: x['kl'])[0]['bin_range']
        scale, zero_point = (
            best_bin_range / self._hist_bins) * self._hist_scale * (
                self._hist_bins / quant_bins), torch.zeros_like(self.max_val)

        scale = max(scale, 1e-8)
        scale = torch.tensor(scale)
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
            if self.qscheme == torch.per_tensor_affine:
                raise NotImplementedError(
                    'KL observer is not designed for asymmetrical quantization'
                )
            scale, zero_point = self._calculate_qparams()
            scale.data = sync_tensor(scale).data
            zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point
