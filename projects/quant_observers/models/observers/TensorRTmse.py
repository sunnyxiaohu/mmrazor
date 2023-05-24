# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
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
class TensorRTHistogramBasedMSEObserver(BaseObserver):
    """mse observer.

    Calculate tensorrt mseobserver of whole calibration dataset.
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
                 num_bins=2048,
                 ch_axis=-1) -> None:
        super(TensorRTHistogramBasedMSEObserver, self).__init__(
            dtype,
            qscheme,
            reduce_range,
            quant_min,
            quant_max,
            factory_kwargs=factory_kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self._num_bins = num_bins
        self.ch_axis = ch_axis
        self._calib_bin_edges = None
        self._calib_hist = None
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        if (self.qscheme == torch.per_tensor_symmetric and self.reduce_range
                and self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric \
                                       quantization for quint8')

    def _compute_amax_mse(self,
                          calib_hist,
                          calib_bin_edges,
                          stride=1,
                          start_bin=128):
        """Returns amax that minimizes MSE of the collected histogram."""

        # If calibrator hasn't collected any data, return none
        if calib_bin_edges is None and calib_hist is None:
            return None

        counts = calib_hist[:].float()
        edges = calib_bin_edges[:].float()
        centers = (edges[1:] + edges[:-1]) / 2

        mses = []
        arguments = []

        for i in range(start_bin, len(centers), stride):

            amax = centers[i]
            # quant_centers = fake_tensor_quant(centers, amax, num_bits, unsigned)
            scale = amax / self.quant_max
            zero_point = torch.tensor(0, dtype=torch.int32)
            quant_centers = torch.fake_quantize_per_tensor_affine(
                centers, scale.item(), int(zero_point.item()), self.quant_min,
                self.quant_max)
            mse = ((quant_centers - centers)**2 * counts).mean()

            mses.append(mse)
            arguments.append(i)

        argmin = np.argmin([x.item() for x in mses])
        calib_amax = centers[arguments[argmin]]

        return calib_amax

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        x = x.abs().float()
        with torch.no_grad():
            x_max = x.max()
            if self._calib_bin_edges is None and self._calib_hist is None:
                self._calib_hist = torch.histc(
                    x, bins=self._num_bins, min=0, max=x_max)
                self._calib_bin_edges = torch.linspace(
                    0, x_max, self._num_bins + 1, device=x.device)
            else:
                if x_max > self._calib_bin_edges[-1]:
                    width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                    self._num_bins = int((x_max / width).ceil().item())
                    self._calib_bin_edges = torch.arange(
                        0, x_max + width, width, device=x.device)

                hist = torch.histc(
                    x,
                    bins=self._num_bins,
                    min=0,
                    max=self._calib_bin_edges[-1])
                hist[:self._calib_hist.numel()] += self._calib_hist
                self._calib_hist = hist
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        self.max_val = self._compute_amax_mse(self._calib_hist,
                                              self._calib_bin_edges)
        scale = self.max_val / self.quant_max
        zero_point = torch.tensor(0, dtype=torch.int32)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point
