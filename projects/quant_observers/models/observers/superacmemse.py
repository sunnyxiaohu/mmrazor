from .fastermse import FasterMSEObserver
import torch
import torch.distributed as dist
from mmrazor.models.observers.base import BaseObserver
from mmrazor.registry import MODELS
from torch.ao.quantization.utils import (is_per_tensor)       
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
class SuperacmeMSEObserver(BaseObserver):
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
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "SuperacmeMSEObserver's qscheme only support \
                    torch.per_tensor_symmetric, torch.per_tensor_affine"
            )
        super(SuperacmeMSEObserver, self).__init__(
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
        self._hist_scale = None
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        if (self.qscheme == torch.per_tensor_symmetric and self.reduce_range
                and self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric \
                                       quantization for quint8')
            
    @torch.jit.export
    def _calculate_qparams(self):
        histogram = FasterMSEObserver.convert_any_to_numpy(self,self._calib_hist).tolist()
        num_of_quant_levels = (self.quant_max - self.quant_min) + 1

        losses = []
        if self.qscheme == torch.per_tensor_affine:

            # at least we can have a min-max result
            step = self._num_bins // num_of_quant_levels + 1
            loss = FasterMSEObserver.compute_mse_loss(self,
                histogram=histogram,
                start=0,
                step=step,
                end=num_of_quant_levels * step)
            losses.append({
                'mse': loss,
                'start': 0,
                'end': num_of_quant_levels * step
            })

            for start in range(0, self._num_bins, 8):
                if (start * self._hist_scale) + self.min_val > 0:
                    break  # start can not > 0

                for step in range(1,
                                  self._num_bins // num_of_quant_levels + 1):
                    end = start + num_of_quant_levels * step
                    if end > (self._num_bins + num_of_quant_levels): break
                    loss = FasterMSEObserver.compute_mse_loss(self,
                        histogram=histogram, start=start, step=step, end=end)
                    losses.append({'mse': loss, 'start': start, 'end': end})

            best_policy = sorted(losses, key=lambda x: x['mse'])[0]
            best_start = best_policy['start']
            best_end = best_policy['end']

            # translate start & end to scale & zero_point.
            range_min, range_max = (best_start * self._hist_scale) + self.min_val, (best_end * self._hist_scale) + self.min_val
            scale, zero_point = MinMaxObserver._calculate_qparams(
                self, min_val=range_min, max_val=range_max)
            return scale, zero_point

        elif self.ch_axis == 0:
            raise NotImplementedError(
                'superacmeMse observer do not support PER_CHANNEL policy now, please wait.'
            )

        elif self.qscheme == torch.per_tensor_symmetric:
            # at least we can have a min-max result
            step = self._num_bins // num_of_quant_levels + 1
            loss = FasterMSEObserver.compute_mse_loss(self,
                histogram=histogram,
                start=0,
                step=step,
                end=num_of_quant_levels * step)
            losses.append({'mse': loss, 'end': num_of_quant_levels * step})

            for step in range(1, self._num_bins // num_of_quant_levels + 1):
                end = num_of_quant_levels * step
                if end > (self._num_bins + num_of_quant_levels): break
                loss = FasterMSEObserver.compute_mse_loss(self,
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

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        with torch.no_grad():
            if self.qscheme == torch.per_tensor_symmetric:
                x = x.abs().float()
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
            else:
                x = x.float()
                x_min =x.min()
                x_max =x.max()
                if self._calib_bin_edges is None and self._calib_hist is None:
                    self._calib_hist = torch.histc(
                        x, bins=self._num_bins, min=x_min, max=x_max)
                    self._calib_bin_edges = torch.linspace(
                        x_min, x_max, self._num_bins + 1, device=x.device)
                    self.min_val =self._calib_bin_edges[0]
                else:
                    if x_max > self._calib_bin_edges[-1]:
                        width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                        if x_min < self._calib_bin_edges[0]:
                            self._num_bins = int(((x_max-x_min) / width).ceil().item())
                            self._calib_bin_edges = torch.arange(
                                x_min-width, x_max + width, width, device=x.device)
                        else:
                            self._num_bins = int(((x_max-self._calib_bin_edges[0]) / width).ceil().item())
                            self._calib_bin_edges = torch.arange(
                                self._calib_bin_edges[0], x_max + width, width, device=x.device)
                            self.min_val =self._calib_bin_edges[0]
                    else:
                        if x_min <self._calib_bin_edges[0]:
                            width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                            self._num_bins = int(((self._calib_bin_edges[-1]-x_min) / width).ceil().item())
                            self._calib_bin_edges = torch.arange(
                                x_min-width, self._calib_bin_edges[-1], width, device=x.device)
                    hist = torch.histc(
                        x,
                        bins=self._num_bins,
                        min=self._calib_bin_edges[0],
                        max=self._calib_bin_edges[-1])
                    hist[:self._calib_hist.numel()] += self._calib_hist
                    self._calib_hist = hist
            if self.qscheme == torch.per_tensor_symmetric:
                hist_range = x_max
            else:
                hist_range = self._calib_bin_edges[-1] - self._calib_bin_edges[0]
            self._hist_scale = hist_range / self._num_bins
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams()
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point