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
class FasterSuperacmeMSEObserver(BaseObserver):
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
                "FasterSuperacmeMSEObserver's qscheme only support \
                    torch.per_tensor_symmetric, torch.per_tensor_affine"
            )
        super(FasterSuperacmeMSEObserver, self).__init__(
            dtype,
            qscheme,
            reduce_range,
            quant_min,
            quant_max,
            factory_kwargs=factory_kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self._num_bins = num_bins
        self._num_bins_neg = num_bins
        self.ch_axis = ch_axis
        self._calib_bin_edges = None
        self._calib_hist = None
        self._calib_bin_edges_neg = None
        self._calib_hist_neg = None
        self._hist_scale = None
        self._hist_scale_neg = None
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        if (self.qscheme == torch.per_tensor_symmetric and self.reduce_range
                and self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric \
                                       quantization for quint8')
    
    def get_best(self,histogram,_num_bins):
        losses = []
        num_of_quant_levels = (self.quant_max - self.quant_min) + 1
        step = _num_bins // num_of_quant_levels + 1
        loss = FasterMSEObserver.compute_mse_loss(self,
            histogram=histogram,
            start=0,
            step=step,
            end=num_of_quant_levels * step)
        losses.append({'mse': loss, 'end': num_of_quant_levels * step})

        for step in range(1, _num_bins // num_of_quant_levels + 1):
            end = num_of_quant_levels * step
            if end > (_num_bins + num_of_quant_levels): break
            loss = FasterMSEObserver.compute_mse_loss(self,
                histogram=histogram, start=0, step=step, end=end)
            losses.append({'mse': loss, 'end': end})

        best_policy = sorted(losses, key=lambda x: x['mse'])[0]
        best_end = best_policy['end']
        return best_end
            
    @torch.jit.export
    def _calculate_qparams(self):
        histogram = FasterMSEObserver.convert_any_to_numpy(self,self._calib_hist).tolist()
        num_of_quant_levels = (self.quant_max - self.quant_min) + 1

        losses = []
        if self.qscheme == torch.per_tensor_affine:
            histogram_neg = FasterMSEObserver.convert_any_to_numpy(self,self._calib_hist_neg).tolist()
            best_end = self.get_best(histogram=histogram,_num_bins = self._num_bins)
            best_start = self.get_best(histogram=histogram_neg,_num_bins = self._num_bins_neg)

            # translate start & end to scale & zero_point.
            range_min, range_max = -(best_start * self._hist_scale_neg), (best_end * self._hist_scale)
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
                x_pos=x[x>0]
                x_neg=x[x<=0]
                x_neg_max =x_neg.abs().max()
                x_pos_max =x_pos.max()
                if self._calib_bin_edges is None and self._calib_hist is None:
                    self._calib_hist = torch.histc(
                        x_pos, bins=self._num_bins, min=0, max=x_pos_max)
                    self._calib_bin_edges = torch.linspace(
                        0, x_pos_max, self._num_bins + 1, device=x.device)
                else:
                    if x_pos_max > self._calib_bin_edges[-1]:
                        width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                        self._num_bins = int((x_pos_max / width).ceil().item())
                        self._calib_bin_edges = torch.arange(
                            0, x_pos_max + width, width, device=x.device)

                    hist = torch.histc(
                        x_pos,
                        bins=self._num_bins,
                        min=0,
                        max=self._calib_bin_edges[-1])
                    hist[:self._calib_hist.numel()] += self._calib_hist
                    self._calib_hist = hist
                    
                if self._calib_bin_edges_neg is None and self._calib_hist_neg is None:
                    self._calib_hist_neg = torch.histc(
                        x_neg.abs(), bins=self._num_bins_neg, min=0, max=x_neg_max)
                    self._calib_bin_edges_neg = torch.linspace(
                        0, x_neg_max, self._num_bins_neg + 1, device=x.device)
                else:
                    if x_neg_max > self._calib_bin_edges_neg[-1]:
                        width = self._calib_bin_edges_neg[1] - self._calib_bin_edges_neg[0]
                        self._num_bins_neg = int((x_neg_max / width).ceil().item())
                        self._calib_bin_edges_neg = torch.arange(
                            0, x_neg_max + width, width, device=x.device)

                    hist = torch.histc(
                        x_neg.abs(),
                        bins=self._num_bins_neg,
                        min=0,
                        max=self._calib_bin_edges_neg[-1])
                    hist[:self._calib_hist_neg.numel()] += self._calib_hist_neg
                    self._calib_hist_neg = hist
            if self.qscheme == torch.per_tensor_symmetric:
                hist_range = x_max
                self._hist_scale = hist_range / self._num_bins
            else:
                hist_range_pos = x_pos_max
                hist_range_neg = x_neg_max
                self._hist_scale = hist_range_pos / self._num_bins
                self._hist_scale_neg = hist_range_neg / self._num_bins_neg
            
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams()
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point